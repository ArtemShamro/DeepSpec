# Design

## Context

See proposal.md (Why) and the two spec deltas for the requirements. These facts about the existing code shape the approach:

- `generate_decoding_sample` (`deepspec/eval/base_evaluator.py`) is the single speculative-decoding loop for all methods. It takes the algorithm as callbacks: `init_context`, `propose`, `update` and an optional `post_verify`. The calls happen in a fixed order:
  - target prefill → `init_context` → per step: `propose` → `verify_draft_tokens` → `post_verify` → commit and KV crop → `update`;
  - the final stop-token step skips `update`.
- A proposal with `draft_token_count == 0` is already supported by `verify_draft_tokens`.
- Both evaluator families size everything from the `max_proposal_tokens` property:
  - DSpark/DFlash: the draft block length passed to `forward_dspark_draft_block` / `build_dspark_proposal`, the output buffer, per-position metrics;
  - EAGLE-3: the number of chain steps.
  - Nothing on the inference path reads `model.block_size`.
- `BaseEvaluator.run_dataset` owns dataset loading, the seeded shuffle and subsample, per-sample `seed_all(seed + idx)`, the chat template (`enable_thinking=False`) and rank striding. It returns the `SimpleNamespace` objects produced by `generate_one_sample` in order.
- `BaseEvaluator.__init__` calls `init_dist`, which creates an NCCL process group. Only one evaluator can therefore exist per process.
- The machine has one RTX 3090 (24 GB). The Python environment and model weights are not yet installed.

## Goals / Non-Goals

**Goals:**
- Measure the existing DeepSpec code paths as they are, with no edits to `deepspec/`.
- Make per-step timing cheap enough that it does not distort the decode loop, and verify that.
- Keep every run reproducible from a config file plus the run manifest.

**Non-Goals:**
- Analysis notebook, long-context synthetic memory sweep.
- Tree drafting for EAGLE-3, CUDA graphs, `torch.compile`, or any other change to how DeepSpec executes models.
- Lossless-output checks against the AR baseline. At temperature 1.0 outputs legitimately differ.
- Multi-node runs. Multiple local GPUs keep working via DeepSpec's rank striding, but the target setup is one GPU.

## Decisions

### D1. Instrument by wrapping the loop callbacks inside `experiments/`

An instrumentation mixin overrides `generate_one_sample`. It calls the unchanged `generate_decoding_sample` with wrapped versions of the evaluator's own `_init_context`, `_propose`, `_update` and `_post_verify` (or a no-op when the evaluator has none, as EAGLE-3 does). The wrapper boundaries map exactly to the phases the specs require:

```
sample start ─ prefill ─▶ init_context ─▶ │ propose │ verify ─▶ post_verify ─ commit ─▶ │ update │ ─ stop check ─▶ next step
            [target prefill] [init_context] [ draft ] [ verification ] [   commit     ] [ update ] [  commit   ]
```

Consecutive phases share boundary events, so the phases tile the sample's GPU-stream timeline:

- Target prefill = sample start → `init_context` entry.
- Init context = `init_context` duration. Step 0's draft starts at its exit; only the loop condition lies between.
- Draft = `propose` duration.
- Verification = `post_verify` entry − `propose` exit. `post_verify` is called immediately after `verify_draft_tokens`.
- Update = `update` duration (EAGLE-3 cache extension, DSpark context-feature extraction); 0 for the final stop-token step, which skips `update`.
- Commit = everything else in the step: `post_verify` entry → `update` entry (inner `post_verify` hook, token commit, KV crop) plus `update` exit → next `propose` entry or sample end (stop-token check; output trimming after the final step).

Sample-level draft / verification / commit / update are the direct sums of the step values, in step order. Decode time and per-token latencies are derived in post-processing (`Σ` of the four step phases).

*Alternatives:*
- Add a `step_hook=None` argument to `generate_decoding_sample`. This is a small upstream-compatible edit, kept as the fallback if wrapping proves insufficient.
- Copy the loop into `experiments/`. Rejected: duplicated logic drifts from DeepSpec and breaks the "measure DeepSpec as it is" goal.

### D2. GPU events for per-step timing; synchronize only at sample boundaries

Each boundary records a `torch.cuda.Event(enable_timing=True)` on the current stream. Recording does not block. At sample end the device is synchronized once and all elapsed times are resolved.

Why this matters: DSpark/DFlash `propose` contains no host sync, so the host normally queues verification kernels while draft kernels still run. A `synchronize()` per boundary would remove that overlap and inflate latency.

Event intervals measure the GPU-stream timeline, including idle gaps caused by host-side launch overhead. That is the latency users experience.

Per-sample target prefill and init-context durations come from the same events. A wall-clock total per sample (`perf_counter` with `synchronize()` at start and end) is stored as a cross-check.

A `step_timing` switch disables the per-step events and keeps the sample-level events (start, prefill end, init-context end, end). The step phases and their sums are then `null`; instrumentation overhead is measured by comparing per-sample `wall_ms` between the two settings.

*Alternative:* `synchronize()` + `perf_counter` at every boundary. Simpler, but changes execution overlap. It may still be useful as a debug mode.

### D3. Accounting data comes from DeepSpec's own lists; wrappers add only timing and drafted length

Per-step accepted tokens, effective draft length and committed tokens come from the returned `accepted_draft_lengths`, `proposal_lengths` and `acceptance_lengths`, aligned by step index. These are the exact inputs to `build_metrics_row`, which guarantees the parity requirement by construction.

The `post_verify` wrapper adds the drafted length (`proposal.draft_token_count`). Requested length is the run's `K` (0 for AR).

### D4. `K` via a `max_proposal_tokens` override

Thin subclasses of `Qwen3DSparkEvaluator` and `Qwen3Eagle3Evaluator` return the configured `K` from `max_proposal_tokens`, reading it from a mutable attribute. Changing `K` between runs inside one process therefore needs no model reload.

- DSpark/DFlash: the drafter runs on a `K`-sized block. This is the approved "smaller block" variant.
- EAGLE-3: `K` chain steps.
- Validation: `1 ≤ K ≤ trained length`, where trained length is `draft_model.block_size` or `draft_model.ttt_length`. Violations fail before decoding.
- `dflash` and `dspark` use the same subclass. The method name is a label tied to its checkpoint.

The DSpark subclass also overrides `_build_confidence_head_recorder` to return `None`. `confidence_threshold` stays at DeepSpec's default 0.0, so proposals keep DeepSpec's default full-length behavior. The confidence head forward that `build_dspark_proposal` runs whenever a head exists is algorithm behavior and is left untouched.

### D5. AR baseline = `BaseEvaluator` subclass proposing zero draft tokens

- `build_models` loads only the target and tokenizer, with the same dtype and `sdpa` as the other evaluators. It returns `None` as the draft model.
- `max_proposal_tokens = 1`, because the loop asserts `≥ 1` and sizes the buffer from it.
- `propose` returns `DraftProposal(0, output_ids[:, start:start+1], None)`.
- `init_context` returns `None`; `update` is a no-op.

The baseline keeps `output_hidden_states=True` on target forwards, like every method. This is a known, uniform overhead (see Risks).

### D6. One process per method, spawned like `eval.py`

The parent process does three things:
1. loads and validates the config (fails fast on unknown methods or invalid `K` where it can without loading models);
2. writes the manifest;
3. for each method in order, runs `torch.multiprocessing.spawn(worker, nprocs=cuda_device_count)` and waits.

Each worker:
- builds one evaluator, which calls `init_dist` once;
- loops `K` (only once for AR) and datasets: warm-up, then measured run, then write records;
- calls `clean_up()` and exits.

This satisfies process isolation and avoids double `init_process_group`.

*Alternative:* one process per (method, K). Cleaner allocator state, but reloads the 8 GB target for every `K`. Peak memory is reset per sample anyway (D8), so per-method isolation is enough.

### D7. Reuse `run_dataset` for selection, seeding and warm-up

- **Measured run:** `run_dataset(dataset, max_samples)` unchanged. Sample indices are reconstructed as `range(rank, N, world_size)`, matching its striding.
- **Warm-up:** `run_dataset(dataset, max_samples=W)`, results discarded. The seeded shuffle makes these the first `W` prompts of the measured selection. Each sample re-seeds with `seed_all(seed + idx)`, so warm-up cannot perturb measured outputs.
- Warm-up runs per (method, K, dataset), so shapes and kernels for that dataset's typical lengths are warm.

### D8. Memory: host-side allocator statistics

- `reset_peak_memory_stats()` at sample start.
- At the `init_context` wrapper's exit (end of init context): read `max_memory_allocated()` as the prefill peak, then reset.
- At sample end: read the decode peak.

The CUDA caching allocator updates these statistics when work is enqueued, so reading them needs no sync. Once per worker, after models load, record `memory_allocated()` and target and draft parameter counts.

### D9. Storage

Run directory: `experiments/results/<UTC timestamp>_<config stem>/`.

| File | Written by | When | Contents |
|---|---|---|---|
| `manifest.json` | parent | at start | resolved config, config path + overrides, git sha + dirty flag, Python/torch/transformers/CUDA versions; `workers/<method>.rank<r>.json` add GPU name, parameter counts, post-load memory |
| `samples.<method>.rank<r>.jsonl` | each worker | flushed after every sample | one line per sample |
| `steps.<method>.rank<r>.jsonl` | each worker | flushed after every sample | one line per step |
| `summary.jsonl` | rank 0 | after each (method, dataset, K) | DeepSpec metrics row from the evaluator's own `allreduce_response_metrics` + `build_metrics_row` on the measured responses |

JSONL keeps partial results readable after an interruption. All files are under the git-ignored `experiments/results/`.

### D10. Config format

Python config files loaded with DeepSpec's `load_config` / `parse_opts_to_config`, with `--opts key=value` overrides, as `train.py` does. Fields:
- `target_name_or_path`;
- `methods`: a list of `{name, draft_name_or_path}`;
- `datasets`: a list of `(name, max_samples)`;
- `draft_lengths`;
- `generation`: `temperature`, `max_new_tokens`, `seed`;
- `warmup_samples`, `step_timing`, `output_root`.

Two configs:
- `experiments/configs/qwen3_4b_main.py`: the full matrix with DeepSpec's sample counts (gsm8k 500, math500 500, aime25 30);
- `experiments/configs/qwen3_4b_smoke.py`: all four methods, `K=3`, a few samples per dataset, small `max_new_tokens`.

### D11. Tests

pytest tests live in `experiments/tests/`:
- **CPU unit tests:** metric recomputation vs `build_metrics_row` on synthetic step lists, including stop-token steps; record building and length consistency; config validation; the K-bounds check.
- **GPU tests**, marked and skipped without CUDA:
  - wrapper timing math on a toy callback sequence;
  - instrumented vs plain evaluator equality on a few samples;
  - the smoke sweep end to end.

## Risks / Trade-offs

- **[Timings are specific to the HF eager stack]** bsz=1, `sdpa`, `DynamicCache`, no CUDA graphs: per-token latency includes substantial launch and Python overhead, so absolute speedups will not transfer to vLLM/SGLang. → Record the stack in the manifest and state it in the report. Relative comparison within the stack stays valid.
- **[Event timing misses host-only gaps before the first event]** The step phases tile the timeline between events, but host-only time before the sample-start event is not on the GPU stream. → Per-sample wall-clock cross-check against the sum of all phases, analyzed in the notebook.
- **[Reduced-K block drafters run off-distribution]** Trained at block 7, run at 3/5. This is an intended part of the study. → The manifest records trained length and `K`, so plots can label it.
- **[EAGLE-3 is chain-only here]** → Label in the report. Not changed.
- **[Uniform `output_hidden_states=True` overhead]** The prompt keeps all 37 hidden states, about 190 KB per token. This inflates the prefill peak for every method including AR. → Prefill and decode peaks are separate. "Additional memory" is computed from the decode peak.
- **[GPU nondeterminism could break bitwise equality instrumented vs plain]** → Events do not change kernels, so equality is expected. If a mismatch appears, re-check at temperature 0 and document before relaxing the test.
- **[Full sweep runtime on one 3090 is long]** 1030 samples × up to 2048 tokens × 10 (method, K) configurations, estimated at tens of hours, not yet measured. → Sample counts are config-overridable. The smoke config gives quick verification. JSONL flushing makes interrupted runs salvageable.
- **[Temperature 1.0 makes methods generate different texts]** Generated lengths and contexts differ per method. → Compare per-token metrics. Per-sample seeds stay identical across methods.

## Migration Plan

Additive only. Nothing in `deepspec/`, `eval.py`, `config/` or `scripts/` changes. Rollback = delete `experiments/` and the `.gitignore` line.
