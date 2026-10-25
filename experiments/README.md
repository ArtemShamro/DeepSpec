# Experiments: speculative decoding sweeps

Infrastructure for the comparison described in [EXPERIMENTS.md](../EXPERIMENTS.md): the
autoregressive Qwen3-4B baseline, EAGLE-3, DFlash and DSpark on GSM8K / Math500 / AIME25 at
draft lengths `K ∈ {3, 5, 7}`, with per-sample and per-step timing, memory and acceptance records.

The runner reuses DeepSpec's evaluators, dataset loading, sample selection, seeding and
decoding loop (`deepspec/eval/base_evaluator.py::generate_decoding_sample`) unchanged. Nothing
under `deepspec/`, `eval.py`, `config/` or `scripts/` is modified.

## Environment

```bash
uv venv .venv --python 3.12          # any venv tool works
uv pip install --python .venv/bin/python -r requirements.txt pytest
source .venv/bin/activate
python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"
# 2.9.1+cu128 5.10.2 True

hf download Qwen/Qwen3-4B
hf download deepseek-ai/eagle3_qwen3_4b_ttt7
hf download deepseek-ai/dflash_qwen3_4b_block7
hf download deepseek-ai/dspark_qwen3_4b_block7
```

## Running

All commands run from the repository root with the venv activated.

```bash
# Smoke check: 4 methods, K=3, 3 samples per dataset, 128 new tokens (~4 min on one RTX 3090)
python -m experiments.run_sweep --config experiments/configs/qwen3_4b_smoke.py

# Full matrix
python -m experiments.run_sweep --config experiments/configs/qwen3_4b_main.py

# Overrides use DeepSpec's --opts syntax (YAML values); they are recorded in the manifest
python -m experiments.run_sweep --config experiments/configs/qwen3_4b_main.py \
    --opts 'methods=[{name: dspark, draft_name_or_path: deepseek-ai/dspark_qwen3_4b_block7}]' \
    --opts 'datasets=[[gsm8k, 10]]' --opts 'draft_lengths=[7]' --opts step_timing=false

# Check that summary.jsonl is reproduced exactly from the step records
python -m experiments.metrics experiments/results/<run_id>
```

Like `eval.py`, the runner starts one worker per visible GPU (`CUDA_VISIBLE_DEVICES`); samples
are strided across ranks exactly as in DeepSpec. Each method runs in its own spawned process,
after the previous method's process has exited.

The config is validated before any model is loaded: unknown methods, `K < 1`, and `K` above the
checkpoint's trained draft length (read from its `config.json`) are rejected.

### Config fields

| Field | Meaning | Default |
|---|---|---|
| `target_name_or_path` | target model | required |
| `methods` | list of `{name, draft_name_or_path}`; `name ∈ {ar, eagle3, dflash, dspark}` | required |
| `datasets` | list of `(name, max_samples)`; names are files in `eval_datasets/` | required |
| `draft_lengths` | list of `K` | required |
| `generation.temperature` / `max_new_tokens` / `seed` | fixed for every method and `K` | 1.0 / 2048 / 980406 (as `eval.py`) |
| `generation.enable_thinking` | chat template mode; DeepSpec always uses `False` | `False` |
| `warmup_samples` | warm-up samples per (method, K, dataset), discarded | 3 |
| `step_timing` | per-step CUDA events on/off (per-sample timing is always on) | `True` |
| `output_root` | results location (git-ignored) | `experiments/results` |

## K and AR semantics

- **EAGLE-3**: `K` chained draft steps (chain only, no tree). The chain stops early when the drafter
  samples a stop token.
- **DFlash / DSpark**: the drafter runs on a block of `K` positions (anchor + `K-1` mask slots) and
  proposes `K` draft tokens. The released checkpoints are trained with block 7, so `K = 3, 5` run
  the drafter off its training block size. This is an intended part of the study.
- **DSpark confidence head**: DeepSpec's calibration recorder is disabled; the threshold stays at
  DeepSpec's default `0.0`, so proposals are always full length. The confidence-head forward that
  `build_dspark_proposal` runs is algorithm code and is left as is.
- **AR baseline (`ar`)**: target model only. Every step proposes zero draft tokens and goes through
  the same `verify_draft_tokens` (target forward, sampling, stop-token handling, KV crop) as the
  speculative methods. It runs once per dataset with `k = 0`, whatever `draft_lengths` lists.
- All methods keep `output_hidden_states=True` on target forwards, as DeepSpec does.

Warm-up reuses `run_dataset(max_samples=W)`. Every sample is re-seeded with `seed + idx`, so
warm-up cannot change the measured outputs (this is tested).

## Output layout

`experiments/results/<UTC timestamp>_<config stem>/`:

| File | Written | Contents |
|---|---|---|
| `manifest.json` | at start, by the parent | resolved config, config path, `opts`, git commit and `dirty` flag, Python/torch/transformers/CUDA/cuDNN versions, trained draft lengths, start time |
| `workers/<method>.rank<r>.json` | after model loading | GPU name, target/draft parameter counts, allocated memory after loading |
| `samples.<method>.rank<r>.jsonl` | flushed after every sample | one line per measured sample |
| `steps.<method>.rank<r>.jsonl` | flushed after every sample | one line per speculative step |
| `summary.jsonl` | rank 0, after each (method, dataset, K) | `metrics` (accepted-token metrics, below) and `deepspec_metrics` (DeepSpec's own `build_metrics_row`) |

### Sample record

| Field | Meaning |
|---|---|
| `method`, `dataset`, `k`, `sample_idx`, `rank` | sample key; `k = 0` for `ar` |
| `prompt_sha1` | hash of the prompt token ids (identical across methods for the same `sample_idx`) |
| `prompt_len`, `generated_len` | tokens |
| `num_steps` | speculative steps (verifications) |
| `stop_reason` | `stop_token` or `length` (hit `max_new_tokens`) |
| `target_prefill_ms` | prompt forward + first-token sampling |
| `init_context_ms` | the method's state initialization from the prompt after target prefill (`init_context`; EAGLE-3 drafter prompt pass, DSpark/DFlash context features) |
| `draft_ms`, `verification_ms`, `commit_ms`, `update_ms` | sums of the step values below, in step order |
| `wall_ms` | host wall clock for the whole sample, device synchronized at start and end (cross-check) |
| `prefill_peak_bytes` | peak allocated memory during target prefill + `init_context` |
| `decode_peak_bytes` | peak allocated memory during decode only (stats reset after prefill) |

`init_context_ms` and `decode_peak_bytes` are `null` when the first sampled token is a stop token
(no decode happens; the four phase sums are then 0). With `step_timing=false` the four phase sums
are `null`. Decode time is not stored: it is `draft_ms + verification_ms + commit_ms + update_ms`.

### Step record

| Field | Meaning |
|---|---|
| sample key + `step` | step 0 is the first step; for DFlash/DSpark its draft latency includes projecting the prompt context |
| `requested_len` | `K` (0 for `ar`) |
| `drafted_len` | tokens the drafter produced |
| `effective_len` | draft length after stop-token truncation (DeepSpec's `proposal_lengths`) |
| `accepted` | accepted draft tokens |
| `committed` | tokens committed this step: `accepted + 1`, or `accepted` when an accepted draft token is a stop token |
| `draft_ms` | `propose`: the drafter producing the proposal |
| `verification_ms` | target forward on the proposal + acceptance decision |
| `commit_ms` | rest of the step: token commit, KV crop, stop-token check, loop bookkeeping (after the final step also output trimming) |
| `update_ms` | `update`: drafter state update with committed tokens (EAGLE-3 cache extension, DSpark/DFlash context features); 0 on the final stop-token step, which skips it |

The four step phases are `null` with `step_timing=false`. They share CUDA-event boundaries, so
`target_prefill_ms + init_context_ms + Σ steps (draft + verification + commit + update)` covers the
sample's GPU-stream timeline with no gaps or overlap; `wall_ms` is that plus host-only time. Invariants: `0 ≤ accepted ≤ effective_len ≤ drafted_len`;
`committed ∈ {accepted, accepted + 1}`; `1 + Σ committed == generated_len`, or `≥` when
`stop_reason == "length"`.

### Summary row

`accepted` counts draft tokens accepted by verification; `committed` also counts the bonus or
corrected token the target appends. The two blocks use different conventions:

| `metrics` field | Definition |
|---|---|
| `num_proposals` | speculative steps |
| `draft_tokens_per_proposal` | Σ `effective_len` / proposals |
| `acceptance_length` | Σ `accepted` / proposals; never exceeds `draft_tokens_per_proposal` |
| `committed_per_step` | Σ `committed` / proposals (decoding progress per target forward) |
| `draft_acceptance_rate` | Σ `accepted` / Σ `effective_len`; `null` when nothing was drafted (`ar`) |
| `accept_rates_by_position` | per position `i`: steps with `accepted > i` / steps with `effective_len > i` |

`deepspec_metrics` is DeepSpec's row unchanged. Its `acceptance_length` is **committed** tokens
per step (it includes the bonus token, so it equals `metrics.committed_per_step`), and its
`verify_rate` is Σ `committed` / (Σ `effective_len` + proposals). Both blocks are recomputed from
the step records by `python -m experiments.metrics <run_dir>`. Runs written before this split
have only DeepSpec's row, stored under `metrics`.

### Timing and memory method

Per-step and per-sample boundaries are CUDA events recorded on the current stream; they are
resolved with one `synchronize()` at sample end, so no host-device sync is added inside the loop
and asynchronous draft kernels are still counted. Intervals measure the GPU-stream timeline,
including idle gaps from host launch overhead. Memory comes from the caching allocator's
`max_memory_allocated()` statistics. Parameter counts are all drafter parameters, including the
drafter's own copies of the embedding and LM head.

## Tests

```bash
python -m pytest experiments/tests -m "not gpu"   # CPU only, seconds
python -m pytest experiments/tests                # + GPU tests, needs the weights (~6 min on a 3090)
```

The GPU tests check: K control and validation; zero-draft AR steps; per-step results at `K=7`
identical to DeepSpec's `Qwen3DSparkEvaluator` / `Qwen3Eagle3Evaluator`; instrumented vs plain
evaluators producing identical tokens and acceptance lists (step timing on and off); phase
attribution on `torch.cuda._sleep` kernels; decode peak below prefill peak; and a two-method run
with warm-up 2 vs 0 giving identical records apart from timing.

## Measured on one RTX 3090 (2026-10-24)

Stack: torch 2.9.1+cu128, transformers 5.10.2, bf16, `sdpa`, `DynamicCache`, bsz 1, no CUDA graphs.
Absolute latencies are specific to this HF eager stack.

**Instrumentation overhead** (DSpark, GSM8K, 10 samples, K=7, 2 runs per setting): mean per-sample
decode time 4751 ms with step timing vs 4752 ms without (−0.01%), measured with the earlier
`decode_ms` field; the per-phase schema records the same number of events per step. Run-to-run spread for the same
setting is about ±3%, so the overhead is below what this measurement can resolve, and within the
≤3% target. Acceptance results were identical in all four runs.

**Smoke config wall time** (`qwen3_4b_smoke.py`, total 250 s; each includes ~10 s spawn + load):

| Method | Wall time | Decode ms/token (K=3, ≤128 tokens) |
|---|---|---|
| ar | 96 s | 57.3 |
| eagle3 | 49 s | 24.7 |
| dflash | 53 s | 26.4 |
| dspark | 45 s | 22.0 |

**Main config estimate.** Mean generated lengths measured with DSpark at K=7 (10 samples each):
GSM8K ≈ 376, Math500 ≈ 939, AIME25 ≈ 1916 tokens (7/10 hit the 2048 cap). One pass over the main
datasets is therefore ≈ 500·376 + 500·939 + 30·1916 ≈ 0.72 M generated tokens.

- AR at ≈ 57 ms/token: ≈ 11 h.
- Each speculative (method, K) at ≈ 13–26 ms/token (DSpark K=7 long-context ≈ 12–14 ms; K=3 short-context ≈ 22–26 ms): ≈ 2.6–5.2 h; 9 such configurations ≈ 23–47 h.
- Prefill and warm-up add a few percent.

Total ≈ 35–60 h on one RTX 3090. The estimate is rough: at temperature 1.0 generated lengths
differ per method, and per-token latency grows with context. Reduce sample counts with `--opts`
or split the run by method if needed; JSONL files stay readable if a run is interrupted.
