# Tasks

## 1. Environment and scaffolding

- [ ] 1.1 Create a project virtualenv and install `requirements.txt` plus `pytest`. Verify that `python -c "import torch, transformers; print(torch.cuda.is_available())"` prints `True` with torch 2.9.1 / transformers 5.10.2.
- [ ] 1.2 Download `Qwen/Qwen3-4B`, `deepseek-ai/eagle3_qwen3_4b_ttt7`, `deepseek-ai/dflash_qwen3_4b_block7` and `deepseek-ai/dspark_qwen3_4b_block7` to the HF cache. Verify that `AutoConfig.from_pretrained` succeeds for each and reports the expected architectures (`Qwen3Eagle3Model` or `Eagle3DraftModel`, and `Qwen3DSparkModel`).
- [ ] 1.3 Run DeepSpec's unchanged `eval.py` on one draft checkpoint with a temporary task list of 2 GSM8K samples via an `agent_space/` script. Verify that it prints a metrics table. This gives a pre-change behavior baseline.
- [ ] 1.4 Create `experiments/` (`__init__.py`, `configs/`, `tests/`) and add `experiments/results/` to `.gitignore`. Verify that a file created under `experiments/results/` does not appear in `git status`.

## 2. Evaluators: K control and AR baseline

- [ ] 2.1 Implement a DSpark/DFlash evaluator subclass with a settable `K` behind `max_proposal_tokens`, `1 ≤ K ≤ block_size` validation, and the confidence recorder disabled (D4). Add a GPU test that verifies: at `K=3`, every proposal has `draft_token_count ≤ 3`; `K=8` raises with method, K and trained length in the message; no confidence statistics are collected.
- [ ] 2.2 Implement the EAGLE-3 evaluator subclass with a settable `K` and `1 ≤ K ≤ ttt_length` validation (D4). Add a GPU test that verifies: at `K=3`, proposals have at most 3 draft tokens; `K=8` raises.
- [ ] 2.3 Implement the AR baseline evaluator: target only, zero-draft proposals (D5). Add a GPU test that verifies: no draft model is loaded; every step has accepted = 0 and committed = 1; generation stops on a stop token or at `max_new_tokens`.
- [ ] 2.4 Add a GPU test that verifies: for `K=7`, the DSpark subclass gives per-step acceptance lists identical to DeepSpec's `Qwen3DSparkEvaluator` on the same 2 samples. The same check runs for EAGLE-3 at `K=7` vs `Qwen3Eagle3Evaluator`.

## 3. Instrumentation

- [ ] 3.1 Implement the instrumentation mixin that wraps `init_context` / `propose` / `post_verify` / `update` with CUDA-event boundaries (D1, D2). It resolves times at sample end with one sync, records drafted length from `post_verify` (D3), and has a `step_timing` on/off switch. Add a GPU unit test with toy callbacks around known `torch.cuda._sleep` kernels. It verifies that draft and verification intervals are attributed to the right phase and that EAGLE-3-style `update` time counts as draft time.
- [ ] 3.2 Implement per-phase memory capture: reset at sample start; prefill peak read and reset at draft-prefill end; decode peak at end; post-load memory and parameter counts per worker (D8). Add a GPU test that verifies the decode peak is below the prefill peak for a long-prompt / short-generation sample.
- [ ] 3.3 Implement record building that merges DeepSpec's per-step lists with the timing data into sample and step records (spec fields, D3). Add CPU tests that verify: step count consistency; the "1 + Σcommitted = generated length unless length-limited" rule; stop-token steps (drafted vs effective length); length-limit flagging.
- [ ] 3.4 Implement metric recomputation from step records (acceptance length, draft tokens per proposal, verify rate, accept_rate@i). Add a CPU test that verifies exact equality with `BaseEvaluator.build_metrics_row` on synthetic step lists, including stop-token and zero-draft (AR) cases.
- [ ] 3.5 Add a GPU test that verifies: the instrumented evaluator produces token ids and acceptance lists identical to the non-instrumented one on 2 samples for each method, with step timing both on and off.

## 4. Runner, configs and storage

- [ ] 4.1 Implement config loading and validation using `load_config` / `parse_opts_to_config` with `--opts` (D10). Unknown methods, `K < 1`, and `K` above the checkpoint's trained length (read from its config without loading weights) fail before model loading; defaults match DeepSpec eval (temperature 1.0, 2048 tokens, seed 980406). Add CPU tests for the defaults, overrides and both failure messages.
- [ ] 4.2 Implement the manifest writer (resolved config, config path + overrides, git sha + dirty flag, library versions) and per-worker info files (GPU name, parameter counts, post-load memory) (D9). Add a CPU test that verifies a dirty tree is flagged, using a temporary git repo.
- [ ] 4.3 Implement the parent process and worker loop (D6, D7):
  - one spawned process per method;
  - warm-up via `run_dataset(max_samples=W)` discarded, then the measured `run_dataset`;
  - AR runs once per dataset;
  - JSONL files flushed per sample;
  - rank 0 appends the DeepSpec metrics row to `summary.jsonl`.

  Verify on GPU: a 2-method, 2-sample run produces all expected files, and warm-up `W=2` vs `W=0` give identical measured records apart from timing fields.
- [ ] 4.4 Add `experiments/configs/qwen3_4b_main.py` (full matrix: 4 methods, 3 datasets with DeepSpec sample counts, `K ∈ {3, 5, 7}`) and `experiments/configs/qwen3_4b_smoke.py` (4 methods, `K=3`, 3 samples per dataset, `max_new_tokens=128`, `W=1`). Verify that both load and validate without starting a run.
- [ ] 4.5 Write `experiments/README.md` with environment setup, run commands for both configs, an output file layout and field reference, and the K/AR semantics. Verify that the documented smoke command runs as written.

## 5. Integration checks

- [ ] 5.1 Run the smoke config end to end on the RTX 3090. Verify:
  - the run completes;
  - every (method, dataset, K) has a summary row;
  - recomputing metrics from `steps.*.jsonl` matches `summary.jsonl` exactly;
  - `git status` is clean apart from the change's own source files.
- [ ] 5.2 Measure instrumentation overhead: run one speculative method on GSM8K (10 samples, K=7) with `step_timing` on and off. Report the relative difference in mean per-sample decode time, targeting ≤ 3%. If it is higher, record it in `experiments/README.md` and in design.md Risks before proceeding.
- [ ] 5.3 Record the measured smoke-run wall time per method in `experiments/README.md`, with an extrapolated estimate for the main config, so the full sweep can be scheduled.
