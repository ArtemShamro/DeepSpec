# Proposal

## Why

EXPERIMENTS.md compares AR Qwen3-4B, EAGLE-3, DFlash and DSpark over GSM8K / Math500 / AIME25 at `K ∈ {3, 5, 7}`. The current DeepSpec eval (`eval.py`) only prints aggregate acceptance metrics:
- no latency or memory measurement;
- no autoregressive baseline;
- no warm-up;
- no way to set `K`;
- per-sample and per-step data is discarded after `all_reduce`.

Every experiment in the matrix depends on this infrastructure, so it is the first piece of work.

## What Changes

- **Sweep runner** under `experiments/`, driven by a Python config:
  - the config sets methods, datasets with sample counts, draft lengths `K` and generation settings;
  - the runner reuses DeepSpec's dataset loading, sample selection, seeding and speculative-decoding loop unchanged;
  - it does warm-up before measurement;
  - it runs each method in its own process so memory measurements are isolated.
- **Draft length override**:
  - EAGLE-3: `K` sets the number of chain draft steps;
  - DFlash / DSpark: the drafter runs on a block of size `K` (anchor + `K-1` mask slots → `K` draft tokens) rather than its trained `block_size=7`;
  - a `K` above the length the checkpoint was trained with is rejected.
- **Autoregressive baseline**: proposes zero draft tokens. It runs through the same target-model loop, KV cache and sampling code as the speculative methods.
- **Per-sample and per-step instrumentation** covering every field listed in EXPERIMENTS.md "Data for Logging":
  - prefill time, draft-side prefill time, decode time;
  - draft and verification latency per step;
  - requested / drafted / effective draft length, accepted and committed tokens;
  - prefill and decode peak allocated memory.
  - Timing uses GPU events, so it adds no extra host-device synchronization.
- **Run artifacts** in `experiments/results/<run_id>/`: a reproducibility manifest, per-rank JSONL for samples and steps, and a summary that includes DeepSpec's own metrics row for each (method, dataset, K).
- **Parity check**: acceptance metrics recomputed from the JSONL must match DeepSpec's `build_metrics_row` for the same run.
- **Smoke sweep config** for fast end-to-end verification on a single GPU.
- `experiments/results/` is git-ignored.

No change to `deepspec/`, `eval.py`, `config/` or `scripts/`: the existing eval behavior is preserved.

## Capabilities

### New Capabilities

- `experiment-sweeps`: configuring and executing reproducible comparison runs. Covers method selection (including the AR baseline), draft-length control, dataset and sample selection, fixed generation settings, warm-up, process isolation and the run manifest.
- `decoding-instrumentation`: what is measured during each decoded sample and speculative step (timing, memory, draft and acceptance accounting), how it is stored, and how it stays consistent with DeepSpec's acceptance metrics.

### Modified Capabilities

None. No specs exist yet.

## Impact

- **New code**: `experiments/` (runner, evaluator subclasses, instrumentation, sweep configs, tests).
- **Reused unchanged**: `deepspec/eval/base_evaluator.py` (loop, verification, dataset handling), `deepspec/eval/dspark/`, `deepspec/eval/eagle3/`, `deepspec/utils` (config loader, seeding, distributed init).
- **Repo hygiene**: `.gitignore` gains `experiments/results/`.
- **Environment**: needs a Python environment with `requirements.txt` plus `pytest`, the `Qwen/Qwen3-4B` target and the released DeepSpec draft checkpoints (`deepseek-ai/{eagle3_qwen3_4b_ttt7, dflash_qwen3_4b_block7, dspark_qwen3_4b_block7}`). None of these are present on the machine yet.
- **Out of scope**: analysis notebook, synthetic long-context memory sweep, new algorithms.
