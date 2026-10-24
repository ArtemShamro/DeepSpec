# Experiments Description

Implementation details may be changed when necessary only after direct user approval, provided that the resulting measurements remain comparable and reproducible.

## Goal

Compare speculative decoding methods for Qwen3-4B.

## Experiment Matrix

Methods:

- autoregressive Qwen3-4B baseline
- EAGLE-3
- DFlash
- DSpark

Datasets:

- GSM8K
- Math500
- AIME25

Draft length `K`:

- `K ∈ {3, 5, 7}`

All values are within the 7 draft tokens the released DeepSpec checkpoints were trained with (`*_block7`, `*_ttt7`).

## Metrics

Primary metrics:

- acceptance length vs draft length
- decode latency per token vs draft length
- peak VRAM vs draft length and context length

Prompt prefill should be excluded from the primary decode-latency comparison.

## Data for Logging

Per sample:

- prompt length
- generated length
- prefill time
- decode time
- peak memory
- number of speculative steps

Per speculative step:

- requested / actual draft length
- accepted draft tokens
- committed tokens
- draft latency
- verification latency

The exact storage format and instrumentation implementation are intentionally left flexible.

## Measurement Protocol

- Perform warm-up runs before collecting latency measurements.
- Account for asynchronous CUDA execution when measuring GPU latency.
- Keep prefill and decode latency separate.
- Measure memory after warm-up and report peak allocated GPU memory.
- Avoid instrumentation that significantly changes the measured execution path.
- Keep generation and sampling settings fixed across compared methods.

# Report

Result analysis should be performed in a Jupyter notebook.

## Main Report Plots and Tables

Base plots:

- acceptance length vs K
- decode latency vs K

Additional plots:

- speedup vs K
- draft / verification latency vs K
- peak VRAM vs context length
- speedup vs additional VRAM

Secondary metrics:

- throughput, tokens/s
- speedup over autoregressive baseline
- draft acceptance rate

## Resource Efficiency Report

Compare methods in terms of:

- drafter parameter count
- peak GPU memory
- additional memory over the target-model baseline
- draft / verification work required per generated token

Exact compute metrics may be adapted depending on what can be measured reliably.

# Reproducibility

Most of the reproducibility infrastructure is provided by the DeepSpec repository.

Every experiment should be reproducible from repository scripts and configs.