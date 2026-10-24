# Spec Delta

## Purpose

Configure and execute reproducible comparison runs of speculative decoding methods and the autoregressive baseline over fixed datasets, draft lengths and generation settings.

## ADDED Requirements

### Requirement: Sweep is defined by a repository config file
A sweep SHALL be fully defined by a config file in the repository plus optional command-line overrides. The config names the target model, the methods and their draft checkpoints, the datasets with per-dataset sample counts, the draft lengths `K`, the generation settings, the seed and the warm-up sample count.

#### Scenario: Run from config
- **WHEN** the user runs the sweep entry point with a config path
- **THEN** every (method, dataset, K) combination in the config is executed and its results are written under one run directory

#### Scenario: Override a config field
- **WHEN** the user passes an override for a config field on the command line
- **THEN** the run uses the overridden value and records both the original config path and the overrides in the run manifest

### Requirement: Supported methods
A sweep SHALL support the methods `ar` (autoregressive target-only baseline), `eagle3`, `dflash` and `dspark` for the Qwen3 target. Each speculative method SHALL be evaluated with the existing DeepSpec draft model and its verification procedure.

#### Scenario: Unknown method
- **WHEN** the config lists a method that is not supported
- **THEN** the run fails before any model is loaded, and the error names the unsupported method

#### Scenario: AR baseline has no draft model
- **WHEN** the `ar` method runs
- **THEN** no draft checkpoint is loaded, and every decoding step commits exactly one target-sampled token

### Requirement: Autoregressive baseline shares the decoding path
The `ar` baseline SHALL generate through the same target-model forward, KV-cache handling, sampling and stop-token handling as the speculative methods, with zero draft tokens per step.

#### Scenario: Baseline step accounting
- **WHEN** an `ar` sample is decoded
- **THEN** each step records a requested draft length of 0, 0 accepted draft tokens and 1 committed token

### Requirement: Draft length control
For speculative methods, `K` SHALL set the number of draft tokens requested per step. For `eagle3`, `K` is the number of chained draft steps. For `dflash` and `dspark`, the drafter runs on a block of size `K` and produces `K` draft tokens.

#### Scenario: Block drafter at reduced K
- **WHEN** `dspark` or `dflash` runs with `K=3` on a checkpoint trained with block size 7
- **THEN** each draft forward processes a 3-position block and proposes at most 3 draft tokens

#### Scenario: K above trained length is rejected
- **WHEN** a speculative method is configured with `K` larger than the draft length its checkpoint was trained with
- **THEN** the run fails before decoding starts, and the error names the method, the requested `K` and the trained length

#### Scenario: AR ignores K
- **WHEN** the config lists several `K` values
- **THEN** the `ar` method runs once per dataset rather than once per `K`

### Requirement: Sample selection and seeding match DeepSpec eval
For each dataset, the sweep SHALL select samples, build prompts and seed each sample exactly as DeepSpec's evaluator does for the same seed and sample count. Every method and every `K` therefore decodes the same prompts with the same per-sample seed.

#### Scenario: Same prompts across methods
- **WHEN** two methods run on the same dataset with the same seed and sample count
- **THEN** both decode the same ordered set of sample indices with identical prompt token ids

### Requirement: Generation settings are fixed per run
Temperature, maximum new tokens, the chat template mode and the seed SHALL be set once per sweep and applied identically to every method and `K`. The defaults SHALL equal DeepSpec eval's defaults.

#### Scenario: Defaults
- **WHEN** a config does not set generation settings
- **THEN** the run uses temperature 1.0, 2048 max new tokens, non-thinking chat template and seed 980406, and records them in the manifest

### Requirement: Warm-up precedes measurement
Before measured samples of each (method, K), the sweep SHALL decode a configured number of warm-up samples with the same model and settings. Warm-up results SHALL be excluded from the recorded samples and SHALL NOT change the per-sample seeding of measured samples.

#### Scenario: Warm-up excluded
- **WHEN** a (method, K) is run with 3 warm-up samples and 10 measured samples
- **THEN** exactly 10 samples are recorded for each dataset, and their outputs equal those of a run with zero warm-up samples

### Requirement: Methods run in isolated processes
Each method SHALL be loaded and evaluated in its own process, so models, caches and allocator state from one method do not affect another method's memory or latency measurements.

#### Scenario: Sequential methods
- **WHEN** a sweep lists `ar` and `dspark`
- **THEN** `dspark` starts in a fresh process after the `ar` process has exited

### Requirement: Diagnostic recorders are disabled during measurement
Diagnostic-only collectors that add work to the decoding loop, such as DSpark's confidence-calibration recorder, SHALL be disabled in sweep runs. Algorithm behavior such as the draft model's own forward computations SHALL remain unchanged.

#### Scenario: DSpark checkpoint with confidence head
- **WHEN** `dspark` runs with a checkpoint that contains a confidence head
- **THEN** no confidence-calibration statistics are collected, and proposals still use DeepSpec's default full-length behavior

### Requirement: Run manifest
Each run directory SHALL contain a manifest recording what is needed to reproduce and interpret the run:
- the resolved config, the original config path and the overrides;
- the git commit and whether the working tree had uncommitted changes;
- library versions (Python, torch, transformers, CUDA);
- GPU model per rank;
- model identifiers and parameter counts for target and draft;
- the start time.

#### Scenario: Dirty working tree
- **WHEN** a sweep is started with uncommitted changes in the repository
- **THEN** the manifest records the commit and marks the working tree as dirty

### Requirement: Results are not committed
The default results location SHALL be excluded from version control.

#### Scenario: After a run
- **WHEN** a sweep writes results to the default location
- **THEN** `git status` shows no new files under that location
