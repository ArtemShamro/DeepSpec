# Spec Delta

## Purpose

Record per-sample and per-speculative-step timing, memory and draft/acceptance data during sweep runs, without changing decoding behavior, so that methods can be compared on acceptance, latency and memory.

## ADDED Requirements

### Requirement: Per-sample record
For every measured sample, a record SHALL be stored with:
- method, dataset, sample index, rank and requested `K`;
- prompt length and generated length in tokens;
- number of speculative steps;
- whether generation stopped on a stop token or on the max-new-tokens limit;
- the per-sample timing and memory fields.

#### Scenario: Sample hits length limit
- **WHEN** a sample generates the maximum number of new tokens without emitting a stop token
- **THEN** its record marks it as stopped by the length limit

### Requirement: Per-step record
For every speculative step, a record SHALL be stored with:
- the sample key and step index;
- requested draft length, drafted length (tokens produced by the drafter) and effective draft length (after stop-token truncation);
- accepted draft tokens and committed tokens;
- draft latency and verification latency.

#### Scenario: Stop token inside accepted draft
- **WHEN** an accepted draft token is a stop token
- **THEN** the step's effective draft length and accepted count end at that token, and its drafted length still reports what the drafter produced

#### Scenario: Step counts are consistent
- **WHEN** a sample's step records are summed
- **THEN** the number of step records equals the sample's step count, and the summed committed tokens plus the one token sampled at prefill equal the sample's generated length, unless the sample was cut at the max-new-tokens limit, in which case the sum is greater than or equal to it

### Requirement: Prefill and decode timing are separated
Each sample SHALL report three times separately:
- target prefill time: prompt forward plus first-token sampling;
- draft-side prefill time: drafter state initialization from prompt features before the first step;
- decode time: from the end of draft-side prefill to the end of generation.

Decode time excludes both prefill components.

#### Scenario: EAGLE-3 draft prefill
- **WHEN** an `eagle3` sample is decoded
- **THEN** the drafter's processing of the prompt before the first step is reported as draft-side prefill time and is not included in decode time

#### Scenario: Block drafter first step
- **WHEN** a `dflash` or `dspark` sample is decoded
- **THEN** the first step's record is identifiable as step 0, so that its draft latency, which includes projecting the prompt context, can be analyzed separately

### Requirement: Draft latency covers all drafter work in a step
A step's draft latency SHALL include all drafter computation attributable to that step: producing the proposal, and updating drafter state with committed tokens after verification. Verification latency SHALL cover the target forward on the proposal and the acceptance decision.

#### Scenario: EAGLE-3 cache extension
- **WHEN** an `eagle3` step commits tokens and the drafter extends its cache with them
- **THEN** that extension time is counted in that step's draft latency

### Requirement: GPU-asynchrony-aware timing without extra synchronization
Per-step timings SHALL measure GPU execution, so they are correct under asynchronous CUDA execution. They SHALL NOT add host-device synchronization points inside the decoding loop beyond those already present. Per-sample wall-clock times SHALL be taken with the device synchronized at sample start and end.

#### Scenario: Asynchronous draft forward
- **WHEN** a drafter's proposal kernels are still running when the host returns from proposing
- **THEN** the recorded draft latency still includes that kernel execution time

### Requirement: Instrumentation overhead is measurable
Per-step timing SHALL be switchable off while keeping per-sample timing, so its overhead on decode time can be measured on the same samples.

#### Scenario: Overhead comparison
- **WHEN** the same (method, dataset, K) is run with per-step timing on and off
- **THEN** both runs produce identical acceptance results, and their per-sample decode times can be compared

### Requirement: Instrumentation does not change decoding results
For the same seed, samples and `K`, enabling instrumentation SHALL NOT change generated tokens or acceptance results compared with running the same evaluator without it.

#### Scenario: Instrumented vs plain run
- **WHEN** a method is run on the same samples with and without instrumentation
- **THEN** the per-step accepted counts and generated token ids are identical

### Requirement: Peak memory per phase
Each sample SHALL report two values of peak allocated GPU memory: the peak during target and draft-side prefill, and the peak during decode. Each run SHALL also record allocated memory after model loading, and target and draft parameter counts.

#### Scenario: Decode peak excludes prefill
- **WHEN** a long prompt's prefill allocates more memory than any decode step
- **THEN** the sample's decode peak reflects only allocations made during decode

### Requirement: Result storage layout
Each rank SHALL write its sample records and step records as JSON Lines files in the run directory. On completion, the run SHALL write a summary file with one entry per (method, dataset, K).

#### Scenario: Interrupted run
- **WHEN** a run is interrupted after some (method, dataset, K) combinations finished
- **THEN** the records for finished combinations remain readable in the JSONL files

### Requirement: Consistency with DeepSpec acceptance metrics
For each (method, dataset, K), the summary SHALL include DeepSpec's own metrics row:
- acceptance length;
- draft tokens per proposal;
- verify rate;
- per-position accept rates.

Recomputing these metrics from the step records SHALL reproduce the same values.

#### Scenario: Parity check
- **WHEN** the acceptance metrics are recomputed from a run's step records
- **THEN** they equal the DeepSpec metrics row stored in that run's summary
