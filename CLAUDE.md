# DeepSpec Research Fork

- This is an LLM research project focused on speculative decoding algorithms.
- The repository is based on a fork of `deepseek-ai/DeepSpec`, which provides data preparation, training, and evaluation pipelines for EAGLE-3, DFlash, and DSpark.
- The main technologies used in this project are PyTorch, Hugging Face Transformers, Triton, and SentencePiece.

## PROJECT GOALS

- Goal: run reproducible experiments to measure and compare speculative decoding algorithms across relevant metrics.
    
    Please read if you need more info about experiments: @EXPERIMENTS.md

- Out of scope: implementing new algorithms (including DFlash2). Compare only the methods already provided by DeepSpec and the autoregressive baseline.

## PROJECT RULES

- Use Spec-Driven Development with OpenSpec for non-trivial changes.
- Explore and understand the relevant code path before modifying it.
- Preserve existing DeepSpec behavior unless a specification explicitly requires a change.
- Prefer minimal, local changes over broad refactoring.
- Do not introduce abstractions solely for hypothetical future requirements.
- Keep experimental methodology reproducible and do not silently change evaluation protocols or baselines.
- Prefer simple, testable implementations suitable for both research experiments and a potential upstream contribution.
- Use `agent_space/` for temporary scripts and debugging artifacts.

## REPO ARCHITECTURE

- deepspec/ — core DeepSpec library: models, trainers, evaluators, data utilities, and common infrastructure. Keep changes here minimal and compatible with upstream DeepSpec.
- config/ — original DeepSpec model/training configurations for EAGLE-3, DFlash, and DSpark.
- scripts/ — original data preparation, training, and evaluation entry scripts.
- eval_datasets/ — benchmark datasets used for evaluation.
- experiments/ — project-specific experiment infrastructure. Put sweep configs, runners, profiling, metric collection, aggregation, and analysis code here.
- experiments/results/ — experiment outputs. Large or temporary results should not be committed.
- openspec/ — specifications and implementation plans for non-trivial changes.
- agent_space/ — temporary scripts, debugging code, inspections, and disposable artifacts created during development. Do not treat this directory as production code.

Changes that implement or fix an algorithm belong in deepspec/. Code that only orchestrates, measures, or analyzes experiments belongs in experiments/.



