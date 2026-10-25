"""Sweep config loading and validation (design D10).

Configs are Python files read with DeepSpec's `load_config`; `--opts key=value`
overrides go through `parse_opts_to_config`, as in `train.py`. Validation runs
in the parent process and needs only checkpoint configs, not weights.
"""

from __future__ import annotations

import copy
from pathlib import Path

from deepspec.eval.base_evaluator import DEFAULT_DATASET_ROOT
from deepspec.utils import load_config, parse_opts_to_config
from deepspec.utils.config import config_to_plain_dict, to_config_node

# Defaults match DeepSpec's eval.py; the chat template mode is fixed by
# BaseEvaluator.run_dataset and only recorded here.
DEFAULTS = {
    "generation": {
        "temperature": 1.0,
        "max_new_tokens": 2048,
        "seed": 980406,
        "enable_thinking": False,
    },
    "warmup_samples": 3,
    "step_timing": True,
    "output_root": "experiments/results",
}
REQUIRED_KEYS = ("target_name_or_path", "methods", "datasets", "draft_lengths")

# method -> (accepted draft-checkpoint architectures, config attribute holding the trained draft length)
SPECULATIVE_METHODS = {
    "eagle3": ({"Qwen3Eagle3Model", "Eagle3DraftModel"}, "ttt_length"),
    "dflash": ({"Qwen3DSparkModel"}, "block_size"),
    "dspark": ({"Qwen3DSparkModel"}, "block_size"),
}
SUPPORTED_METHODS = ("ar", *SPECULATIVE_METHODS)


def _merge_defaults(cfg: dict, defaults: dict) -> None:
    for key, value in defaults.items():
        if key not in cfg:
            cfg[key] = copy.deepcopy(value)
        elif isinstance(value, dict):
            _merge_defaults(cfg[key], value)


def load_sweep_config(config_path: str, opts: list[str] | None = None):
    cfg = load_config(config_path)
    cfg = to_config_node(config_to_plain_dict(cfg))
    _merge_defaults(cfg, DEFAULTS)
    cfg = parse_opts_to_config(list(opts or []), cfg)
    cfg = to_config_node(config_to_plain_dict(cfg))
    cfg.datasets = [(str(name), int(count)) for name, count in cfg.datasets]
    cfg.draft_lengths = [int(k) for k in cfg.draft_lengths]
    return cfg


def _load_draft_config(name_or_path: str):
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(name_or_path)


def validate_sweep_config(
    cfg,
    *,
    draft_config_loader=_load_draft_config,
    dataset_root: str = DEFAULT_DATASET_ROOT,
) -> dict[str, int]:
    """Raises ValueError on an invalid config; returns trained draft length per speculative method."""
    missing = [key for key in REQUIRED_KEYS if key not in cfg]
    if missing:
        raise ValueError(f"config is missing required keys: {missing}")

    names = [method["name"] for method in cfg.methods]
    for name in names:
        if name not in SUPPORTED_METHODS:
            raise ValueError(
                f"unsupported method {name!r}; supported methods: {list(SUPPORTED_METHODS)}"
            )
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate methods in config: {names}")

    if not cfg.draft_lengths:
        raise ValueError("draft_lengths must not be empty")
    for k in cfg.draft_lengths:
        if k < 1:
            raise ValueError(f"draft length K={k} must be >= 1")

    if cfg.generation.enable_thinking:
        raise ValueError("DeepSpec eval always uses the non-thinking chat template")
    for dataset_name, max_samples in cfg.datasets:
        if not (Path(dataset_root) / f"{dataset_name}.jsonl").exists():
            raise ValueError(f"dataset {dataset_name!r} not found under {dataset_root}")
        if max_samples < 1:
            raise ValueError(f"dataset {dataset_name!r}: sample count must be >= 1")

    trained_lengths = {}
    for method in cfg.methods:
        name = method["name"]
        if name == "ar":
            continue
        draft_path = method.get("draft_name_or_path")
        if not draft_path:
            raise ValueError(f"method {name!r} requires draft_name_or_path")
        architectures, length_attr = SPECULATIVE_METHODS[name]
        draft_config = draft_config_loader(draft_path)
        arch = (getattr(draft_config, "architectures", None) or [None])[0]
        if arch not in architectures:
            raise ValueError(
                f"method {name!r}: checkpoint {draft_path} has architecture {arch!r}, "
                f"expected one of {sorted(architectures)}"
            )
        trained = int(getattr(draft_config, length_attr))
        for k in cfg.draft_lengths:
            if k > trained:
                raise ValueError(
                    f"method {name!r}: draft length K={k} exceeds the trained draft "
                    f"length {trained} of checkpoint {draft_path}"
                )
        trained_lengths[name] = trained
    return trained_lengths
