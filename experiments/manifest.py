"""Run manifest and per-worker info files (design D9)."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers

from deepspec.utils import CustomJSONEncoder


def git_state(repo_dir: str | Path) -> dict:
    def run(*cmd):
        return subprocess.check_output(["git", *cmd], cwd=repo_dir, encoding="utf-8").strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def library_versions() -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, cls=CustomJSONEncoder)
        handle.write("\n")


def write_manifest(
    run_dir: Path,
    *,
    cfg,
    config_path: str,
    opts: list[str],
    trained_draft_lengths: dict[str, int],
    repo_dir: str | Path,
    start_time: datetime,
) -> dict:
    manifest = {
        "run_id": run_dir.name,
        "start_time": start_time.isoformat(),
        "config_path": str(Path(config_path).resolve()),
        "opts": list(opts),
        "config": cfg,
        "trained_draft_lengths": trained_draft_lengths,
        "git": git_state(repo_dir),
        "versions": library_versions(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    write_json(run_dir / "manifest.json", manifest)
    return manifest


def count_parameters(model) -> int | None:
    if model is None:
        return None
    return sum(p.numel() for p in model.parameters())


def write_worker_info(run_dir: Path, *, method: dict, evaluator) -> dict:
    """Called once per worker right after its models are loaded."""
    info = {
        "method": method["name"],
        "rank": evaluator.global_rank,
        "world_size": evaluator.world_size,
        "gpu_name": torch.cuda.get_device_name(evaluator.device),
        "target_name_or_path": evaluator.args.target_name_or_path,
        "draft_name_or_path": method.get("draft_name_or_path"),
        "target_params": count_parameters(evaluator.target_model),
        "draft_params": count_parameters(evaluator.draft_model),
        "trained_draft_length": getattr(evaluator, "trained_draft_length", None),
        "post_load_allocated_bytes": torch.cuda.memory_allocated(evaluator.device),
    }
    write_json(
        run_dir / "workers" / f"{method['name']}.rank{evaluator.global_rank}.json",
        info,
    )
    return info
