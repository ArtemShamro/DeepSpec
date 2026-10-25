"""End-to-end runner check: spawned workers, output files, warm-up isolation."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.metrics import check_parity, read_jsonl

pytestmark = pytest.mark.gpu

REPO_ROOT = Path(__file__).resolve().parents[2]
TIMING_FIELDS = {
    "target_prefill_ms",
    "draft_prefill_ms",
    "decode_ms",
    "wall_ms",
    "prefill_peak_bytes",
    "decode_peak_bytes",
    "draft_ms",
    "verify_ms",
}


def run_sweep(output_root: Path, warmup: int) -> Path:
    opts = [
        f"output_root={output_root}",
        "methods=[{name: ar, draft_name_or_path: null}, "
        "{name: dspark, draft_name_or_path: deepseek-ai/dspark_qwen3_4b_block7}]",
        "datasets=[[gsm8k, 2]]",
        "draft_lengths=[3]",
        "generation.max_new_tokens=64",
        f"warmup_samples={warmup}",
    ]
    cmd = [sys.executable, "-m", "experiments.run_sweep", "--config", "experiments/configs/qwen3_4b_smoke.py"]
    for opt in opts:
        cmd += ["--opts", opt]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    (run_dir,) = output_root.iterdir()
    return run_dir


def strip_timing(rows):
    return [{k: v for k, v in row.items() if k not in TIMING_FIELDS} for row in rows]


def test_two_method_run(tmp_path, evaluators):
    evaluators.release()  # free the GPU for the spawned workers
    warm = run_sweep(tmp_path / "warm", warmup=2)
    cold = run_sweep(tmp_path / "cold", warmup=0)

    for run_dir in (warm, cold):
        names = {p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*") if p.is_file()}
        assert names == {
            "manifest.json",
            "summary.jsonl",
            "samples.ar.rank0.jsonl",
            "steps.ar.rank0.jsonl",
            "samples.dspark.rank0.jsonl",
            "steps.dspark.rank0.jsonl",
            "workers/ar.rank0.json",
            "workers/dspark.rank0.json",
        }
        assert check_parity(run_dir) == []
        summary = read_jsonl([run_dir / "summary.jsonl"])
        assert [(r["method"], r["dataset"], r["k"]) for r in summary] == [("ar", "gsm8k", 0), ("dspark", "gsm8k", 3)]
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["config"]["generation"]["max_new_tokens"] == 64
        ar_info = json.loads((run_dir / "workers/ar.rank0.json").read_text())
        assert ar_info["draft_params"] is None and ar_info["target_params"] > 0

    for kind in ("samples", "steps"):
        for method in ("ar", "dspark"):
            name = f"{kind}.{method}.rank0.jsonl"
            warm_rows = read_jsonl([warm / name])
            assert len(warm_rows) > 0
            assert strip_timing(warm_rows) == strip_timing(read_jsonl([cold / name]))
    samples = read_jsonl([warm / "samples.dspark.rank0.jsonl"])
    assert [s["sample_idx"] for s in samples] == [0, 1]
    # Same prompts across methods.
    ar_samples = read_jsonl([warm / "samples.ar.rank0.jsonl"])
    assert [s["prompt_sha1"] for s in samples] == [s["prompt_sha1"] for s in ar_samples]
