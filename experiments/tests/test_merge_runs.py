import json

import pytest

from experiments.merge_runs import merge_runs
from experiments.metrics import check_parity


def _write_run(root, run_id, start_time, groups, *, temperature=1.0):
    """groups: {(method, dataset, k): num_samples}; one step per sample."""
    run_dir = root / run_id
    (run_dir / "workers").mkdir(parents=True)
    samples, steps, summary = [], [], []
    for (method, dataset, k), n in groups.items():
        key = {"method": method, "dataset": dataset, "k": k, "rank": 0}
        for idx in range(n):
            samples.append({**key, "sample_idx": idx})
            steps.append({**key, "sample_idx": idx, "step": 0, "effective_len": k, "accepted": min(idx, k),
                          "committed": min(idx, k) + 1})
        summary.append({**key, "num_positions": max(k, 1)})
    for kind, records in (("samples", samples), ("steps", steps)):
        for method in {r["method"] for r in records}:
            with open(run_dir / f"{kind}.{method}.rank0.jsonl", "w") as handle:
                handle.writelines(json.dumps(r) + "\n" for r in records if r["method"] == method)
    (run_dir / "summary.jsonl").write_text("".join(json.dumps(r) + "\n" for r in summary))
    for method in {m for m, _, _ in groups}:
        (run_dir / "workers" / f"{method}.rank0.json").write_text(json.dumps({"method": method}))
    manifest = {
        "run_id": run_id,
        "start_time": start_time,
        "config_path": "cfg.py",
        "opts": [],
        "config": {
            "target_name_or_path": "t",
            "generation": {"temperature": temperature},
            "warmup_samples": 3,
            "step_timing": True,
            "methods": [{"name": m} for m in sorted({m for m, _, _ in groups})],
            "draft_lengths": sorted({k for _, _, k in groups if k}),
        },
        "git": {"commit": "abc", "dirty": False},
        "versions": {},
        "cuda_device_count": 1,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    return run_dir


def test_newest_run_wins_and_samples_are_truncated(tmp_path):
    old = _write_run(tmp_path, "old", "2026-01-01", {("ar", "ds", 0): 4, ("m", "ds", 3): 1})
    new = _write_run(tmp_path, "new", "2026-01-02", {("m", "ds", 3): 2})
    out = tmp_path / "merged"
    summary = merge_runs([new, old], out, {"ds": 2})

    assert [(r["method"], r["k"], r["metrics"]["num_samples"]) for r in summary] == [("ar", 0, 2), ("m", 3, 2)]
    assert check_parity(out) == []
    manifest = json.loads((out / "manifest.json").read_text())
    assert {s["run_id"]: s["groups"] for s in manifest["merged_from"]} == {
        "old": [["ar", "ds", 0]],
        "new": [["m", "ds", 3]],
    }


def test_incomplete_group_is_rejected(tmp_path):
    run = _write_run(tmp_path, "run", "2026-01-01", {("m", "ds", 3): 1})
    with pytest.raises(ValueError, match="does not cover"):
        merge_runs([run], tmp_path / "merged", {"ds": 2})


def test_incomparable_runs_are_rejected(tmp_path):
    a = _write_run(tmp_path, "a", "2026-01-01", {("m", "ds", 3): 2})
    b = _write_run(tmp_path, "b", "2026-01-02", {("m", "ds", 4): 2}, temperature=0.0)
    with pytest.raises(ValueError, match="generation"):
        merge_runs([a, b], tmp_path / "merged", {"ds": 2})
