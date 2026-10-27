"""Merge several sweep runs into one run directory the report notebook can read.

    python -m experiments.merge_runs --out experiments/results/<name> \
        --datasets gsm8k=50,math500=50,aime25=30 <run_dir> [<run_dir> ...]

Each (method, dataset, k) is taken from the newest run (manifest `start_time`)
that logged it, so a group interrupted in an older run and redone in a newer
one comes from the newer run. Records are kept unchanged; only samples with
`sample_idx < N` of their dataset are kept, so every group covers the same
prompts. A group that does not cover `0..N-1` exactly is an error.
`summary.jsonl` is recomputed from the kept step records with the same
functions `run_sweep` uses, so `python -m experiments.metrics <out>` holds.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

from experiments.metrics import acceptance_metrics, counts_from_steps, deepspec_metrics_row, read_jsonl

# Config fields that must agree across merged runs for the numbers to be comparable.
COMPARABLE_FIELDS = ("target_name_or_path", "generation", "warmup_samples", "step_timing")


def group_key(record: dict) -> tuple:
    return record["method"], record["dataset"], record["k"]


def load_run(run_dir: Path) -> dict:
    return {
        "dir": run_dir,
        "manifest": json.loads((run_dir / "manifest.json").read_text()),
        "samples": read_jsonl(sorted(run_dir.glob("samples.*.jsonl"))),
        "steps": read_jsonl(sorted(run_dir.glob("steps.*.jsonl"))),
        "summary": read_jsonl([run_dir / "summary.jsonl"]),
    }


def merge_runs(run_dirs: list[Path], out_dir: Path, dataset_sizes: dict[str, int]) -> list[dict]:
    runs = sorted((load_run(Path(d)) for d in run_dirs), key=lambda r: r["manifest"]["start_time"])
    for field in COMPARABLE_FIELDS:
        values = {json.dumps(r["manifest"]["config"].get(field), sort_keys=True) for r in runs}
        if len(values) > 1:
            raise ValueError(f"runs differ in config field {field!r}: {sorted(values)}")

    # Newest run wins per group.
    owner = {}
    for run in runs:
        for record in run["samples"]:
            if record["dataset"] in dataset_sizes:
                owner[group_key(record)] = run
    groups = sorted(owner, key=lambda g: (g[0], g[2], list(dataset_sizes).index(g[1])))

    def kept(run, records):
        return [
            r for r in records
            if owner.get(group_key(r)) is run and r["sample_idx"] < dataset_sizes[r["dataset"]]
        ]

    samples = [r for run in runs for r in kept(run, run["samples"])]
    steps = [r for run in runs for r in kept(run, run["steps"])]

    sample_idx = defaultdict(list)
    for record in samples:
        sample_idx[group_key(record)].append(record["sample_idx"])
    for key in groups:
        expected = list(range(dataset_sizes[key[1]]))
        if sorted(sample_idx[key]) != expected:
            raise ValueError(f"{key} from {owner[key]['dir']} does not cover sample_idx 0..{expected[-1]} exactly")

    out_dir.mkdir(parents=True, exist_ok=False)
    by_file = defaultdict(list)
    for kind, records in (("samples", samples), ("steps", steps)):
        for record in records:
            by_file[f"{kind}.{record['method']}.rank{record['rank']}.jsonl"].append(record)
    for name, records in by_file.items():
        with open(out_dir / name, "w", encoding="utf-8") as handle:
            handle.writelines(json.dumps(r) + "\n" for r in records)

    num_positions = {group_key(row): int(row["num_positions"]) for run in runs for row in run["summary"]}
    steps_by_key = defaultdict(list)
    for record in steps:
        steps_by_key[group_key(record)].append(record)
    summary = []
    for key in groups:
        method, dataset, k = key
        counts = counts_from_steps(
            steps_by_key[key], num_samples=len(sample_idx[key]), num_positions=num_positions[key]
        )
        summary.append({
            "method": method,
            "dataset": dataset,
            "k": k,
            "num_positions": num_positions[key],
            "metrics": acceptance_metrics(dataset_name=dataset, metric_summary=counts),
            "deepspec_metrics": deepspec_metrics_row(dataset_name=dataset, metric_summary=counts),
        })
    with open(out_dir / "summary.jsonl", "w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(row) + "\n" for row in summary)

    # Worker info (parameter counts, memory after loading) per method from the newest run that has it.
    (out_dir / "workers").mkdir()
    for run in runs:
        for path in (run["dir"] / "workers").glob("*.json"):
            shutil.copy(path, out_dir / "workers" / path.name)

    write_merged_manifest(runs, out_dir, dataset_sizes, owner)
    return summary


def write_merged_manifest(runs, out_dir: Path, dataset_sizes: dict[str, int], owner: dict) -> None:
    newest = runs[-1]["manifest"]
    methods = {}
    trained = {}
    draft_lengths = set()
    for run in runs:
        config = run["manifest"]["config"]
        methods.update({m["name"]: m for m in config["methods"]})
        trained.update(run["manifest"].get("trained_draft_lengths", {}))
        draft_lengths.update(config["draft_lengths"])
    commits = {r["manifest"]["git"]["commit"] for r in runs}
    manifest = {
        "run_id": out_dir.name,
        "start_time": newest["start_time"],
        "config_path": newest["config_path"],
        "opts": [],
        "config": {
            **newest["config"],
            "datasets": [[name, size] for name, size in dataset_sizes.items()],
            "draft_lengths": sorted(draft_lengths),
            "methods": list(methods.values()),
        },
        "trained_draft_lengths": trained,
        "git": {
            "commit": commits.pop() if len(commits) == 1 else None,
            "dirty": any(r["manifest"]["git"]["dirty"] for r in runs),
        },
        "versions": newest["versions"],
        "cuda_device_count": newest["cuda_device_count"],
        "merged_from": [
            {
                "run_id": run["manifest"]["run_id"],
                "git": run["manifest"]["git"],
                "opts": run["manifest"]["opts"],
                "groups": [list(key) for key, src in sorted(owner.items(), key=str) if src is run],
            }
            for run in runs
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--datasets", required=True, help="name=N,...; keeps sample_idx < N")
    parser.add_argument("run_dirs", nargs="+")
    args = parser.parse_args()
    dataset_sizes = {name: int(n) for name, n in (item.split("=") for item in args.datasets.split(","))}
    summary = merge_runs([Path(d) for d in args.run_dirs], Path(args.out), dataset_sizes)
    print(f"merged {len(summary)} groups into {args.out}")


if __name__ == "__main__":
    main()
