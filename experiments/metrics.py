"""Recomputes DeepSpec's acceptance metrics row from stored step records, and
checks a run's parity against the rows in its `summary.jsonl`.

Usage: python -m experiments.metrics <run_dir>
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def recompute_metrics_row(
    *,
    dataset_name: str,
    num_samples: int,
    step_records: list[dict],
    num_positions: int,
) -> dict:
    """Same arithmetic as `BaseEvaluator.allreduce_response_metrics` +
    `build_metrics_row`, so equal inputs give bit-identical floats."""
    proposal_count = len(step_records)
    if proposal_count == 0:
        return {
            "dataset": dataset_name,
            "num_samples": int(num_samples),
            "draft_tokens_per_proposal": 0.0,
            "acceptance_length": 0.0,
            "verify_rate": 0.0,
            "accept_rates_by_position": [None] * num_positions,
        }
    acceptance_length_sum = sum(int(r["committed"]) for r in step_records)
    proposal_length_sum = sum(int(r["effective_len"]) for r in step_records)
    accept_rates_by_position = []
    for pos_idx in range(num_positions):
        proposed = sum(1 for r in step_records if r["effective_len"] > pos_idx)
        accepted = sum(1 for r in step_records if r["accepted"] > pos_idx)
        accept_rates_by_position.append(None if proposed == 0 else accepted / proposed)
    return {
        "dataset": dataset_name,
        "num_samples": int(num_samples),
        "draft_tokens_per_proposal": proposal_length_sum / proposal_count,
        "acceptance_length": acceptance_length_sum / proposal_count,
        "verify_rate": acceptance_length_sum / (proposal_length_sum + proposal_count),
        "accept_rates_by_position": accept_rates_by_position,
    }


def read_jsonl(paths) -> list[dict]:
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def check_parity(run_dir: str | Path) -> list[str]:
    """Returns a list of mismatch descriptions; empty means every summary row
    is reproduced exactly from the step records."""
    run_dir = Path(run_dir)
    samples = read_jsonl(sorted(run_dir.glob("samples.*.jsonl")))
    steps = read_jsonl(sorted(run_dir.glob("steps.*.jsonl")))
    summary = read_jsonl([run_dir / "summary.jsonl"])

    def group_key(record):
        return record["method"], record["dataset"], record["k"]

    samples_by_key = defaultdict(int)
    for record in samples:
        samples_by_key[group_key(record)] += 1
    steps_by_key = defaultdict(list)
    for record in steps:
        steps_by_key[group_key(record)].append(record)

    mismatches = []
    for row in summary:
        key = group_key(row)
        recomputed = recompute_metrics_row(
            dataset_name=row["dataset"],
            num_samples=samples_by_key[key],
            step_records=steps_by_key[key],
            num_positions=int(row["num_positions"]),
        )
        if recomputed != row["metrics"]:
            mismatches.append(f"{key}: summary={row['metrics']} recomputed={recomputed}")
    return mismatches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()
    mismatches = check_parity(args.run_dir)
    for mismatch in mismatches:
        print("MISMATCH", mismatch)
    if mismatches:
        raise SystemExit(1)
    print("parity OK")


if __name__ == "__main__":
    main()
