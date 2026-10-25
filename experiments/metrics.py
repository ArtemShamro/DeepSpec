"""Acceptance metrics from DeepSpec's per-step counts, and a parity check of a
run's `summary.jsonl` against its stored step records.

Two token counts per step are kept apart:
- accepted: draft tokens accepted by verification (`accepted`);
- committed: tokens appended to the output this step, i.e. accepted plus the
  bonus/corrected target token (`committed`).

Each summary row holds two metric blocks:
- `metrics`: computed here; `acceptance_length` = accepted draft tokens per
  proposal, so it never exceeds `draft_tokens_per_proposal`;
- `deepspec_metrics`: DeepSpec's `build_metrics_row` unchanged. Its
  `acceptance_length` is *committed* tokens per step (includes the bonus token).

Usage: python -m experiments.metrics <run_dir>
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def counts_from_steps(step_records: list[dict], *, num_samples: int, num_positions: int) -> dict:
    """Same counts as `BaseEvaluator.allreduce_response_metrics` (its key names)."""
    proposals_at_pos = [0] * num_positions
    accepted_at_pos = [0] * num_positions
    for r in step_records:
        for pos_idx in range(num_positions):
            if r["effective_len"] > pos_idx:
                proposals_at_pos[pos_idx] += 1
            if r["accepted"] > pos_idx:
                accepted_at_pos[pos_idx] += 1
    return {
        "sample_count": int(num_samples),
        "proposal_count": len(step_records),
        # DeepSpec's name; the per-step values are committed tokens.
        "acceptance_length_sum": sum(int(r["committed"]) for r in step_records),
        "proposal_length_sum": sum(int(r["effective_len"]) for r in step_records),
        "proposals_at_pos": proposals_at_pos,
        "accepted_at_pos": accepted_at_pos,
    }


def deepspec_metrics_row(*, dataset_name: str, metric_summary: dict) -> dict:
    """Same arithmetic as `BaseEvaluator.build_metrics_row`, so equal inputs give
    bit-identical floats."""
    num_positions = len(metric_summary["proposals_at_pos"])
    proposal_count = int(metric_summary["proposal_count"])
    if proposal_count == 0:
        return {
            "dataset": dataset_name,
            "num_samples": int(metric_summary["sample_count"]),
            "draft_tokens_per_proposal": 0.0,
            "acceptance_length": 0.0,
            "verify_rate": 0.0,
            "accept_rates_by_position": [None] * num_positions,
        }
    committed_sum = int(metric_summary["acceptance_length_sum"])
    proposal_length_sum = int(metric_summary["proposal_length_sum"])
    return {
        "dataset": dataset_name,
        "num_samples": int(metric_summary["sample_count"]),
        "draft_tokens_per_proposal": proposal_length_sum / proposal_count,
        "acceptance_length": committed_sum / proposal_count,
        "verify_rate": committed_sum / (proposal_length_sum + proposal_count),
        "accept_rates_by_position": _accept_rates_by_position(metric_summary),
    }


def acceptance_metrics(*, dataset_name: str, metric_summary: dict) -> dict:
    """Accepted-draft-token metrics from DeepSpec's (all-reduced) counts.

    Each step adds 1 to `accepted_at_pos[i]` for every `i < accepted`, so the
    sum over positions is the total number of accepted draft tokens.
    """
    proposal_count = int(metric_summary["proposal_count"])
    accepted_sum = sum(int(x) for x in metric_summary["accepted_at_pos"])
    committed_sum = int(metric_summary["acceptance_length_sum"])
    proposal_length_sum = int(metric_summary["proposal_length_sum"])
    assert 0 <= accepted_sum <= proposal_length_sum, (
        f"accepted draft tokens {accepted_sum} exceed proposed draft tokens {proposal_length_sum}"
    )
    assert accepted_sum <= committed_sum <= accepted_sum + proposal_count
    row = {
        "dataset": dataset_name,
        "num_samples": int(metric_summary["sample_count"]),
        "num_proposals": proposal_count,
        "draft_tokens_per_proposal": 0.0,
        "acceptance_length": 0.0,
        "committed_per_step": 0.0,
        "draft_acceptance_rate": None,
        "accept_rates_by_position": _accept_rates_by_position(metric_summary),
    }
    if proposal_count > 0:
        row["draft_tokens_per_proposal"] = proposal_length_sum / proposal_count
        row["acceptance_length"] = accepted_sum / proposal_count
        row["committed_per_step"] = committed_sum / proposal_count
    if proposal_length_sum > 0:
        row["draft_acceptance_rate"] = accepted_sum / proposal_length_sum
    return row


def _accept_rates_by_position(metric_summary: dict) -> list:
    return [
        None if proposed == 0 else accepted / proposed
        for proposed, accepted in zip(metric_summary["proposals_at_pos"], metric_summary["accepted_at_pos"])
    ]


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
        counts = counts_from_steps(
            steps_by_key[key],
            num_samples=samples_by_key[key],
            num_positions=int(row["num_positions"]),
        )
        if "deepspec_metrics" not in row:
            # Runs written before the split stored only DeepSpec's row, under `metrics`.
            expected = {"metrics": deepspec_metrics_row}
        else:
            expected = {"metrics": acceptance_metrics, "deepspec_metrics": deepspec_metrics_row}
        for field, compute in expected.items():
            recomputed = compute(dataset_name=row["dataset"], metric_summary=counts)
            if recomputed != row[field]:
                mismatches.append(f"{key} {field}: summary={row[field]} recomputed={recomputed}")
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
