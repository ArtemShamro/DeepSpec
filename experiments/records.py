"""Builds per-sample and per-step records from a decoded response (design D3).

Accounting (accepted / effective / committed) comes from DeepSpec's own
per-step lists, the same inputs `build_metrics_row` consumes; the
instrumentation only adds the drafted length and timing.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from types import SimpleNamespace


def build_records(
    *,
    key: dict,
    prompt_len: int,
    generated_len: int,
    stopped_by_stop_token: bool,
    max_new_tokens: int,
    requested_draft_length: int,
    acceptance_lengths: list[int],
    proposal_lengths: list[int],
    accepted_draft_lengths: list[int],
    drafted_lengths: list[int],
    timing: dict,
    extra: dict | None = None,
) -> tuple[dict, list[dict]]:
    """Returns `(sample_record, step_records)`.

    `key` identifies the sample (method, dataset, k, sample_idx, rank) and is
    copied into every record. `timing` is a `SampleTiming` as a dict.
    """
    num_steps = len(acceptance_lengths)
    assert num_steps == len(proposal_lengths) == len(accepted_draft_lengths) == len(
        drafted_lengths
    ), "per-step lists must have equal length"
    step_draft_ms = timing["step_draft_ms"]
    step_verify_ms = timing["step_verify_ms"]
    if step_draft_ms is not None:
        assert len(step_draft_ms) == len(step_verify_ms) == num_steps

    length_limited = not stopped_by_stop_token
    committed_total = 1 + sum(int(x) for x in acceptance_lengths)
    if length_limited:
        assert generated_len == max_new_tokens, (
            f"sample without a stop token must reach max_new_tokens={max_new_tokens}, "
            f"got generated length {generated_len}"
        )
        assert committed_total >= generated_len
    else:
        assert committed_total == generated_len, (
            f"1 + sum(committed)={committed_total} != generated length {generated_len}"
        )

    sample_record = {
        **key,
        **(extra or {}),
        "prompt_len": int(prompt_len),
        "generated_len": int(generated_len),
        "num_steps": num_steps,
        "stop_reason": "length" if length_limited else "stop_token",
        "target_prefill_ms": timing["target_prefill_ms"],
        "draft_prefill_ms": timing["draft_prefill_ms"],
        "decode_ms": timing["decode_ms"],
        "wall_ms": timing["wall_ms"],
        "prefill_peak_bytes": timing["prefill_peak_bytes"],
        "decode_peak_bytes": timing["decode_peak_bytes"],
    }
    step_records = [
        {
            **key,
            "step": step,
            "requested_len": int(requested_draft_length),
            "drafted_len": int(drafted_lengths[step]),
            "effective_len": int(proposal_lengths[step]),
            "accepted": int(accepted_draft_lengths[step]),
            "committed": int(acceptance_lengths[step]),
            "draft_ms": None if step_draft_ms is None else step_draft_ms[step],
            "verify_ms": None if step_verify_ms is None else step_verify_ms[step],
        }
        for step in range(num_steps)
    ]
    return sample_record, step_records


def records_from_response(
    response: SimpleNamespace,
    *,
    key: dict,
    max_new_tokens: int,
) -> tuple[dict, list[dict]]:
    prompt_ids = response.output_ids[0, : response.num_input_tokens].tolist()
    prompt_sha1 = hashlib.sha1(str(prompt_ids).encode()).hexdigest()
    return build_records(
        key=key,
        prompt_len=response.num_input_tokens,
        generated_len=response.num_output_tokens,
        stopped_by_stop_token=response.stopped_by_stop_token,
        max_new_tokens=max_new_tokens,
        requested_draft_length=response.requested_draft_length,
        acceptance_lengths=response.acceptance_lengths,
        proposal_lengths=response.proposal_lengths,
        accepted_draft_lengths=response.accepted_draft_lengths,
        drafted_lengths=response.drafted_lengths,
        timing=asdict(response.timing),
        extra={"prompt_sha1": prompt_sha1},
    )
