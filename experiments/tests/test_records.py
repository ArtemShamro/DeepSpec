import pytest

from experiments.records import build_records

KEY = {"method": "dspark", "dataset": "gsm8k", "k": 3, "sample_idx": 0, "rank": 0}


def _timing(num_steps, step_timing=True):
    return {
        "target_prefill_ms": 10.0,
        "draft_prefill_ms": 1.0,
        "decode_ms": 50.0,
        "wall_ms": 62.0,
        "prefill_peak_bytes": 100,
        "decode_peak_bytes": 90,
        "step_draft_ms": [1.0] * num_steps if step_timing else None,
        "step_verify_ms": [2.0] * num_steps if step_timing else None,
    }


def _build(**overrides):
    kwargs = dict(
        key=KEY,
        prompt_len=20,
        generated_len=8,
        stopped_by_stop_token=True,
        max_new_tokens=100,
        requested_draft_length=3,
        # 4 + 2 + 1 committed + 1 prefill token = 8
        acceptance_lengths=[4, 2, 1],
        proposal_lengths=[3, 3, 3],
        accepted_draft_lengths=[3, 1, 0],
        drafted_lengths=[3, 3, 3],
        timing=_timing(3),
    )
    kwargs.update(overrides)
    return build_records(**kwargs)


def test_step_count_and_fields():
    sample, steps = _build()
    assert sample["num_steps"] == len(steps) == 3
    assert sample["stop_reason"] == "stop_token"
    assert [s["step"] for s in steps] == [0, 1, 2]
    assert steps[0] == {
        **KEY,
        "step": 0,
        "requested_len": 3,
        "drafted_len": 3,
        "effective_len": 3,
        "accepted": 3,
        "committed": 4,
        "draft_ms": 1.0,
        "verify_ms": 2.0,
    }
    assert 1 + sum(s["committed"] for s in steps) == sample["generated_len"]


def test_committed_mismatch_rejected():
    with pytest.raises(AssertionError):
        _build(generated_len=9)


def test_stop_token_inside_accepted_draft():
    # Last step: drafter produced 3, the 2nd accepted token is a stop token, so
    # effective = accepted = committed = 2 while drafted stays 3.
    sample, steps = _build(
        generated_len=7,
        acceptance_lengths=[4, 2],
        proposal_lengths=[3, 2],
        accepted_draft_lengths=[3, 2],
        drafted_lengths=[3, 3],
        timing=_timing(2),
    )
    last = steps[-1]
    assert (last["drafted_len"], last["effective_len"], last["accepted"], last["committed"]) == (3, 2, 2, 2)
    assert sample["stop_reason"] == "stop_token"


def test_length_limit_flagged():
    # Committed tokens overshoot the limit; output is cut at max_new_tokens.
    sample, _ = _build(stopped_by_stop_token=False, max_new_tokens=6, generated_len=6)
    assert sample["stop_reason"] == "length"
    assert sample["generated_len"] == 6


def test_length_limited_must_reach_limit():
    with pytest.raises(AssertionError):
        _build(stopped_by_stop_token=False, max_new_tokens=100)


def test_step_timing_off():
    _, steps = _build(timing=_timing(3, step_timing=False))
    assert all(s["draft_ms"] is None and s["verify_ms"] is None for s in steps)


def test_zero_step_sample():
    sample, steps = _build(
        generated_len=1,
        acceptance_lengths=[],
        proposal_lengths=[],
        accepted_draft_lengths=[],
        drafted_lengths=[],
        timing=_timing(0),
    )
    assert sample["num_steps"] == 0 and steps == []
