import pytest

from experiments.records import STEP_PHASES, build_records

KEY = {"method": "dspark", "dataset": "gsm8k", "k": 3, "sample_idx": 0, "rank": 0}


def _timing(num_steps, step_timing=True):
    return {
        "target_prefill_ms": 10.0,
        "init_context_ms": 1.0,
        "wall_ms": 62.0,
        "prefill_peak_bytes": 100,
        "decode_peak_bytes": 90,
        "step_draft_ms": [1.0 + i for i in range(num_steps)] if step_timing else None,
        "step_verification_ms": [2.0] * num_steps if step_timing else None,
        "step_commit_ms": [0.5] * num_steps if step_timing else None,
        # The final stop-token step skips update.
        "step_update_ms": ([0.25] * (num_steps - 1) + [0.0])[:num_steps] if step_timing else None,
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
        "verification_ms": 2.0,
        "commit_ms": 0.5,
        "update_ms": 0.25,
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
    sample, steps = _build(timing=_timing(3, step_timing=False))
    for phase in STEP_PHASES:
        assert sample[phase] is None
        assert all(s[phase] is None for s in steps)
    assert sample["target_prefill_ms"] == 10.0 and sample["wall_ms"] == 62.0


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


@pytest.mark.parametrize(
    "overrides",
    [
        # accepted > effective_len
        dict(accepted_draft_lengths=[3, 4, 0], acceptance_lengths=[4, 5, 1], generated_len=11),
        # effective_len > drafted_len
        dict(drafted_lengths=[3, 2, 3]),
        # negative accepted
        dict(accepted_draft_lengths=[3, 1, -1], acceptance_lengths=[4, 2, 0], generated_len=7),
        # committed = accepted + 2
        dict(acceptance_lengths=[5, 1, 1]),
    ],
)
def test_step_accounting_invariants(overrides):
    with pytest.raises(AssertionError):
        _build(**overrides)


def test_zero_draft_step_invariants():
    # AR step: nothing drafted or accepted, one target token committed.
    _, steps = _build(
        requested_draft_length=0,
        generated_len=4,
        acceptance_lengths=[1, 1, 1],
        proposal_lengths=[0, 0, 0],
        accepted_draft_lengths=[0, 0, 0],
        drafted_lengths=[0, 0, 0],
    )
    assert all((s["drafted_len"], s["accepted"], s["committed"]) == (0, 0, 1) for s in steps)


def test_sample_phases_are_step_sums():
    sample, steps = _build()
    assert sample["draft_ms"] == 1.0 + 2.0 + 3.0
    assert sample["verification_ms"] == 6.0
    assert sample["commit_ms"] == 1.5
    assert sample["update_ms"] == 0.5  # final step skipped update
    for phase in STEP_PHASES:
        assert sample[phase] == sum(s[phase] for s in steps)
    assert sample["init_context_ms"] == 1.0
    for removed in ("draft_prefill_ms", "decode_ms", "verify_ms"):
        assert removed not in sample and all(removed not in s for s in steps)


def test_zero_step_sample_with_step_timing():
    sample, _ = _build(
        generated_len=1,
        acceptance_lengths=[],
        proposal_lengths=[],
        accepted_draft_lengths=[],
        drafted_lengths=[],
        timing=_timing(0),
    )
    assert all(sample[phase] == 0 for phase in STEP_PHASES)


def test_partial_step_timing_rejected():
    timing = _timing(3)
    timing["step_commit_ms"] = None
    with pytest.raises(AssertionError):
        _build(timing=timing)
