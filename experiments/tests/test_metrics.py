import random
from types import SimpleNamespace

import pytest
import torch

import deepspec.eval.base_evaluator as base_evaluator
from deepspec.eval.base_evaluator import BaseEvaluator
from experiments.metrics import acceptance_metrics, counts_from_steps, deepspec_metrics_row


def _deepspec(responses, num_positions, monkeypatch):
    """DeepSpec's own metric path, run single-process on CPU: (counts, row)."""
    monkeypatch.setattr(base_evaluator.dist, "all_reduce", lambda *a, **k: None)
    fake = SimpleNamespace(max_proposal_tokens=num_positions, device=torch.device("cpu"))
    summary = BaseEvaluator.allreduce_response_metrics(fake, responses)
    return summary, BaseEvaluator.build_metrics_row(fake, dataset_name="ds", metric_summary=summary)


def _step_records(responses):
    return [
        {"committed": a, "effective_len": p, "accepted": d}
        for r in responses
        for a, p, d in zip(r.acceptance_lengths, r.proposal_lengths, r.accepted_draft_lengths)
    ]


def _random_response(rng, k, num_steps, stop_last):
    acc, prop, accepted = [], [], []
    for step in range(num_steps):
        proposal = rng.randint(0, k)
        a = rng.randint(0, proposal)
        if stop_last and step == num_steps - 1 and a > 0:
            # Stop token at accepted position a-1: effective = accepted = committed.
            acc.append(a)
            prop.append(a)
        else:
            acc.append(a + 1)
            prop.append(proposal)
        accepted.append(a)
    return SimpleNamespace(acceptance_lengths=acc, proposal_lengths=prop, accepted_draft_lengths=accepted)


def _check(responses, num_positions, monkeypatch):
    """Recomputed counts and DeepSpec row equal DeepSpec's; returns our metrics."""
    deepspec_counts, deepspec_row = _deepspec(responses, num_positions, monkeypatch)
    counts = counts_from_steps(
        _step_records(responses), num_samples=len(responses), num_positions=num_positions
    )
    assert counts == deepspec_counts
    assert deepspec_metrics_row(dataset_name="ds", metric_summary=counts) == deepspec_row
    got = acceptance_metrics(dataset_name="ds", metric_summary=counts)

    accepted = sum(sum(r.accepted_draft_lengths) for r in responses)
    committed = sum(sum(r.acceptance_lengths) for r in responses)
    proposals = sum(len(r.proposal_lengths) for r in responses)
    assert got["num_proposals"] == proposals
    if proposals:
        assert got["acceptance_length"] == accepted / proposals
        assert got["committed_per_step"] == committed / proposals
        # DeepSpec's acceptance_length is committed-based.
        assert deepspec_row["acceptance_length"] == got["committed_per_step"]
    assert 0 <= got["acceptance_length"] <= got["draft_tokens_per_proposal"] <= num_positions
    assert got["draft_tokens_per_proposal"] == deepspec_row["draft_tokens_per_proposal"]
    assert got["accept_rates_by_position"] == deepspec_row["accept_rates_by_position"]
    return got


@pytest.mark.parametrize("seed", range(5))
def test_random_responses(seed, monkeypatch):
    rng = random.Random(seed)
    k = rng.randint(1, 7)
    responses = [
        _random_response(rng, k, rng.randint(0, 30), stop_last=rng.random() < 0.5)
        for _ in range(rng.randint(1, 6))
    ]
    _check(responses, k, monkeypatch)


def test_bonus_token_excluded(monkeypatch):
    # Every draft token accepted plus a bonus token: committed = K + 1 per step,
    # acceptance length = K.
    responses = [
        SimpleNamespace(acceptance_lengths=[4, 4], proposal_lengths=[3, 3], accepted_draft_lengths=[3, 3])
    ]
    got = _check(responses, 3, monkeypatch)
    assert got["acceptance_length"] == 3.0 == got["draft_tokens_per_proposal"]
    assert got["committed_per_step"] == 4.0
    assert got["draft_acceptance_rate"] == 1.0


def test_corrected_token_excluded(monkeypatch):
    # All drafts rejected: only the corrected target token is committed.
    responses = [
        SimpleNamespace(acceptance_lengths=[1, 1], proposal_lengths=[3, 3], accepted_draft_lengths=[0, 0])
    ]
    got = _check(responses, 3, monkeypatch)
    assert got["acceptance_length"] == 0.0
    assert got["committed_per_step"] == 1.0
    assert got["draft_acceptance_rate"] == 0.0


def test_stop_token_case(monkeypatch):
    responses = [
        SimpleNamespace(acceptance_lengths=[4, 2], proposal_lengths=[3, 2], accepted_draft_lengths=[3, 2])
    ]
    got = _check(responses, 3, monkeypatch)
    assert got["acceptance_length"] == 2.5 and got["committed_per_step"] == 3.0
    assert got["draft_acceptance_rate"] == 1.0


def test_zero_draft_ar_case(monkeypatch):
    responses = [
        SimpleNamespace(acceptance_lengths=[1] * 5, proposal_lengths=[0] * 5, accepted_draft_lengths=[0] * 5),
        SimpleNamespace(acceptance_lengths=[], proposal_lengths=[], accepted_draft_lengths=[]),
    ]
    got = _check(responses, 1, monkeypatch)
    assert got["acceptance_length"] == 0.0 and got["committed_per_step"] == 1.0
    assert got["draft_acceptance_rate"] is None and got["accept_rates_by_position"] == [None]


def test_no_steps(monkeypatch):
    responses = [SimpleNamespace(acceptance_lengths=[], proposal_lengths=[], accepted_draft_lengths=[])]
    got = _check(responses, 3, monkeypatch)
    assert got["acceptance_length"] == 0.0 and got["num_proposals"] == 0


def test_rejects_accepted_above_proposed():
    counts = counts_from_steps(
        [{"committed": 4, "effective_len": 2, "accepted": 3}], num_samples=1, num_positions=3
    )
    with pytest.raises(AssertionError):
        acceptance_metrics(dataset_name="ds", metric_summary=counts)
