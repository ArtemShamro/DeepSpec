import random
from types import SimpleNamespace

import pytest
import torch

import deepspec.eval.base_evaluator as base_evaluator
from deepspec.eval.base_evaluator import BaseEvaluator
from experiments.metrics import recompute_metrics_row


def _deepspec_row(responses, num_positions, monkeypatch):
    """DeepSpec's own metric path, run single-process on CPU."""
    monkeypatch.setattr(base_evaluator.dist, "all_reduce", lambda *a, **k: None)
    fake = SimpleNamespace(max_proposal_tokens=num_positions, device=torch.device("cpu"))
    summary = BaseEvaluator.allreduce_response_metrics(fake, responses)
    return BaseEvaluator.build_metrics_row(fake, dataset_name="ds", metric_summary=summary)


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


@pytest.mark.parametrize("seed", range(5))
def test_matches_build_metrics_row(seed, monkeypatch):
    rng = random.Random(seed)
    k = rng.randint(1, 7)
    responses = [
        _random_response(rng, k, rng.randint(0, 30), stop_last=rng.random() < 0.5)
        for _ in range(rng.randint(1, 6))
    ]
    expected = _deepspec_row(responses, k, monkeypatch)
    got = recompute_metrics_row(
        dataset_name="ds",
        num_samples=len(responses),
        step_records=_step_records(responses),
        num_positions=k,
    )
    assert got == expected


def test_stop_token_case(monkeypatch):
    responses = [
        SimpleNamespace(acceptance_lengths=[4, 2], proposal_lengths=[3, 2], accepted_draft_lengths=[3, 2])
    ]
    expected = _deepspec_row(responses, 3, monkeypatch)
    got = recompute_metrics_row(
        dataset_name="ds", num_samples=1, step_records=_step_records(responses), num_positions=3
    )
    assert got == expected


def test_zero_draft_ar_case(monkeypatch):
    responses = [
        SimpleNamespace(acceptance_lengths=[1] * 5, proposal_lengths=[0] * 5, accepted_draft_lengths=[0] * 5),
        SimpleNamespace(acceptance_lengths=[], proposal_lengths=[], accepted_draft_lengths=[]),
    ]
    expected = _deepspec_row(responses, 1, monkeypatch)
    got = recompute_metrics_row(
        dataset_name="ds", num_samples=2, step_records=_step_records(responses), num_positions=1
    )
    assert got == expected
    assert got["acceptance_length"] == 1.0 and got["accept_rates_by_position"] == [None]


def test_no_steps(monkeypatch):
    responses = [SimpleNamespace(acceptance_lengths=[], proposal_lengths=[], accepted_draft_lengths=[])]
    expected = _deepspec_row(responses, 3, monkeypatch)
    got = recompute_metrics_row(dataset_name="ds", num_samples=1, step_records=[], num_positions=3)
    assert got == expected
