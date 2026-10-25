"""Model-level GPU tests: K control, AR baseline, parity with DeepSpec's own
evaluators, instrumentation transparency and per-phase memory.

Tests are grouped by method so the shared evaluator cache reloads rarely.
"""

from contextlib import contextmanager

import pytest
import torch

import deepspec.eval.dspark.evaluator as dspark_evaluator_module
from deepspec.eval.base_evaluator import resolve_stop_token_ids
from deepspec.eval.dspark import Qwen3DSparkEvaluator
from deepspec.eval.eagle3 import Qwen3Eagle3Evaluator
from experiments.evaluators import ARBaselineEvaluator, DSparkKEvaluator, Eagle3KEvaluator
from experiments.records import records_from_response

pytestmark = pytest.mark.gpu

PLAIN_CLASSES = {
    "ar": ARBaselineEvaluator,
    "eagle3": Eagle3KEvaluator,
    "dflash": DSparkKEvaluator,
    "dspark": DSparkKEvaluator,
}


@contextmanager
def as_class(evaluator, cls):
    """Runs the same loaded models through another evaluator class."""
    original = evaluator.__class__
    evaluator.__class__ = cls
    try:
        yield evaluator
    finally:
        evaluator.__class__ = original


def run_gsm8k(evaluator, n=2):
    return evaluator.run_dataset(dataset_name="gsm8k", max_samples=n)


def step_lists(responses):
    return [
        (
            r.output_ids.tolist(),
            r.acceptance_lengths,
            r.proposal_lengths,
            r.accepted_draft_lengths,
        )
        for r in responses
    ]


# ---------------------------------------------------------------- DSpark / DFlash


def test_dspark_k3_block_and_proposals(evaluators, monkeypatch):
    evaluator = evaluators.get("dspark")
    evaluator.set_k(3)
    block_shapes = []
    original_forward = dspark_evaluator_module.forward_dspark_draft_block

    def spy(model, *, draft_input_ids, block_size, **kwargs):
        block_shapes.append((draft_input_ids.shape[1], block_size))
        return original_forward(model, draft_input_ids=draft_input_ids, block_size=block_size, **kwargs)

    monkeypatch.setattr(dspark_evaluator_module, "forward_dspark_draft_block", spy)
    responses = run_gsm8k(evaluator)
    assert block_shapes and set(block_shapes) == {(3, 3)}
    for response in responses:
        # Confidence threshold 0 keeps DeepSpec's full-length proposals.
        assert set(response.drafted_lengths) == {3}
        assert max(response.proposal_lengths) <= 3


def test_dspark_k_above_trained_length(evaluators):
    evaluator = evaluators.get("dspark")
    with pytest.raises(ValueError, match=r"'dspark'.*K=8.*7"):
        evaluator.set_k(8)
    with pytest.raises(ValueError, match="K=0"):
        evaluator.set_k(0)


def test_dspark_confidence_recorder_disabled(evaluators):
    evaluator = evaluators.get("dspark")
    assert evaluator.draft_model.confidence_head is not None
    assert evaluator.confidence_head_recorder is None


def test_dspark_k7_matches_deepspec_evaluator(evaluators):
    evaluator = evaluators.get("dspark")
    evaluator.set_k(7)
    with as_class(evaluator, DSparkKEvaluator):
        ours = run_gsm8k(evaluator)
    with as_class(evaluator, Qwen3DSparkEvaluator):
        # Exactly what DeepSpec's eval runs, including its calibration recorder.
        evaluator.confidence_head_recorder = Qwen3DSparkEvaluator._build_confidence_head_recorder(evaluator)
        assert evaluator.confidence_head_recorder is not None
        evaluator.confidence_head_recorder.start()
        try:
            reference = run_gsm8k(evaluator)
        finally:
            evaluator.confidence_head_recorder = None
    assert step_lists(ours) == step_lists(reference)


@pytest.mark.parametrize("method", ["dspark", "dflash"])
@pytest.mark.parametrize("step_timing", [True, False])
def test_dspark_family_instrumentation_transparent(evaluators, method, step_timing):
    _check_instrumentation_transparent(evaluators, method, step_timing, k=3)


# ---------------------------------------------------------------- EAGLE-3


def test_eagle3_k3_proposals(evaluators):
    evaluator = evaluators.get("eagle3")
    evaluator.set_k(3)
    responses = run_gsm8k(evaluator)
    for response in responses:
        assert response.drafted_lengths and max(response.drafted_lengths) <= 3
        assert max(response.proposal_lengths) <= 3


def test_eagle3_k_above_trained_length(evaluators):
    evaluator = evaluators.get("eagle3")
    with pytest.raises(ValueError, match=r"'eagle3'.*K=8.*7"):
        evaluator.set_k(8)


def test_eagle3_k7_matches_deepspec_evaluator(evaluators):
    evaluator = evaluators.get("eagle3")
    evaluator.set_k(7)
    with as_class(evaluator, Eagle3KEvaluator):
        ours = run_gsm8k(evaluator)
    with as_class(evaluator, Qwen3Eagle3Evaluator):
        reference = run_gsm8k(evaluator)
    assert step_lists(ours) == step_lists(reference)


@pytest.mark.parametrize("step_timing", [True, False])
def test_eagle3_instrumentation_transparent(evaluators, step_timing):
    _check_instrumentation_transparent(evaluators, "eagle3", step_timing, k=3)


# ---------------------------------------------------------------- AR baseline


def test_ar_baseline_steps(evaluators):
    evaluator = evaluators.get("ar")
    assert evaluator.draft_model is None
    max_new_tokens = int(evaluator.args.max_new_tokens)
    for idx, response in enumerate(run_gsm8k(evaluator, n=3)):
        assert set(response.accepted_draft_lengths) <= {0}
        assert set(response.proposal_lengths) <= {0}
        assert set(response.acceptance_lengths) <= {1}
        assert set(response.drafted_lengths) <= {0}
        assert response.stopped_by_stop_token or response.num_output_tokens == max_new_tokens
        # Enforces 1 + sum(committed) vs generated length.
        sample, steps = records_from_response(
            response,
            key={"method": "ar", "dataset": "gsm8k", "k": 0, "sample_idx": idx, "rank": 0},
            max_new_tokens=max_new_tokens,
        )
        assert all(s["requested_len"] == 0 for s in steps)


def test_ar_baseline_stops_on_stop_token(evaluators):
    evaluator = evaluators.get("ar", max_new_tokens=1024)
    responses = run_gsm8k(evaluator, n=2)
    stop_token_ids = resolve_stop_token_ids(evaluator.target_model, evaluator.tokenizer)
    for response in responses:
        assert response.stopped_by_stop_token
        assert int(response.output_ids[0, -1]) in stop_token_ids
        assert 1 + sum(response.acceptance_lengths) == response.num_output_tokens


@pytest.mark.parametrize("step_timing", [True, False])
def test_ar_instrumentation_transparent(evaluators, step_timing):
    _check_instrumentation_transparent(evaluators, "ar", step_timing, k=None)


def test_decode_peak_excludes_prefill(evaluators):
    evaluator = evaluators.get("ar")
    evaluator.args.max_new_tokens = 8
    try:
        torch.manual_seed(0)
        input_ids = torch.randint(1000, 20000, (1, 6000), device=evaluator.device)
        response = evaluator.generate_one_sample(input_ids=input_ids, stop_token_ids=None)
    finally:
        evaluator.args.max_new_tokens = 128
    timing = response.timing
    assert timing.decode_peak_bytes is not None
    assert timing.decode_peak_bytes < timing.prefill_peak_bytes
    # The decode peak still includes the resident model weights.
    assert timing.decode_peak_bytes > 7e9


# ---------------------------------------------------------------- shared


def _check_instrumentation_transparent(evaluators, method, step_timing, k):
    evaluator = evaluators.get(method)
    if k is not None:
        evaluator.set_k(k)
    evaluator.step_timing = step_timing
    instrumented = run_gsm8k(evaluator)
    with as_class(evaluator, PLAIN_CLASSES[method]):
        plain = run_gsm8k(evaluator)
    assert step_lists(instrumented) == step_lists(plain)
    for response in instrumented:
        assert len(response.drafted_lengths) == len(response.acceptance_lengths)
        if step_timing:
            assert len(response.timing.step_draft_ms) == len(response.acceptance_lengths)
        else:
            assert response.timing.step_draft_ms is None
