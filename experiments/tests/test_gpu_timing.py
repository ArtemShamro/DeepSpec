"""Phase attribution of SampleInstrument on known-duration GPU kernels.

The toy callbacks enqueue `torch.cuda._sleep` kernels and return immediately,
so the host runs ahead of the GPU exactly as in an asynchronous draft forward.
"""

from types import SimpleNamespace

import pytest
import torch

from experiments.instrumentation import SampleInstrument

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def sleep_ms():
    cycles = 1_000_000
    torch.cuda._sleep(cycles)  # warm-up
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    torch.cuda._sleep(cycles * 10)
    end.record()
    torch.cuda.synchronize()
    cycles_per_ms = cycles * 10 / start.elapsed_time(end)

    def sleep(ms):
        torch.cuda._sleep(int(ms * cycles_per_ms))

    return sleep


def _close(measured, expected):
    return abs(measured - expected) <= 0.1 * expected + 0.3


def _run_toy_sample(sleep, step_timing):
    instrument = SampleInstrument(step_timing=step_timing)

    def init_context(**kwargs):
        sleep(2)
        return SimpleNamespace()

    def propose(**kwargs):
        sleep(3)
        return SimpleNamespace(draft_token_count=3)

    def update(context, verification):
        # EAGLE-3-style drafter cache extension.
        sleep(4)

    instrument.begin()
    sleep(5)  # target prefill
    init_context, propose, update, post_verify = instrument.wrap(
        init_context=init_context, propose=propose, update=update, post_verify=None
    )
    context = init_context(initial_output=None)
    for step in range(2):
        proposal = propose(context=context)
        sleep(6)  # verification
        post_verify(proposal, None)
        sleep(1)  # commit / KV crop
        if step == 0:  # the final stop-token step skips update
            update(context, None)
    return instrument, instrument.end()


def test_phase_attribution(sleep_ms):
    instrument, timing = _run_toy_sample(sleep_ms, step_timing=True)
    assert _close(timing.target_prefill_ms, 5)
    assert _close(timing.draft_prefill_ms, 2)
    assert _close(timing.decode_ms, (3 + 6 + 1 + 4) + (3 + 6 + 1))
    assert all(_close(m, e) for m, e in zip(timing.step_draft_ms, [3 + 4, 3]))
    assert all(_close(m, 6) for m in timing.step_verify_ms)
    assert instrument.drafted_lengths == [3, 3]
    assert timing.wall_ms >= timing.target_prefill_ms + timing.draft_prefill_ms + timing.decode_ms - 0.5


def test_step_timing_off_keeps_sample_timing(sleep_ms):
    instrument, timing = _run_toy_sample(sleep_ms, step_timing=False)
    assert timing.step_draft_ms is None and timing.step_verify_ms is None
    assert _close(timing.target_prefill_ms, 5)
    assert _close(timing.decode_ms, 24)
    assert instrument.drafted_lengths == [3, 3]
