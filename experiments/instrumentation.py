"""Per-sample / per-step timing and memory capture around DeepSpec's decoding loop.

The evaluator's own callbacks are wrapped and passed to the unchanged
`generate_decoding_sample` (design D1). Boundaries are CUDA events recorded on
the current stream; they are resolved with a single synchronize at sample end,
so no host-device sync is added inside the loop (D2).

    sample start ─ prefill ─▶ init_context ─▶ │ propose │ verify ─▶ post_verify ─▶ commit ─▶ │ update │
                [target prefill] [draft prefill] [ draft ] [ verification ]                  [ draft ]
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import torch

from deepspec.eval.base_evaluator import generate_decoding_sample


@dataclass
class SampleTiming:
    target_prefill_ms: float
    draft_prefill_ms: float | None
    decode_ms: float | None
    wall_ms: float
    prefill_peak_bytes: int
    decode_peak_bytes: int | None
    # Per-step lists; None when step timing is off.
    step_draft_ms: list[float] | None
    step_verify_ms: list[float] | None


class _EventPool:
    """Reuses timing events across samples; safe because every sample ends with a sync."""

    def __init__(self):
        self._events: list[torch.cuda.Event] = []
        self._next = 0

    def reset(self) -> None:
        self._next = 0

    def record(self) -> torch.cuda.Event:
        if self._next == len(self._events):
            self._events.append(torch.cuda.Event(enable_timing=True))
        event = self._events[self._next]
        self._next += 1
        event.record()
        return event


# Indices into a step's event list.
_PROPOSE_START, _PROPOSE_END, _VERIFY_END, _UPDATE_START, _UPDATE_END = range(5)


class SampleInstrument:
    """Collects one sample's timing, memory and drafted lengths.

    Usage: `begin()`, run the loop with the callbacks from `wrap(...)`, `end()`.
    """

    def __init__(self, step_timing: bool = True):
        self.step_timing = step_timing
        self._pool = _EventPool()

    def begin(self) -> None:
        self._pool.reset()
        self.drafted_lengths: list[int] = []
        self._steps: list[list[torch.cuda.Event | None]] = []
        self._prefill_end = None
        self._draft_prefill_end = None
        self._prefill_peak_bytes = None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self._wall_start = time.perf_counter()
        self._start = self._pool.record()

    def wrap(
        self,
        *,
        init_context: Callable[..., Any],
        propose: Callable[..., Any],
        update: Callable[..., None],
        post_verify: Callable[..., None] | None,
    ):
        pool = self._pool

        def timed_init_context(**kwargs):
            self._prefill_end = pool.record()
            context = init_context(**kwargs)
            self._draft_prefill_end = pool.record()
            self._prefill_peak_bytes = torch.cuda.max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            return context

        def recording_post_verify(proposal, verification):
            if self.step_timing:
                self._steps[-1][_VERIFY_END] = pool.record()
            self.drafted_lengths.append(int(proposal.draft_token_count))
            if post_verify is not None:
                post_verify(proposal, verification)

        if not self.step_timing:
            return timed_init_context, propose, update, recording_post_verify

        def timed_propose(**kwargs):
            step = [pool.record(), None, None, None, None]
            self._steps.append(step)
            proposal = propose(**kwargs)
            step[_PROPOSE_END] = pool.record()
            return proposal

        def timed_update(context, verification):
            step = self._steps[-1]
            step[_UPDATE_START] = pool.record()
            update(context, verification)
            step[_UPDATE_END] = pool.record()

        return timed_init_context, timed_propose, timed_update, recording_post_verify

    def end(self) -> SampleTiming:
        end = self._pool.record()
        peak_bytes = torch.cuda.max_memory_allocated()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - self._wall_start) * 1e3

        if self._draft_prefill_end is None:
            # The first sampled token was a stop token: no draft prefill, no decode.
            return SampleTiming(
                target_prefill_ms=self._start.elapsed_time(end),
                draft_prefill_ms=None,
                decode_ms=None,
                wall_ms=wall_ms,
                prefill_peak_bytes=peak_bytes,
                decode_peak_bytes=None,
                step_draft_ms=[] if self.step_timing else None,
                step_verify_ms=[] if self.step_timing else None,
            )

        step_draft_ms = step_verify_ms = None
        if self.step_timing:
            step_draft_ms, step_verify_ms = [], []
            for step in self._steps:
                draft_ms = step[_PROPOSE_START].elapsed_time(step[_PROPOSE_END])
                if step[_UPDATE_START] is not None:
                    draft_ms += step[_UPDATE_START].elapsed_time(step[_UPDATE_END])
                step_draft_ms.append(draft_ms)
                step_verify_ms.append(step[_PROPOSE_END].elapsed_time(step[_VERIFY_END]))

        return SampleTiming(
            target_prefill_ms=self._start.elapsed_time(self._prefill_end),
            draft_prefill_ms=self._prefill_end.elapsed_time(self._draft_prefill_end),
            decode_ms=self._draft_prefill_end.elapsed_time(end),
            wall_ms=wall_ms,
            prefill_peak_bytes=self._prefill_peak_bytes,
            decode_peak_bytes=peak_bytes,
            step_draft_ms=step_draft_ms,
            step_verify_ms=step_verify_ms,
        )


class InstrumentedMixin:
    """Overrides `generate_one_sample` to run the evaluator's own callbacks
    through `SampleInstrument`. Generation arguments are the ones DeepSpec's
    evaluators pass.

    Attributes set by the caller:
    - `step_timing`: per-step CUDA events on/off (sample-level timing stays on);
    - `sample_sink`: optional callable invoked with every finished response.
    """

    step_timing: bool = True
    sample_sink: Callable[[SimpleNamespace], None] | None = None

    @property
    def requested_draft_length(self) -> int:
        return int(getattr(self, "k", 0))

    def generate_one_sample(
        self,
        *,
        input_ids: torch.Tensor,
        stop_token_ids: list[int] | None,
    ) -> SimpleNamespace:
        instrument = self.__dict__.get("_sample_instrument")
        if instrument is None:
            instrument = self._sample_instrument = SampleInstrument()
        instrument.step_timing = bool(self.step_timing)

        instrument.begin()
        init_context, propose, update, post_verify = instrument.wrap(
            init_context=self._init_context,
            propose=self._propose,
            update=self._update,
            post_verify=getattr(self, "_post_verify", None),
        )
        response = generate_decoding_sample(
            target_model=self.target_model,
            input_ids=input_ids,
            max_new_tokens=int(self.args.max_new_tokens),
            max_proposal_tokens=self.max_proposal_tokens,
            temperature=float(self.args.temperature),
            stop_token_ids=stop_token_ids,
            init_context=init_context,
            propose=propose,
            update=update,
            post_verify=post_verify,
        )
        response.timing = instrument.end()
        response.drafted_lengths = list(instrument.drafted_lengths)
        response.requested_draft_length = self.requested_draft_length
        response.stopped_by_stop_token = bool(
            stop_token_ids
            and response.num_output_tokens > 0
            and int(response.output_ids[0, -1].item()) in stop_token_ids
        )
        if self.sample_sink is not None:
            self.sample_sink(response)
        return response
