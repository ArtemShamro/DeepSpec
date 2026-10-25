"""Sweep evaluators: DeepSpec evaluators with a settable draft length K, plus
an autoregressive (zero-draft) baseline that runs through the same loop.

Nothing here changes how DeepSpec drafts or verifies; the subclasses only
redirect `max_proposal_tokens` to a mutable `k` (design D4/D5).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from deepspec.eval.base_evaluator import (
    BaseEvaluator,
    DraftProposal,
    VerificationResult,
    generate_decoding_sample,
)
from deepspec.eval.dspark import Qwen3DSparkEvaluator
from deepspec.eval.eagle3 import Qwen3Eagle3Evaluator

from experiments.instrumentation import InstrumentedMixin


def check_draft_length(method: str, k: int, trained_length: int) -> None:
    if not 1 <= int(k) <= int(trained_length):
        raise ValueError(
            f"method {method!r}: draft length K={k} is outside [1, {trained_length}]; "
            f"the checkpoint was trained with draft length {trained_length}."
        )


class _SettableKMixin:
    """`max_proposal_tokens` returns `self.k`, which defaults to the trained length."""

    def _init_k(self) -> None:
        self.k = self.trained_draft_length

    @property
    def max_proposal_tokens(self) -> int:
        return int(self.k)

    def set_k(self, k: int) -> None:
        check_draft_length(self.args.method, k, self.trained_draft_length)
        self.k = int(k)


class DSparkKEvaluator(_SettableKMixin, Qwen3DSparkEvaluator):
    """DSpark/DFlash with a `K`-sized draft block (anchor + K-1 mask slots)."""

    def __init__(self, local_rank: int, args):
        super().__init__(local_rank, args)
        self._init_k()

    @property
    def trained_draft_length(self) -> int:
        return int(self.draft_model.block_size)

    def _build_confidence_head_recorder(self):
        # Diagnostic-only; adds work to the decoding loop.
        return None


class Eagle3KEvaluator(_SettableKMixin, Qwen3Eagle3Evaluator):
    """EAGLE-3 with `K` chained draft steps."""

    def __init__(self, local_rank: int, args):
        super().__init__(local_rank, args)
        self._init_k()

    @property
    def trained_draft_length(self) -> int:
        return int(self.draft_model.ttt_length)


class ARBaselineEvaluator(BaseEvaluator):
    """Target-only baseline: every step proposes zero draft tokens."""

    EVAL_ATTN_IMPLEMENTATION = "sdpa"

    @property
    def max_proposal_tokens(self) -> int:
        # generate_decoding_sample asserts >= 1 and sizes its buffer from it.
        return 1

    def build_models(self):
        target_model = AutoModelForCausalLM.from_pretrained(
            self.args.target_name_or_path,
            dtype=torch.bfloat16,
            attn_implementation=self.EVAL_ATTN_IMPLEMENTATION,
        ).to(device=self.device).eval()
        tokenizer = AutoTokenizer.from_pretrained(self.args.target_name_or_path)
        return target_model, None, tokenizer

    def _init_context(self, **kwargs) -> None:
        return None

    def _propose(
        self,
        *,
        context: None,
        output_ids: torch.Tensor,
        position_ids: torch.Tensor,
        start: int,
        stop_token_ids: list[int] | None = None,
    ) -> DraftProposal:
        return DraftProposal(
            draft_token_count=0,
            verify_input_ids=output_ids[:, start : start + 1],
            draft_probs=None,
        )

    def _update(self, context: None, verification: VerificationResult) -> None:
        return None

    def generate_one_sample(
        self,
        *,
        input_ids: torch.Tensor,
        stop_token_ids: list[int] | None,
    ) -> SimpleNamespace:
        return generate_decoding_sample(
            target_model=self.target_model,
            input_ids=input_ids,
            max_new_tokens=int(self.args.max_new_tokens),
            max_proposal_tokens=self.max_proposal_tokens,
            temperature=float(self.args.temperature),
            stop_token_ids=stop_token_ids,
            init_context=self._init_context,
            propose=self._propose,
            update=self._update,
        )


class SweepDSparkEvaluator(InstrumentedMixin, DSparkKEvaluator):
    pass


class SweepEagle3Evaluator(InstrumentedMixin, Eagle3KEvaluator):
    pass


class SweepARBaselineEvaluator(InstrumentedMixin, ARBaselineEvaluator):
    pass


SWEEP_EVALUATORS = {
    "ar": SweepARBaselineEvaluator,
    "eagle3": SweepEagle3Evaluator,
    "dflash": SweepDSparkEvaluator,
    "dspark": SweepDSparkEvaluator,
}
