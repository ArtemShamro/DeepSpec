import os
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs CUDA (and model weights for model tests)")


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


DRAFTS = {
    "ar": None,
    "eagle3": "deepseek-ai/eagle3_qwen3_4b_ttt7",
    "dflash": "deepseek-ai/dflash_qwen3_4b_block7",
    "dspark": "deepseek-ai/dspark_qwen3_4b_block7",
}
TARGET = "Qwen/Qwen3-4B"


class _EvaluatorCache:
    """Keeps at most one evaluator (one 8 GB target) alive; each evaluator
    owns the process group, so the previous one is cleaned up first."""

    def __init__(self):
        self.key = None
        self.evaluator = None

    def get(self, method: str, max_new_tokens: int = 128):
        from experiments.evaluators import SWEEP_EVALUATORS
        from experiments.run_sweep import build_eval_args
        from deepspec.utils.config import to_config_node

        key = (method, max_new_tokens)
        if key != self.key:
            self.release()
            cfg = to_config_node(
                {
                    "target_name_or_path": TARGET,
                    "generation": {"temperature": 1.0, "max_new_tokens": max_new_tokens, "seed": 980406},
                }
            )
            args = build_eval_args(cfg, {"name": method, "draft_name_or_path": DRAFTS[method]})
            self.evaluator = SWEEP_EVALUATORS[method](0, args)
            self.key = key
        evaluator = self.evaluator
        evaluator.step_timing = True
        evaluator.sample_sink = None
        if hasattr(evaluator, "trained_draft_length"):
            evaluator.set_k(evaluator.trained_draft_length)
        return evaluator

    def release(self):
        if self.evaluator is None:
            return
        import gc

        self.evaluator.clean_up()
        self.evaluator = None
        self.key = None
        gc.collect()
        torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def evaluators():
    cache = _EvaluatorCache()
    yield cache
    cache.release()


@pytest.fixture(autouse=True, scope="session")
def _repo_root_cwd():
    # DeepSpec loads eval datasets relative to the working directory.
    previous = os.getcwd()
    os.chdir(REPO_ROOT)
    yield
    os.chdir(previous)
