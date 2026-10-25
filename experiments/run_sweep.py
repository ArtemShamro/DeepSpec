"""Sweep entry point (design D6/D7).

    python -m experiments.run_sweep --config experiments/configs/qwen3_4b_smoke.py \
        [--opts key=value ...]

The parent validates the config, writes the manifest, then runs each method in
its own spawned process group (one worker per visible GPU, like eval.py).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch

from deepspec.eval.base_evaluator import load_and_process_dataset
from deepspec.utils.config import config_to_plain_dict, to_config_node
from experiments.evaluators import SWEEP_EVALUATORS
from experiments.manifest import write_manifest, write_worker_info
from experiments.records import records_from_response
from experiments.sweep_config import load_sweep_config, validate_sweep_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_eval_args(cfg, method: dict) -> SimpleNamespace:
    """The argument namespace DeepSpec evaluators read (cf. eval.py)."""
    return SimpleNamespace(
        method=method["name"],
        target_name_or_path=cfg.target_name_or_path,
        draft_name_or_path=method.get("draft_name_or_path"),
        max_new_tokens=int(cfg.generation.max_new_tokens),
        temperature=float(cfg.generation.temperature),
        seed=int(cfg.generation.seed),
        confidence_threshold=0.0,
        tensorboard_dir=None,
        step=None,
        tasks=[],
    )


class SampleWriter:
    """`sample_sink` that appends one sample's records and flushes.

    `run_dataset` visits sample indices `range(rank, n, world_size)` in order,
    so the i-th call corresponds to the i-th index of that range.
    """

    def __init__(self, *, samples_file, steps_file, key: dict, indices, max_new_tokens: int):
        self.samples_file = samples_file
        self.steps_file = steps_file
        self.key = key
        self.indices = iter(indices)
        self.max_new_tokens = max_new_tokens

    def __call__(self, response) -> None:
        sample, steps = records_from_response(
            response,
            key={**self.key, "sample_idx": next(self.indices)},
            max_new_tokens=self.max_new_tokens,
        )
        self.samples_file.write(json.dumps(sample) + "\n")
        self.steps_file.writelines(json.dumps(step) + "\n" for step in steps)
        self.samples_file.flush()
        self.steps_file.flush()


def worker(local_rank: int, cfg: dict, method: dict, run_dir: str) -> None:
    cfg = to_config_node(cfg)
    run_dir = Path(run_dir)
    name = method["name"]
    evaluator = SWEEP_EVALUATORS[name](local_rank, build_eval_args(cfg, method))
    evaluator.step_timing = bool(cfg.step_timing)
    write_worker_info(run_dir, method=method, evaluator=evaluator)
    rank, world_size = evaluator.global_rank, evaluator.world_size
    max_new_tokens = int(cfg.generation.max_new_tokens)

    draft_lengths = [0] if name == "ar" else cfg.draft_lengths
    with open(run_dir / f"samples.{name}.rank{rank}.jsonl", "a", encoding="utf-8") as samples_file, open(
        run_dir / f"steps.{name}.rank{rank}.jsonl", "a", encoding="utf-8"
    ) as steps_file:
        for k in draft_lengths:
            if name != "ar":
                evaluator.set_k(k)
            for dataset_name, max_samples in cfg.datasets:
                evaluator.sample_sink = None
                if cfg.warmup_samples > 0:
                    evaluator.run_dataset(dataset_name=dataset_name, max_samples=int(cfg.warmup_samples))

                num_samples = min(int(max_samples), len(load_and_process_dataset(dataset_name)))
                writer = SampleWriter(
                    samples_file=samples_file,
                    steps_file=steps_file,
                    key={"method": name, "dataset": dataset_name, "k": k, "rank": rank},
                    indices=range(rank, num_samples, world_size),
                    max_new_tokens=max_new_tokens,
                )
                evaluator.sample_sink = writer
                responses = evaluator.run_dataset(dataset_name=dataset_name, max_samples=int(max_samples))
                evaluator.sample_sink = None
                assert next(writer.indices, None) is None, "sample index reconstruction out of sync"

                metric_summary = evaluator.allreduce_response_metrics(responses)
                if rank == 0:
                    metrics_row = evaluator.build_metrics_row(
                        dataset_name=dataset_name,
                        metric_summary=metric_summary,
                    )
                    summary_row = {
                        "method": name,
                        "dataset": dataset_name,
                        "k": k,
                        "num_positions": evaluator.max_proposal_tokens,
                        "metrics": metrics_row,
                    }
                    with open(run_dir / "summary.jsonl", "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(summary_row) + "\n")
                    print(json.dumps(summary_row), flush=True)
    evaluator.clean_up()


def _free_port() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--opts", action="append", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = os.path.abspath(args.config)
    # DeepSpec loads eval datasets relative to the working directory.
    os.chdir(REPO_ROOT)
    cfg = load_sweep_config(config_path, args.opts)
    trained_draft_lengths = validate_sweep_config(cfg)

    start_time = datetime.now(timezone.utc)
    run_dir = Path(cfg.output_root) / f"{start_time:%Y%m%dT%H%M%SZ}_{Path(config_path).stem}"
    run_dir.mkdir(parents=True, exist_ok=False)
    write_manifest(
        run_dir,
        cfg=cfg,
        config_path=config_path,
        opts=args.opts,
        trained_draft_lengths=trained_draft_lengths,
        repo_dir=REPO_ROOT,
        start_time=start_time,
    )
    print(f"run directory: {run_dir}", flush=True)

    plain_cfg = config_to_plain_dict(cfg)
    user_port = "MASTER_PORT" in os.environ
    for method in plain_cfg["methods"]:
        if not user_port:
            os.environ["MASTER_PORT"] = _free_port()
        print(f"=== method {method['name']} ===", flush=True)
        torch.multiprocessing.spawn(
            worker,
            args=(plain_cfg, method, str(run_dir)),
            nprocs=torch.cuda.device_count(),
        )
    print(f"done: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
