from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.sweep_config import load_sweep_config, validate_sweep_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
SMOKE = str(CONFIG_DIR / "qwen3_4b_smoke.py")
MAIN = str(CONFIG_DIR / "qwen3_4b_main.py")


def fake_draft_configs(path):
    if "eagle3" in path:
        return SimpleNamespace(architectures=["Qwen3Eagle3Model"], ttt_length=7)
    return SimpleNamespace(architectures=["Qwen3DSparkModel"], block_size=7)


def _write_config(tmp_path, body):
    path = tmp_path / "cfg.py"
    path.write_text(body)
    return str(path)


MINIMAL = """
target_name_or_path = "Qwen/Qwen3-4B"
methods = [{"name": "ar", "draft_name_or_path": None},
           {"name": "dspark", "draft_name_or_path": "deepseek-ai/dspark_qwen3_4b_block7"}]
datasets = [("gsm8k", 2)]
draft_lengths = [3]
"""


def test_defaults_match_deepspec_eval(tmp_path):
    cfg = load_sweep_config(_write_config(tmp_path, MINIMAL))
    assert cfg.generation == {
        "temperature": 1.0,
        "max_new_tokens": 2048,
        "seed": 980406,
        "enable_thinking": False,
    }
    assert cfg.datasets == [("gsm8k", 2)]


def test_overrides(tmp_path):
    cfg = load_sweep_config(
        _write_config(tmp_path, MINIMAL),
        ["generation.max_new_tokens=64", "draft_lengths=[5,7]", "step_timing=false"],
    )
    assert cfg.generation.max_new_tokens == 64
    assert cfg.generation.seed == 980406
    assert cfg.draft_lengths == [5, 7]
    assert cfg.step_timing is False


def test_unknown_override_key(tmp_path):
    with pytest.raises(KeyError):
        load_sweep_config(_write_config(tmp_path, MINIMAL), ["generation.top_p=0.9"])


def test_unknown_method(tmp_path):
    cfg = load_sweep_config(
        _write_config(tmp_path, MINIMAL.replace('"name": "ar"', '"name": "medusa"'))
    )
    with pytest.raises(ValueError, match="unsupported method 'medusa'"):
        validate_sweep_config(cfg, draft_config_loader=fake_draft_configs)


def test_k_below_one(tmp_path):
    cfg = load_sweep_config(_write_config(tmp_path, MINIMAL), ["draft_lengths=[0]"])
    with pytest.raises(ValueError, match="K=0"):
        validate_sweep_config(cfg, draft_config_loader=fake_draft_configs)


def test_k_above_trained_length_allowed(tmp_path):
    cfg = load_sweep_config(_write_config(tmp_path, MINIMAL), ["draft_lengths=[3,8,16]"])
    trained = validate_sweep_config(cfg, draft_config_loader=fake_draft_configs)
    assert trained["dspark"] == 7


@pytest.mark.parametrize("path", [SMOKE, MAIN])
def test_shipped_configs_validate(path):
    cfg = load_sweep_config(path)
    trained = validate_sweep_config(cfg, draft_config_loader=fake_draft_configs)
    assert [m["name"] for m in cfg.methods] == ["ar", "eagle3", "dflash", "dspark"]
    assert trained == {"eagle3": 7, "dflash": 7, "dspark": 7}
