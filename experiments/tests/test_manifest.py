import json
import subprocess
from datetime import datetime, timezone

from deepspec.utils.config import to_config_node
from experiments.manifest import git_state, write_manifest


def _git(repo, *cmd):
    subprocess.run(["git", *cmd], cwd=repo, check=True, capture_output=True)


def _make_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.txt").write_text("a\n")
    _git(repo, "add", "a.txt")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    return repo


def test_clean_then_dirty(tmp_path):
    repo = _make_repo(tmp_path)
    clean = git_state(repo)
    assert clean["dirty"] is False and len(clean["commit"]) == 40

    (repo / "a.txt").write_text("changed\n")
    dirty = git_state(repo)
    assert dirty == {"commit": clean["commit"], "dirty": True}


def test_manifest_records_dirty_tree(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "untracked.py").write_text("x = 1\n")
    run_dir = tmp_path / "run"
    write_manifest(
        run_dir,
        cfg=to_config_node({"generation": {"seed": 1}}),
        config_path="cfg.py",
        opts=["generation.seed=1"],
        trained_draft_lengths={"dspark": 7},
        repo_dir=repo,
        start_time=datetime.now(timezone.utc),
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["git"]["dirty"] is True
    assert manifest["opts"] == ["generation.seed=1"]
    assert manifest["config"] == {"generation": {"seed": 1}}
    assert set(manifest["versions"]) >= {"python", "torch", "transformers", "cuda"}
