"""Pre-scored Stage 2 (PRESCORED_TASKS): static scoring of the PR #57 mechanics
mazes takes minutes to tens of minutes per maze, so a fleet run can seed the
coordinator's tasks/ with bundles scored on another machine. Stage 2 reuses a
bundle only when its inputs_hash matches the current spec and scorer config;
anything else is recomputed, never trusted."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from interface.loader import default_maze_path
from scripts import run_pipeline as pipeline
from scripts.distributed_run_pipeline import prepare_job
from scripts.prepare_combined_job import build as build_combined_job

REPO = Path(__file__).resolve().parents[1]
_MANIFEST = REPO / "gridworld" / "fixtures" / "manifest.json"
_TASK = "validation_10_v01_empty_room"
DMAX = 1000.0


def _run_config(tmp_path: Path, name: str = "rc.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps({"models": {"stub": {
        "provider": "claude", "model": "stub-model", "group": "stub",
        "tasks": [str(default_maze_path("V01_empty_room.json"))],
    }}}), encoding="utf-8")
    return path


def _prepare(run_config: Path, root: Path) -> dict:
    return prepare_job(run_config_path=run_config, manifest_path=_MANIFEST, seeds=[0], conditions=None,
                       artifacts_root=root, run_set_id="pre", difficulty_max_static_score=DMAX)


def _count_scoring(monkeypatch) -> list[str]:
    calls: list[str] = []
    real = pipeline.score_task_file

    def counting(source, **kwargs):
        calls.append(str(source))
        return real(source, **kwargs)

    monkeypatch.setattr(pipeline, "score_task_file", counting)
    return calls


def test_a_matching_seed_skips_stage2_and_plans_the_same_units(tmp_path, monkeypatch):
    rc = _run_config(tmp_path)
    scored = _prepare(rc, tmp_path / "local")
    shutil.copytree(tmp_path / "local" / "tasks", tmp_path / "coord" / "tasks")
    calls = _count_scoring(monkeypatch)
    seeded = _prepare(rc, tmp_path / "coord")
    assert calls == []
    assert [u["unit_id"] for u in seeded["units"]] == [u["unit_id"] for u in scored["units"]]


def test_a_stale_seed_is_recomputed_not_trusted(tmp_path, monkeypatch):
    rc = _run_config(tmp_path)
    _prepare(rc, tmp_path / "local")
    shutil.copytree(tmp_path / "local" / "tasks", tmp_path / "coord" / "tasks")
    scored = tmp_path / "coord" / "tasks" / _TASK / "scored_static.json"
    payload = json.loads(scored.read_text())
    payload["inputs_hash"] = "scored-by-other-code"
    payload["static_score"] = 1.0
    scored.write_text(json.dumps(payload))
    calls = _count_scoring(monkeypatch)
    _prepare(rc, tmp_path / "coord")
    assert len(calls) == 1
    assert json.loads(scored.read_text())["inputs_hash"] != "scored-by-other-code"


def _specs(tmp_path: Path) -> list[dict]:
    return [{"run_config": str(_run_config(tmp_path, f"{n}.json")), "manifest": str(_MANIFEST),
             "conditions": None, "prompt_variant": None, "run_id": n} for n in ("a", "b")]


def test_combined_job_scores_a_shared_task_once(tmp_path, monkeypatch):
    calls = _count_scoring(monkeypatch)
    build_combined_job(_specs(tmp_path), artifacts_root=tmp_path / "combined", run_set_id="m",
                       seeds=[0], difficulty_max=DMAX)
    assert len(calls) == 1  # part b reuses part a's bundle


def test_combined_job_uses_a_prescored_root(tmp_path, monkeypatch):
    _prepare(_run_config(tmp_path), tmp_path / "local")
    shutil.copytree(tmp_path / "local" / "tasks", tmp_path / "combined" / "tasks")
    calls = _count_scoring(monkeypatch)
    build_combined_job(_specs(tmp_path), artifacts_root=tmp_path / "combined", run_set_id="m",
                       seeds=[0], difficulty_max=DMAX)
    assert calls == []


def test_combined_job_replaces_a_stale_seed_in_the_root_workers_read(tmp_path):
    """The coordinator serves task bundles from the combined root's tasks/, so a
    stale seed there must be replaced by the part's recomputed bundle."""
    _prepare(_run_config(tmp_path), tmp_path / "local")
    fresh = json.loads((tmp_path / "local" / "tasks" / _TASK / "scored_static.json").read_text())
    shutil.copytree(tmp_path / "local" / "tasks", tmp_path / "combined" / "tasks")
    seeded = tmp_path / "combined" / "tasks" / _TASK / "scored_static.json"
    seeded.write_text(json.dumps({**fresh, "inputs_hash": "scored-by-other-code", "static_score": 1.0}))
    build_combined_job(_specs(tmp_path), artifacts_root=tmp_path / "combined", run_set_id="m",
                       seeds=[0], difficulty_max=DMAX)
    assert json.loads(seeded.read_text())["inputs_hash"] == fresh["inputs_hash"]


# --------------------------------------------------------------------------- #
# Launcher knob: PRESCORED_TASKS=<local artifacts root with tasks/>
# --------------------------------------------------------------------------- #
def _bash(snippet: str, env: dict) -> subprocess.CompletedProcess:
    e = os.environ.copy()
    e.pop("MAX_RUN_DURATION", None)
    e["DIFFICULTY_MAX"] = "3000"  # required, no default
    e["API_PREFLIGHT_CMD"] = "true"  # no live API probe from tests (tests/test_api_preflight.py covers it)
    e.update(env)
    return subprocess.run(["bash", "-c", snippet], capture_output=True, text=True, cwd=REPO, env=e)


def _capturing_gcloud(tmp_path: Path) -> Path:
    log = tmp_path / "g.log"
    fake = tmp_path / "gcloud"
    fake.write_text('#!/usr/bin/env bash\necho "ARGS: $*" >> "$GCLOUD_LOG"\n'
                    'if [[ "$*" == *"tar -x"* ]]; then tar -t >> "$GCLOUD_LOG"; else cat > /dev/null; fi\n')
    fake.chmod(0o755)
    return log


@pytest.mark.parametrize("start", ["start_coordinator", "start_coordinator_combined"])
def test_coordinator_start_ships_the_prescored_tasks_before_prepare(tmp_path, start):
    seed = tmp_path / "local"
    (seed / "tasks" / _TASK).mkdir(parents=True)
    (seed / "tasks" / _TASK / "scored_static.json").write_text("{}")
    log = _capturing_gcloud(tmp_path)
    r = _bash(f"source ./lib/distributed_start.sh; COORD=r-coord; ZONE=z1; RUN_ID=r1; RUN_CONFIG=rc; "
              f"MANIFEST=m; MASSIVE_BATCHES=1; {start}",
              {"PATH": f"{tmp_path}:{os.environ['PATH']}", "GCLOUD_LOG": str(log), "PRESCORED_TASKS": str(seed)})
    assert r.returncode == 0, r.stderr
    calls = [line for line in log.read_text().splitlines() if line.startswith("ARGS:")]
    assert "artifacts/r1" in calls[0] and "tar -x" in calls[0]
    assert f"tasks/{_TASK}/scored_static.json" in log.read_text()
    assert len(calls) > 1  # then the coordinator recipe itself


def test_coordinator_start_refuses_a_prescored_root_without_tasks(tmp_path):
    log = _capturing_gcloud(tmp_path)
    r = _bash("source ./lib/distributed_start.sh; COORD=r-coord; ZONE=z1; RUN_ID=r1; RUN_CONFIG=rc; "
              "MANIFEST=m; start_coordinator",
              {"PATH": f"{tmp_path}:{os.environ['PATH']}", "GCLOUD_LOG": str(log),
               "PRESCORED_TASKS": str(tmp_path / "empty")})
    assert r.returncode != 0 and "PRESCORED_TASKS" in r.stderr
    assert not log.exists()  # nothing reached the coordinator


def test_no_prescored_tasks_ships_nothing_extra(tmp_path):
    log = _capturing_gcloud(tmp_path)
    r = _bash("source ./lib/distributed_start.sh; COORD=r-coord; ZONE=z1; RUN_ID=r1; RUN_CONFIG=rc; "
              "MANIFEST=m; start_coordinator",
              {"PATH": f"{tmp_path}:{os.environ['PATH']}", "GCLOUD_LOG": str(log), "PRESCORED_TASKS": ""})
    assert r.returncode == 0, r.stderr
    assert "tar -x" not in log.read_text()


def test_launch_checks_the_prescored_root_before_creating_any_vm(tmp_path):
    gcloud = tmp_path / "gcloud"
    gcloud.write_text('#!/usr/bin/env bash\necho "GCLOUD $*"\n')
    gcloud.chmod(0o755)
    rc = tmp_path / "rc.json"
    rc.write_text(json.dumps({"models": {"q": {"provider": "qwen_vllm", "hardware_profile": "local-gpu",
                                               "group": "qwen36-27b", "worker_count": 1}}}))
    r = _bash("source ./launch_distributed.sh; require_clean_tree() { return 0; }; main",
              {"PATH": f"{tmp_path}:{os.environ['PATH']}", "MAX_RUN_DURATION": "6h", "RUN_ID": "r3",
               "RUNS_DIR": str(tmp_path / ".runs"), "RUN_CONFIG": str(rc), "MANIFEST": "m",
               "ZONES": "zoneA", "PRESCORED_TASKS": str(tmp_path / "nope")})
    assert r.returncode != 0 and "PRESCORED_TASKS" in r.stderr
    assert "instances create" not in r.stdout
