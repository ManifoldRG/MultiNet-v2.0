"""DIFFICULTY_MAX normalises every run's difficulty
score, so a launcher default silently scored runs against another scale (the
default was 1000; R1 used 3000). It is required, like MAX_RUN_DURATION: no
default, fail closed before any VM or coordinator call."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def bash(snippet: str, env: dict | None = None) -> subprocess.CompletedProcess:
    e = os.environ.copy()
    for key in ("MAX_RUN_DURATION", "DIFFICULTY_MAX", "BATCH_CAP"):
        e.pop(key, None)
    e["API_PREFLIGHT_CMD"] = "true"  # no live API probe from tests (tests/test_api_preflight.py covers it)
    if env:
        e.update(env)
    return subprocess.run(["bash", "-c", snippet], capture_output=True, text=True, cwd=REPO, env=e)


@pytest.mark.parametrize("value", ["", "abc", "0", "-5", "1e3"])
def test_require_difficulty_max_refuses_missing_or_malformed(value):
    r = bash("source lib/cost_safety.sh; require_difficulty_max", env={"DIFFICULTY_MAX": value})
    assert r.returncode != 0
    assert "DIFFICULTY_MAX" in r.stderr


@pytest.mark.parametrize("value", ["8000", "3000.5"])
def test_require_difficulty_max_accepts_a_positive_number(value):
    r = bash("source lib/cost_safety.sh; require_difficulty_max", env={"DIFFICULTY_MAX": value})
    assert r.returncode == 0, r.stderr


def _logging_gcloud(tmp_path: Path) -> Path:
    log = tmp_path / "g.log"
    fake = tmp_path / "gcloud"
    fake.write_text('#!/usr/bin/env bash\necho "GCLOUD $*" | tee -a "$GCLOUD_LOG"\ncat > /dev/null\n')
    fake.chmod(0o755)
    return log


def test_launch_without_difficulty_max_aborts_before_any_vm(tmp_path):
    log = _logging_gcloud(tmp_path)
    rc = tmp_path / "rc.json"
    rc.write_text(json.dumps({"models": {"q": {"provider": "qwen_vllm", "hardware_profile": "local-gpu",
                                               "group": "qwen36-27b", "worker_count": 1}}}))
    r = bash("bash ./launch_distributed.sh",
             env={"PATH": f"{tmp_path}:{os.environ['PATH']}", "GCLOUD_LOG": str(log), "MAX_RUN_DURATION": "6h",
                  "RUN_ID": "r7", "RUNS_DIR": str(tmp_path / ".runs"), "RUN_CONFIG": str(rc), "MANIFEST": "m",
                  "ZONES": "zoneA"})
    assert r.returncode != 0
    assert "DIFFICULTY_MAX is required" in r.stderr
    assert "instances create" not in (log.read_text() if log.exists() else "")


@pytest.mark.parametrize("start", ["start_coordinator", "start_coordinator_combined"])
def test_coordinator_start_without_difficulty_max_calls_nothing(tmp_path, start):
    log = _logging_gcloud(tmp_path)
    r = bash(f"source ./lib/distributed_start.sh; COORD=c; ZONE=z; RUN_ID=r; RUN_CONFIG=rc; MANIFEST=m; "
             f"MASSIVE_BATCHES=1; {start}",
             env={"PATH": f"{tmp_path}:{os.environ['PATH']}", "GCLOUD_LOG": str(log)})
    assert r.returncode != 0
    assert "DIFFICULTY_MAX is required" in r.stderr
    assert not log.exists()


@pytest.mark.parametrize("start", ["start_coordinator", "start_coordinator_combined"])
def test_coordinator_start_passes_the_given_difficulty_max(tmp_path, start):
    log = _logging_gcloud(tmp_path)
    r = bash(f"source ./lib/distributed_start.sh; COORD=c; ZONE=z; RUN_ID=r; RUN_CONFIG=rc; MANIFEST=m; "
             f"MASSIVE_BATCHES=1; {start}",
             env={"PATH": f"{tmp_path}:{os.environ['PATH']}", "GCLOUD_LOG": str(log), "DIFFICULTY_MAX": "8000"})
    assert r.returncode == 0, r.stderr
    assert "DIFFICULTY_MAX='8000'" in log.read_text()


@pytest.mark.parametrize("sub", ["provision", "next-batch 2", "run-massive 2"])
def test_sweep_batch_subcommands_require_difficulty_max(tmp_path, sub):
    from test_sweep_run import FULL_FLEET, _write_manifest

    _write_manifest(tmp_path / ".runs", "swp", FULL_FLEET)
    log = _logging_gcloud(tmp_path)
    r = bash(f'PATH="{tmp_path}:$PATH" GCLOUD_LOG="{log}" ZONE=z1 SWEEP_ID=swp RUNS_DIR="{tmp_path / ".runs"}" '
             f"./sweep_run.sh {sub}", env={"MAX_RUN_DURATION": "120h", "BATCH_CAP": "9h"})
    assert r.returncode != 0
    assert "DIFFICULTY_MAX is required" in r.stderr
    assert "instances create" not in (log.read_text() if log.exists() else "")
