"""Tests for the distributed-run cost safety net.

Covers the testable logic in ``launch_smoke_4vm.sh`` and ``monitor_run.sh``:
duration parsing, the required ``MAX_RUN_DURATION`` gate, the on-VM watchdog
horizon math, the monitor's status -> exit-code classification, and the
safety-critical "egress gates the spin-down" ordering.

The gcloud orchestration itself is not unit-tested here (it is external-command
glue); these tests exercise the pure logic by sourcing the scripts and, where a
side-effect matters, by overriding the gcloud-calling functions.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
LAUNCH = REPO / "launch_smoke_4vm.sh"
MONITOR = REPO / "monitor_run.sh"


def bash(snippet: str, env: dict | None = None) -> subprocess.CompletedProcess:
    e = os.environ.copy()
    # Keep these unset unless a test opts in, so we exercise the "required" gates.
    e.pop("MAX_RUN_DURATION", None)
    e.pop("MOONSHOT_API_KEY", None)
    if env:
        e.update(env)
    return subprocess.run(
        ["bash", "-c", snippet],
        capture_output=True,
        text=True,
        cwd=REPO,
        env=e,
    )


# --------------------------------------------------------------------------- #
# parse_duration_seconds
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "value,expected",
    [
        ("6h", "21600"),
        ("90m", "5400"),
        ("1h30m", "5400"),
        ("3600s", "3600"),
        ("1d", "86400"),
        ("45s", "45"),
        ("1d2h3m4s", "93784"),
    ],
)
def test_parse_duration_valid(value, expected):
    r = bash(f"source ./launch_smoke_4vm.sh; parse_duration_seconds {value}")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == expected


@pytest.mark.parametrize("value", ["", "6", "6hh", "1x", "abc", "h", "1.5h"])
def test_parse_duration_invalid_returns_nonzero(value):
    r = bash(f"source ./launch_smoke_4vm.sh; parse_duration_seconds '{value}'")
    assert r.returncode != 0


# --------------------------------------------------------------------------- #
# require_max_run_duration  (required, no default)
# --------------------------------------------------------------------------- #

def test_require_max_run_duration_unset_fails():
    r = bash("source ./launch_smoke_4vm.sh; require_max_run_duration")
    assert r.returncode != 0
    assert "MAX_RUN_DURATION" in r.stderr


def test_require_max_run_duration_malformed_fails():
    r = bash("source ./launch_smoke_4vm.sh; require_max_run_duration",
             env={"MAX_RUN_DURATION": "6hh"})
    assert r.returncode != 0


def test_require_max_run_duration_valid_passes():
    r = bash("source ./launch_smoke_4vm.sh; require_max_run_duration",
             env={"MAX_RUN_DURATION": "6h"})
    assert r.returncode == 0, r.stderr


# --------------------------------------------------------------------------- #
# watchdog_minutes  (Layer 1 fires 1h after the Layer 0 floor)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "value,expected",
    [
        ("6h", "420"),     # 360 + 60
        ("90m", "150"),    # 90 + 60
        ("1d", "1500"),    # 1440 + 60
    ],
)
def test_watchdog_minutes(value, expected):
    r = bash(f"source ./launch_smoke_4vm.sh; watchdog_minutes {value}")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == expected


# --------------------------------------------------------------------------- #
# launch path / subcommand gating (real script + fake gcloud)
# --------------------------------------------------------------------------- #

def _fake_gcloud(tmp_path: Path) -> Path:
    fake = tmp_path / "gcloud"
    fake.write_text('#!/usr/bin/env bash\necho "GCLOUD $*"\n')
    fake.chmod(0o755)
    return fake


def test_launch_without_max_run_duration_aborts(tmp_path):
    _fake_gcloud(tmp_path)
    env = {"PATH": f"{tmp_path}:{os.environ['PATH']}", "MOONSHOT_API_KEY": "x"}
    r = bash("bash ./launch_smoke_4vm.sh", env=env)
    assert r.returncode != 0
    assert "MAX_RUN_DURATION" in r.stderr


def test_stop_subcommand_does_not_require_max_run_duration(tmp_path):
    _fake_gcloud(tmp_path)
    env = {"PATH": f"{tmp_path}:{os.environ['PATH']}"}
    r = bash("bash ./launch_smoke_4vm.sh stop", env=env)
    assert r.returncode == 0, r.stderr
    assert "instances stop" in r.stdout
    assert "instances delete" not in r.stdout


def test_delete_subcommand_uses_delete_verb(tmp_path):
    _fake_gcloud(tmp_path)
    env = {"PATH": f"{tmp_path}:{os.environ['PATH']}"}
    r = bash("bash ./launch_smoke_4vm.sh delete", env=env)
    assert r.returncode == 0, r.stderr
    assert "instances delete" in r.stdout


# --------------------------------------------------------------------------- #
# monitor_run.sh  status -> exit code
#   0 running, 10 complete, 20 stalled
# --------------------------------------------------------------------------- #

def _status(verified, total, progress_total=0, **extra):
    units = {"verified": verified}
    units.update(extra)
    return json.dumps(
        {
            "job_id": "j",
            "unit_count": total,
            "units": units,
            "worker_count": 3,
            "progress_total": progress_total,
        }
    )


def test_monitor_running_exit_0(tmp_path):
    env = {"MONITOR_TEST_STATUS": _status(4, 6, pending=2),
           "MONITOR_NOW_EPOCH": "1000"}
    r = bash(f"./monitor_run.sh --once --state-file {tmp_path}/s.json "
             f"--coord c --zone z --stall-minutes 10", env=env)
    assert r.returncode == 0, r.stderr


def test_monitor_complete_exit_10(tmp_path):
    env = {"MONITOR_TEST_STATUS": _status(6, 6), "MONITOR_NOW_EPOCH": "1000"}
    r = bash(f"./monitor_run.sh --once --state-file {tmp_path}/s.json "
             f"--coord c --zone z --stall-minutes 10", env=env)
    assert r.returncode == 10, r.stderr


def test_monitor_stalled_exit_20(tmp_path):
    state = f"{tmp_path}/s.json"
    # First sight: records signature + now=1000, reports running.
    env1 = {"MONITOR_TEST_STATUS": _status(4, 6, pending=2),
            "MONITOR_NOW_EPOCH": "1000"}
    r1 = bash(f"./monitor_run.sh --once --state-file {state} "
              f"--coord c --zone z --stall-minutes 1", env=env1)
    assert r1.returncode == 0, r1.stderr
    # Second sight: same counts, 120s later, threshold 1min -> stalled.
    env2 = {"MONITOR_TEST_STATUS": _status(4, 6, pending=2),
            "MONITOR_NOW_EPOCH": "1120"}
    r2 = bash(f"./monitor_run.sh --once --state-file {state} "
              f"--coord c --zone z --stall-minutes 1", env=env2)
    assert r2.returncode == 20, r2.stderr


def test_monitor_complete_takes_precedence_over_stall(tmp_path):
    state = f"{tmp_path}/s.json"
    env1 = {"MONITOR_TEST_STATUS": _status(6, 6), "MONITOR_NOW_EPOCH": "1000"}
    r1 = bash(f"./monitor_run.sh --once --state-file {state} "
              f"--coord c --zone z --stall-minutes 1", env=env1)
    assert r1.returncode == 10, r1.stderr
    env2 = {"MONITOR_TEST_STATUS": _status(6, 6), "MONITOR_NOW_EPOCH": "9999"}
    r2 = bash(f"./monitor_run.sh --once --state-file {state} "
              f"--coord c --zone z --stall-minutes 1", env=env2)
    assert r2.returncode == 10, r2.stderr


# --------------------------------------------------------------------------- #
# Safety-critical: egress must succeed before any spin-down.
# --------------------------------------------------------------------------- #

def test_egress_failure_blocks_stop():
    snippet = (
        "source ./monitor_run.sh; "
        "run_finalize(){ return 0; }; "
        "egress_artifacts(){ return 1; }; "
        "stop_all_vms(){ echo STOPPED; }; "
        "complete_actions"
    )
    r = bash(snippet)
    assert "STOPPED" not in r.stdout
    assert r.returncode != 0


def test_egress_success_allows_stop():
    snippet = (
        "source ./monitor_run.sh; "
        "run_finalize(){ return 0; }; "
        "egress_artifacts(){ return 0; }; "
        "stop_all_vms(){ echo STOPPED; }; "
        "complete_actions"
    )
    r = bash(snippet)
    assert "STOPPED" in r.stdout
    assert r.returncode == 0, r.stderr


def test_finalize_failure_blocks_egress_and_stop():
    snippet = (
        "source ./monitor_run.sh; "
        "run_finalize(){ return 1; }; "
        "egress_artifacts(){ echo EGRESSED; return 0; }; "
        "stop_all_vms(){ echo STOPPED; }; "
        "complete_actions"
    )
    r = bash(snippet)
    assert "EGRESSED" not in r.stdout
    assert "STOPPED" not in r.stdout
    assert r.returncode != 0


def test_monitor_progress_advance_resets_stall(tmp_path):
    state = f"{tmp_path}/s.json"
    env1 = {"MONITOR_TEST_STATUS": _status(0, 6, progress_total=10, running=2, pending=4),
            "MONITOR_NOW_EPOCH": "1000"}
    r1 = bash(f"./monitor_run.sh --once --state-file {state} "
              f"--coord c --zone z --stall-minutes 1", env=env1)
    assert r1.returncode == 0, r1.stderr
    # 120s later: identical unit counts but progress advanced -> NOT stalled
    env2 = {"MONITOR_TEST_STATUS": _status(0, 6, progress_total=25, running=2, pending=4),
            "MONITOR_NOW_EPOCH": "1120"}
    r2 = bash(f"./monitor_run.sh --once --state-file {state} "
              f"--coord c --zone z --stall-minutes 1", env=env2)
    assert r2.returncode == 0, r2.stderr


def test_monitor_no_progress_stalls(tmp_path):
    state = f"{tmp_path}/s.json"
    env1 = {"MONITOR_TEST_STATUS": _status(0, 6, progress_total=10, running=2, pending=4),
            "MONITOR_NOW_EPOCH": "1000"}
    r1 = bash(f"./monitor_run.sh --once --state-file {state} "
              f"--coord c --zone z --stall-minutes 1", env=env1)
    assert r1.returncode == 0, r1.stderr
    env2 = {"MONITOR_TEST_STATUS": _status(0, 6, progress_total=10, running=2, pending=4),
            "MONITOR_NOW_EPOCH": "1120"}
    r2 = bash(f"./monitor_run.sh --once --state-file {state} "
              f"--coord c --zone z --stall-minutes 1", env=env2)
    assert r2.returncode == 20, r2.stderr


def test_monitor_absent_progress_total_still_stalls(tmp_path):
    # An old coordinator emits no progress_total -> behaves exactly as before.
    state = f"{tmp_path}/s.json"
    s = json.dumps({"job_id": "j", "unit_count": 6,
                    "units": {"verified": 0, "running": 2, "pending": 4}, "worker_count": 3})
    r1 = bash(f"./monitor_run.sh --once --state-file {state} --coord c --zone z --stall-minutes 1",
              env={"MONITOR_TEST_STATUS": s, "MONITOR_NOW_EPOCH": "1000"})
    assert r1.returncode == 0, r1.stderr
    r2 = bash(f"./monitor_run.sh --once --state-file {state} --coord c --zone z --stall-minutes 1",
              env={"MONITOR_TEST_STATUS": s, "MONITOR_NOW_EPOCH": "1120"})
    assert r2.returncode == 20, r2.stderr


# --------------------------------------------------------------------------- #
# launch_qwen_smoke.sh  (3-VM Qwen-only launcher)
# --------------------------------------------------------------------------- #

def test_qwen_launch_syntax_ok():
    r = bash("bash -n ./launch_qwen_smoke.sh")
    assert r.returncode == 0, r.stderr


def test_qwen_launch_without_max_run_duration_aborts(tmp_path):
    _fake_gcloud(tmp_path)
    env = {"PATH": f"{tmp_path}:{os.environ['PATH']}"}
    r = bash("bash ./launch_qwen_smoke.sh", env=env)
    assert r.returncode != 0
    assert "MAX_RUN_DURATION" in r.stderr


def test_qwen_stop_subcommand_three_vms_no_kimi(tmp_path):
    _fake_gcloud(tmp_path)
    env = {"PATH": f"{tmp_path}:{os.environ['PATH']}"}
    r = bash("bash ./launch_qwen_smoke.sh stop", env=env)
    assert r.returncode == 0, r.stderr
    assert "instances stop" in r.stdout
    assert "mn-qwen-coord" in r.stdout
    assert "mn-qwen-1" in r.stdout and "mn-qwen-2" in r.stdout
    assert "kimi" not in r.stdout.lower()


def test_qwen_watchdog_minutes_helper():
    r = bash("source ./launch_qwen_smoke.sh; watchdog_minutes 6h")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "420"
