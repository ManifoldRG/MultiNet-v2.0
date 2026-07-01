"""Direct-source tests for lib/cost_safety.sh (the shared cost-safety foundation)."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
LIB = REPO / "lib" / "cost_safety.sh"


def bash(snippet: str, env: dict | None = None) -> subprocess.CompletedProcess:
    e = os.environ.copy()
    e.pop("MAX_RUN_DURATION", None)
    if env:
        e.update(env)
    return subprocess.run(["bash", "-c", snippet], capture_output=True, text=True, cwd=REPO, env=e)


def test_lib_sourcing_is_side_effect_free():
    r = bash("source ./lib/cost_safety.sh; echo OK")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "OK"


@pytest.mark.parametrize("value,expected", [
    ("6h", "21600"), ("90m", "5400"), ("1h30m", "5400"),
    ("3600s", "3600"), ("1d", "86400"), ("1d2h3m4s", "93784"),
])
def test_parse_duration_valid(value, expected):
    r = bash(f"source ./lib/cost_safety.sh; parse_duration_seconds {value}")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == expected


@pytest.mark.parametrize("value", ["", "6", "6hh", "1x", "abc", "h", "1.5h"])
def test_parse_duration_invalid(value):
    r = bash(f"source ./lib/cost_safety.sh; parse_duration_seconds '{value}'")
    assert r.returncode != 0


@pytest.mark.parametrize("value,expected", [("6h", "420"), ("90m", "150"), ("1d", "1500")])
def test_watchdog_minutes(value, expected):
    r = bash(f"source ./lib/cost_safety.sh; watchdog_minutes {value}")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == expected


def test_require_max_run_duration_unset_fails():
    r = bash("source ./lib/cost_safety.sh; require_max_run_duration")
    assert r.returncode != 0
    assert "MAX_RUN_DURATION" in r.stderr


def test_require_max_run_duration_valid_passes():
    r = bash("source ./lib/cost_safety.sh; require_max_run_duration", env={"MAX_RUN_DURATION": "6h"})
    assert r.returncode == 0, r.stderr


def test_validate_run_id_rejects_bad_chars():
    r = bash("source ./lib/cost_safety.sh; validate_run_id", env={"RUN_ID": "bad id/slash"})
    assert r.returncode != 0


def test_arm_watchdog_fresh_vm_ssh_fail_warns_and_returns_0():
    r = bash("source ./lib/cost_safety.sh; gcloud() { return 1; }; export -f gcloud; "
             "MAX_RUN_DURATION=6h arm_watchdog myvm us-central1-a 1")
    assert r.returncode == 0, r.stderr
    assert "WARNING" in r.stderr


def test_arm_watchdog_reused_vm_ssh_fail_is_fatal():
    r = bash("source ./lib/cost_safety.sh; gcloud() { return 1; }; export -f gcloud; "
             "MAX_RUN_DURATION=6h arm_watchdog myvm us-central1-a 0")
    assert r.returncode != 0


def test_require_max_run_duration_malformed_fails():
    r = bash("source ./lib/cost_safety.sh; require_max_run_duration", env={"MAX_RUN_DURATION": "6"})
    assert r.returncode != 0
