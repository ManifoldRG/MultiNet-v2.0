"""next-run-fixes item 2: a launch preflight refuses a run whose API key or
account cannot work, before any VM starts.

Every API model in the run-config gets one tiny live request through its real
agent (an invalidated key like Astra's 2026-10-13 401, a missing model, an
exhausted account). Moonshot also has a balance API, checked against the
model's spend cap; Anthropic and OpenAI expose no balance to a standard key, so
the probe is the check. SKIP_API_PREFLIGHT=1 is dev-only and says so loudly.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from interface.agents.billing import BillingExhausted
from scripts import preflight_api

REPO = Path(__file__).resolve().parent.parent

MODELS = {
    "claude_opus55": {"provider": "claude", "model": "claude-opus-5-5"},
    "gpt6_astra": {"provider": "openai", "model": "gpt-6-astra"},
    "kimi_k26": {"provider": "kimi", "model": "kimi-k2.6", "spend_cap_usd": 20},
    "qwen": {"provider": "qwen_vllm_api", "model": "Qwen/Qwen3.6-27B"},
}


def _check(probe_errors=None, balance=100.0):
    probe_errors = probe_errors or {}
    probed = []

    def probe(provider, cfg):
        probed.append(provider)
        if provider in probe_errors:
            raise probe_errors[provider]

    rows = preflight_api.preflight({"models": MODELS}, probe=probe, balance=lambda: balance)
    return {r["model"]: r for r in rows}, probed


def test_all_working_accounts_pass_and_local_models_are_skipped():
    rows, probed = _check()
    assert {m: r["status"] for m, r in rows.items()} == {
        "claude_opus55": "ok", "gpt6_astra": "ok", "kimi_k26": "ok", "qwen": "skipped"}
    assert sorted(probed) == ["claude", "kimi", "openai"]
    assert "100.00" in rows["kimi_k26"]["detail"]


def test_an_invalidated_key_fails():
    rows, _ = _check({"openai": RuntimeError('OpenAI API HTTP 401: {"error": {"message": "Your API key has been invalidated."}}')})
    assert rows["gpt6_astra"]["status"] == "failed" and "401" in rows["gpt6_astra"]["detail"]
    assert rows["claude_opus55"]["status"] == "ok"


def test_an_exhausted_account_fails_as_billing():
    rows, _ = _check({"claude": BillingExhausted("claude", "credit balance too low")})
    assert rows["claude_opus55"]["status"] == "failed" and "billing" in rows["claude_opus55"]["detail"]


def test_a_moonshot_balance_below_the_spend_cap_fails():
    rows, _ = _check(balance=7.09)
    assert rows["kimi_k26"]["status"] == "failed"
    assert "7.09" in rows["kimi_k26"]["detail"] and "20" in rows["kimi_k26"]["detail"]


_MECHANICS_15_2D = REPO / "gridworld" / "fixtures" / "run_config.mechanics_15_2d.json"


@pytest.mark.parametrize("name", sorted(json.loads(_MECHANICS_15_2D.read_text())["models"]))
def test_the_probe_builds_each_api_model_through_its_real_agent(name, monkeypatch):
    """The probe's small config must pass the real agent's construction checks
    (kimi-k3 refuses enable_thinking false); the agent is built, never called."""
    import scripts.run_pipeline as run_pipeline

    for key in ("ANTHROPIC_API_KEY", "MOONSHOT_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(key, "test-key-never-sent")
    built, real_build = [], run_pipeline._build_agent_from_spec

    def build(provider, small):
        real_build(provider, small)  # raises on a config the agent refuses
        built.append(small)
        return (lambda messages: "OK"), "stub"

    monkeypatch.setattr(run_pipeline, "_build_agent_from_spec", build)
    cfg = json.loads(_MECHANICS_15_2D.read_text())["models"][name]
    preflight_api._probe(str(cfg["provider"]).lower(), cfg)
    assert len(built) == 1 and built[0]["max_tokens"] <= 256 and "spend_cap_usd" not in built[0]


def _cli(tmp_path: Path, env: dict) -> subprocess.CompletedProcess:
    rc = tmp_path / "rc.json"
    rc.write_text(json.dumps({"models": {"qwen": MODELS["qwen"]}}))
    e = {**os.environ, **env}
    return subprocess.run([sys.executable, "-m", "scripts.preflight_api", "--run-config", str(rc)],
                          capture_output=True, text=True, cwd=REPO, env=e)


def test_cli_passes_a_local_only_run_config_without_network(tmp_path):
    r = _cli(tmp_path, {"SKIP_API_PREFLIGHT": ""})
    assert r.returncode == 0, r.stderr
    assert "skipped" in r.stdout


def test_cli_skip_is_loud(tmp_path):
    r = _cli(tmp_path, {"SKIP_API_PREFLIGHT": "1"})
    assert r.returncode == 0
    assert "API PREFLIGHT SKIPPED" in r.stderr and "dev only" in r.stderr


# --------------------------------------------------------------------------- #
# Launch and sweep run it before any VM
# --------------------------------------------------------------------------- #
def _bash(snippet: str, env: dict) -> subprocess.CompletedProcess:
    e = os.environ.copy()
    for key in ("MAX_RUN_DURATION", "SKIP_API_PREFLIGHT"):
        e.pop(key, None)
    e["DIFFICULTY_MAX"] = "3000"
    e.update(env)
    return subprocess.run(["bash", "-c", snippet], capture_output=True, text=True, cwd=REPO, env=e)


def test_launch_refuses_a_failing_preflight_before_any_vm(tmp_path):
    gcloud = tmp_path / "gcloud"
    gcloud.write_text('#!/usr/bin/env bash\necho "GCLOUD $*"\n')
    gcloud.chmod(0o755)
    calls = tmp_path / "preflight.log"
    fake = tmp_path / "fake_preflight.sh"
    fake.write_text(f'#!/usr/bin/env bash\necho "$*" >> "{calls}"\necho "gpt6_astra failed: 401" >&2\nexit 1\n')
    fake.chmod(0o755)
    rc = tmp_path / "rc.json"
    rc.write_text(json.dumps({"models": {"q": {"provider": "qwen_vllm", "hardware_profile": "local-gpu",
                                               "group": "qwen36-27b", "worker_count": 1}}}))
    r = _bash("bash ./launch_distributed.sh",
              {"PATH": f"{tmp_path}:{os.environ['PATH']}", "MAX_RUN_DURATION": "6h", "RUN_ID": "rp",
               "RUNS_DIR": str(tmp_path / ".runs"), "RUN_CONFIG": str(rc), "MANIFEST": "m", "ZONES": "zoneA",
               "API_PREFLIGHT_CMD": str(fake)})
    assert r.returncode != 0
    assert f"--run-config {rc}" in calls.read_text()
    assert "instances create" not in r.stdout


def test_skip_api_preflight_is_loud_in_the_launcher():
    r = _bash("source lib/cost_safety.sh; require_api_preflight rc.json",
              {"SKIP_API_PREFLIGHT": "1", "API_PREFLIGHT_CMD": "false"})
    assert r.returncode == 0
    assert "API PREFLIGHT SKIPPED" in r.stderr


@pytest.mark.parametrize("sub", ["next-batch 2", "run-massive 2"])
def test_sweep_batch_subcommands_preflight_their_run_config(tmp_path, sub):
    from test_sweep_run import FULL_FLEET, _write_manifest

    runs = tmp_path / ".runs"
    _write_manifest(runs, "swp", FULL_FLEET)
    gcloud = tmp_path / "gcloud"
    gcloud.write_text('#!/usr/bin/env bash\necho "GCLOUD $*" >> "$GCLOUD_LOG"\n')
    gcloud.chmod(0o755)
    fake = tmp_path / "fake_preflight.sh"
    fake.write_text('#!/usr/bin/env bash\necho "PREFLIGHT $*" >&2\nexit 1\n')
    fake.chmod(0o755)
    r = _bash(f'PATH="{tmp_path}:$PATH" GCLOUD_LOG="{tmp_path / "g.log"}" ZONE=z1 SWEEP_ID=swp RUNS_DIR="{runs}" '
              f"./sweep_run.sh {sub}",
              {"MAX_RUN_DURATION": "120h", "BATCH_CAP": "9h", "API_PREFLIGHT_CMD": str(fake)})
    assert r.returncode != 0
    assert "PREFLIGHT --run-config" in r.stderr
