"""R1-frontier (OpenAI) run-config parity + the cost estimator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from interface.agents.openai_agent import OpenAIAgent
from scripts import estimate_frontier_cost as est
from scripts.distributed_topology import derive_topology
from scripts.run_pipeline import (
    _build_agent_from_spec,
    check_run_config_expectations,
    load_run_config,
)

_FIX = Path(__file__).resolve().parents[1] / "gridworld" / "fixtures"
_R1 = _FIX / "run_config.r1.json"
_FRONTIER = _FIX / "run_config.r1_frontier.json"
_CALIB = _FIX / "run_config.r1_frontier_calib_terra.json"
_PROBE = _FIX / "run_config.r1_frontier_probe_astra.json"


def test_probe_is_frontier_params_on_one_smoke_maze():
    front = load_run_config(_FRONTIER)["models"]["gpt6_astra"]
    probe = load_run_config(_PROBE)["models"]["gpt6_astra"]
    diff = {k for k in front.keys() | probe.keys() if front.get(k) != probe.get(k)}
    assert diff <= {"spend_cap_usd", "max_in_flight", "tasks"}, diff
    rc = load_run_config(_PROBE)
    check_run_config_expectations(rc, _FIX.parents[1] / rc["manifest"], None)


@pytest.mark.parametrize("path", [_FRONTIER, _CALIB, _PROBE])
def test_config_is_the_r1_cell_verbatim(path):
    rc, r1 = load_run_config(path), load_run_config(_R1)
    assert rc["experiment_config"] == r1["experiment_config"]
    assert "conditions" not in rc
    (model,) = rc["models"].values()
    # Same output cap as the R1 API models; thinking on at Claude-R1's effort.
    assert model["max_tokens"] == r1["models"]["claude_opus"]["max_tokens"] == 64000
    assert model["reasoning_effort"] == r1["models"]["claude_opus"]["effort"] == "xhigh"
    assert "temperature" not in model
    assert model["spend_cap_usd"] > 0


def test_frontier_runs_the_r1_panel():
    rc = load_run_config(_FRONTIER)
    r1 = load_run_config(_R1)
    assert rc["manifest"] == r1["manifest"]
    check_run_config_expectations(rc, _FIX.parents[1] / rc["manifest"], None)
    model = rc["models"]["gpt6_astra"]
    assert (model["model"], model["service_tier"]) == ("gpt-6-astra", "flex")


def test_calib_differs_from_frontier_only_in_model_tier_cap_and_manifest():
    front = load_run_config(_FRONTIER)["models"]["gpt6_astra"]
    calib = load_run_config(_CALIB)["models"]["gpt56_terra"]
    allowed = {"model", "service_tier", "spend_cap_usd", "max_in_flight", "max_attempts"}
    diff = {k for k in front.keys() | calib.keys() if front.get(k) != calib.get(k)}
    assert diff <= allowed, diff


@pytest.mark.parametrize("path", [_FRONTIER, _CALIB, _PROBE])
def test_configs_build_openai_agents(monkeypatch, path):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    for name, cfg in load_run_config(path)["models"].items():
        agent, _ = _build_agent_from_spec(name, cfg)
        assert isinstance(agent, OpenAIAgent)
        assert agent.config.spend_cap_usd == cfg["spend_cap_usd"]


def test_topology_is_one_api_worker_needing_openai_key():
    topo = derive_topology(load_run_config(_FRONTIER), "r1-frontier")
    assert [w["kind"] for w in topo["workers"]] == ["api"]
    assert topo["workers"][0]["provider"] == "openai"
    assert topo["required_credentials"] == ["OPENAI_API_KEY"]
    assert topo["has_gpu"] is False


def _write_episode(root, model, task, usages, success=False):
    d = root / "runs" / task / "minigrid" / model / "seed_0" / "default"
    d.mkdir(parents=True)
    transcript = [{"kind": "reset"}] + [
        {"kind": "query", "usage": u, "stop_reason": "stop"} for u in usages
    ]
    (d / "episode.json").write_text(
        json.dumps({"success": success, "end_reason": "stalled", "transcript": transcript})
    )


def test_estimator_projects_from_proxy_profile(tmp_path, capsys):
    u = {"input_tokens": 1000, "output_tokens": 2000, "reasoning_tokens": 1800, "cached_input_tokens": 0}
    _write_episode(tmp_path, "gpt-5.6-terra", "t1", [u] * 10)
    _write_episode(tmp_path, "gpt-5.6-terra", "t2", [u] * 30, success=True)
    assert est.main(["--artifacts-root", str(tmp_path)]) == 0
    s = est.summarize("gpt-5.6-terra", est.collect(tmp_path)["gpt-5.6-terra"])
    assert (s["episodes"], s["queries"], s["queries_per_episode_mean"]) == (2, 40, 20.0)
    assert s["per_query_reasoning_mean"] == 1800
    # Terra standard: 40k in x $2 + 80k out x $12
    assert s["actual_spend_usd_by_tier"]["default"] == pytest.approx(0.08 + 0.96)
    proj = est.project(s, 50)
    sc = proj["scenarios"]["calibration (gpt-5.6-terra)"]
    # 20 q/ep x 50 = 1000 queries; flex astra = 1M in x $5 + 2M out x $25
    assert sc["panel_queries"] == 1000
    assert sc["usd_flex"] == pytest.approx(55.0)
    assert sc["usd_standard"] == pytest.approx(110.0)
    assert proj["is_astra_measured"] is False
    assert "PROXY" in capsys.readouterr().out


def test_estimator_prefers_measured_astra(tmp_path):
    u = {"input_tokens": 1000, "output_tokens": 500}
    _write_episode(tmp_path, "gpt-5.6-terra", "t1", [u] * 5)
    _write_episode(tmp_path, "gpt-6-astra", "t1", [u] * 5)
    assert est.main(["--artifacts-root", str(tmp_path), "--json", str(tmp_path / "r.json")]) == 0
    report = json.loads((tmp_path / "r.json").read_text())
    assert report["projection"]["is_astra_measured"] is True
