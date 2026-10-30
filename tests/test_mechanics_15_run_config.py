"""Guards for the mechanics-15 run configs (Kimi K3 + Opus 5.5 + GPT-6 Astra): 2D,
and the NL and 3D runs that differ from it only in observation or render."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gridworld.render_settings import RenderSettings
from scripts.distributed_topology import derive_topology
from scripts.run_pipeline import (
    _build_agent_from_spec,
    _experiment_config_hash_payload,
    _runtime_model_config,
)

_FIX = Path(__file__).resolve().parents[1] / "gridworld" / "fixtures"
_CFG = json.loads((_FIX / "run_config.mechanics_15_2d.json").read_text())
_R1 = json.loads((_FIX / "run_config.r1.json").read_text())


def test_cell_is_r1_with_full_context_and_prompt_cache():
    # prompt_cache is billing only (stripped from every hash), so the cell is still R1's.
    expected = {**_R1["experiment_config"], "context_window": "full", "prompt_cache": True}
    assert _CFG["experiment_config"] == expected
    assert "conditions" not in _CFG
    assert "render" not in _CFG, "2D run: MiniGrid backend, no render block"
    assert _CFG["manifest"] == "gridworld/fixtures/manifest.mechanics_15.json"


def test_models_and_equal_caps():
    models = _CFG["models"]
    assert {m["model"] for m in models.values()} == {"kimi-k3", "claude-opus-5-5", "gpt-6-astra"}
    assert {m["max_tokens"] for m in models.values()} == {64000}
    assert not _CFG.get("allow_unequal_max_tokens")
    assert all(m.get("enable_thinking", True) for m in models.values())
    assert all("temperature" not in m for m in models.values())


def test_configs_build_agents(monkeypatch):
    for var in ("MOONSHOT_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(var, "secret")
    m = _CFG["models"]
    kimi, _ = _build_agent_from_spec("kimi_k3", m["kimi_k3"])
    assert (kimi.config.model, kimi.config.reasoning_effort) == ("kimi-k3", "max")
    claude, _ = _build_agent_from_spec("claude_opus55", m["claude_opus55"])
    assert (claude.config.model, claude.config.effort) == ("claude-opus-5-5", "xhigh")
    astra, _ = _build_agent_from_spec("gpt6_astra", m["gpt6_astra"])
    assert (astra.config.service_tier, astra.config.reasoning_effort) == ("flex", "xhigh")
    assert astra.config.spend_cap_usd and astra.config.spend_cap_usd > 0


def test_topology_mixes_lockstep_and_sync_workers():
    topo = derive_topology(_CFG, "mech15")
    roles = {w["model_group"]: w["worker_role"] for w in topo["workers"]}
    assert roles == {"kimi-api": "worker", "claude-api": "lockstep-worker", "openai-api": "worker"}
    assert topo["has_gpu"] is False
    assert topo["required_credentials"] == [
        "ANTHROPIC_API_KEY", "MOONSHOT_API_KEY", "OPENAI_API_KEY"]


def test_worker_role_fields_are_hash_invariant():
    """Choosing the driver or its concurrency must not change a unit's hash, or
    moving a group between lockstep and sync would re-run already-paid units."""
    base = dict(_CFG["models"]["claude_opus55"])
    bare = {k: v for k, v in base.items() if k not in ("worker_role", "worker_concurrency")}
    assert _runtime_model_config(base) == _runtime_model_config(bare)


def test_nl_run_is_the_2d_run_as_text():
    nl = json.loads((_FIX / "run_config.mechanics_15_nl.json").read_text())
    assert nl["experiment_config"] == {**_CFG["experiment_config"], "observation": "text_only"}
    assert "render" not in nl, "text_only cannot take a 3D render block"
    assert nl["manifest"] == _CFG["manifest"]
    assert nl["models"] == _CFG["models"]


def test_3d_run_is_the_2d_run_in_mujoco_with_a_start_map():
    d3 = json.loads((_FIX / "run_config.mechanics_15_3d.json").read_text())
    assert d3["experiment_config"] == _CFG["experiment_config"]
    assert d3["render"] == {"backend": "mujoco3d", "camera": "first_person", "start_map": True}
    rs = RenderSettings.from_run_config(d3)
    assert (rs.backend, rs.camera, rs.start_map, rs.resolution) == ("mujoco3d", "first_person", True, "grid")
    assert d3["manifest"] == _CFG["manifest"]
    assert d3["models"] == _CFG["models"]


@pytest.mark.parametrize("mode", ["2d", "nl", "3d"])
def test_calibration_is_the_full_run_on_one_maze(mode):
    full = json.loads((_FIX / f"run_config.mechanics_15_{mode}.json").read_text())
    calib = json.loads((_FIX / f"run_config.mechanics_15_{mode}_calib.json").read_text())
    assert calib["experiment_config"] == full["experiment_config"]
    assert calib.get("render") == full.get("render")
    assert calib["manifest"] == full["manifest"]
    for name, model in full["models"].items():
        strip = ("tasks", "worker_concurrency")
        assert {k: v for k, v in calib["models"][name].items() if k not in strip} == {
            k: v for k, v in model.items() if k not in strip}
        assert calib["models"][name]["tasks"] == [
            "mech15_M10_18x18_dense_kr_sg_kb_portals_death_rotating_frozen_0"]


def test_prompt_cache_is_on_and_hash_invariant(monkeypatch):
    """Opus and Astra cache the stable history prefix (breakpoints on the newest and
    previous-newest history steps). The flags are set on the entries because fleet
    agents are built from the entry, and the switch is a billing knob: flipping it
    must not re-key already-paid episodes."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    models = _CFG["models"]
    for name in ("claude_opus55", "gpt6_astra"):
        entry = models[name]
        assert entry["enable_prompt_cache"] is True, name
        agent, _ = _build_agent_from_spec(name, entry)
        assert agent.config.enable_prompt_cache is True, name
        flipped = {**entry, "enable_prompt_cache": False}
        assert _runtime_model_config(entry) == _runtime_model_config(flipped)
    # Opus runs on the lockstep batch path: the 1h batch TTL (default) outlives
    # most rounds, where 5m would expire between them.
    opus, _ = _build_agent_from_spec("claude_opus55", models["claude_opus55"])
    assert opus.config.batch_cache_ttl == "1h"
    # Kimi K3's latency is erratic, so its implicit cache is written on the 1h tier.
    monkeypatch.setenv("MOONSHOT_API_KEY", "secret")
    kimi, _ = _build_agent_from_spec("kimi_k3", models["kimi_k3"])
    assert kimi.config.prompt_cache_ttl == "1h"
    no_ttl = {k: v for k, v in models["kimi_k3"].items() if k != "prompt_cache_ttl"}
    assert _runtime_model_config(models["kimi_k3"]) == _runtime_model_config(no_ttl)
    assert _experiment_config_hash_payload(_CFG["experiment_config"]) == (
        _experiment_config_hash_payload({**_CFG["experiment_config"], "prompt_cache": False})
    )


def test_fog_run_is_the_2d_astra_cell_under_fog_on_four_mazes():
    from gridworld.render_settings import FOG_RENDER_VERSION
    from interface.config import ExperimentConfig

    fog = json.loads((_FIX / "run_config.mechanics_15_2d_fog_astra.json").read_text())
    assert fog["experiment_config"] == _CFG["experiment_config"]
    assert fog["manifest"] == _CFG["manifest"]
    assert list(fog["models"]) == ["gpt6_astra"]
    astra, base = fog["models"]["gpt6_astra"], _CFG["models"]["gpt6_astra"]
    changed = ("tasks", "spend_cap_usd")
    assert {k: v for k, v in astra.items() if k not in changed} == {
        k: v for k, v in base.items() if k not in changed}
    assert astra["spend_cap_usd"] == 25
    rs = RenderSettings.from_run_config(fog)
    assert (rs.backend, rs.observability, rs.view_size) == ("minigrid", "fog_of_war", 7)
    assert rs.label == f"minigrid_fog_of_war_v7_r{FOG_RENDER_VERSION}"
    rs.check_experiment(ExperimentConfig.from_dict({**ExperimentConfig().to_dict(), **fog["experiment_config"]}))
    manifest = {t["task_id"]: t for t in json.loads((_FIX / "manifest.mechanics_15.json").read_text())["tasks"]}
    tasks = [manifest[t] for t in astra["tasks"]]
    assert len(tasks) == 4
    assert {m for t in tasks for m in t["new_mechanisms"]} == {"portals", "death", "rotating", "frozen"}
    # The clean square-2D solve (Astra 91 steps vs BFS 90) runs first, before the cap can bind.
    assert tasks[0]["task_id"] == "mech15_M7_18x18_dense_kr_sg_kb_portals_0"
