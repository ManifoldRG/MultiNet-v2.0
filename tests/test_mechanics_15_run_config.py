"""Guards for the mechanics-15 2D run config (Kimi K3 + Opus 5.5 + GPT-6 Astra)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.distributed_topology import derive_topology
from scripts.run_pipeline import _build_agent_from_spec, _runtime_model_config

_FIX = Path(__file__).resolve().parents[1] / "gridworld" / "fixtures"
_CFG = json.loads((_FIX / "run_config.mechanics_15_2d.json").read_text())
_R1 = json.loads((_FIX / "run_config.r1.json").read_text())


def test_cell_is_r1_with_full_context():
    expected = {**_R1["experiment_config"], "context_window": "full"}
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


def test_calibration_is_the_full_run_on_one_maze():
    calib = json.loads((_FIX / "run_config.mechanics_15_2d_calib.json").read_text())
    assert calib["experiment_config"] == _CFG["experiment_config"]
    assert calib["manifest"] == _CFG["manifest"]
    for name, model in _CFG["models"].items():
        strip = ("tasks", "worker_concurrency")
        assert {k: v for k, v in calib["models"][name].items() if k not in strip} == {
            k: v for k, v in model.items() if k not in strip}
        assert calib["models"][name]["tasks"] == [
            "mech15_M10_18x18_dense_kr_sg_kb_portals_death_rotating_frozen_0"]


def test_opus_prompt_cache_is_off_and_hash_invariant(monkeypatch):
    """Opus caching stays off for full context (writes every query, never reads) and the
    switch is a billing knob: flipping it must not re-key already-paid episodes."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    opus = _CFG["models"]["claude_opus55"]
    agent, _ = _build_agent_from_spec("claude_opus55", opus)
    assert agent.config.enable_prompt_cache is False
    flipped = {**opus, "enable_prompt_cache": True}
    assert _runtime_model_config(opus) == _runtime_model_config(flipped)
