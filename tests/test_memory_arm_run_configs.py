"""Guards for the memory-arm NL run configs: GPT-6 Astra on 5 mechanics-15
mazes, the R2 NL cell, and nothing but the memory arm differing between them."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from interface.config import ExperimentConfig
from scripts.run_pipeline import (
    _build_agent_from_spec,
    _expected_run_hash,
    _runtime_model_config,
    check_run_config_expectations,
    load_manifest,
)
from start_map_test_utils import r1_spec

_FIX = Path(__file__).resolve().parents[1] / "gridworld" / "fixtures"
_R2_NL = json.loads((_FIX / "run_config.mechanics_15_nl.json").read_text())
ARMS = ("full_context", "scratchpad_window", "full_context_scratchpad")
CONFIGS = {
    arm: json.loads((_FIX / f"run_config.mechanics_15_nl_memory_{arm}.json").read_text())
    for arm in ARMS
}
ARM_FIELDS = {
    "full_context": {"memory": "full_context"},
    "scratchpad_window": {
        "memory": "scratchpad_window",
        "context_window": "last_n",
        "context_n": 5,
        "scratchpad_max_chars": 2000,
    },
    "full_context_scratchpad": {"memory": "full_context_scratchpad", "scratchpad_max_chars": 2000},
}


@pytest.mark.parametrize("arm", ARMS)
def test_cell_is_r2_nl_plus_the_arm(arm):
    cfg = CONFIGS[arm]
    assert cfg["experiment_config"] == {**_R2_NL["experiment_config"], **ARM_FIELDS[arm]}
    assert cfg["manifest"] == _R2_NL["manifest"]
    assert "conditions" not in cfg and "render" not in cfg
    config = ExperimentConfig.from_dict(cfg["experiment_config"])
    assert config.memory == arm and config.observation == "text_only"


@pytest.mark.parametrize("arm", ARMS)
def test_astra_only_as_in_r2_on_the_same_five_mazes(arm, monkeypatch):
    cfg = CONFIGS[arm]
    assert list(cfg["models"]) == ["gpt6_astra"]
    astra = cfg["models"]["gpt6_astra"]
    r2 = _R2_NL["models"]["gpt6_astra"]
    differs = {"spend_cap_usd", "worker_concurrency", "max_in_flight", "tasks"}
    assert {k: v for k, v in astra.items() if k not in differs} == {
        k: v for k, v in r2.items() if k not in differs
    }
    assert astra["tasks"] == CONFIGS["full_context"]["models"]["gpt6_astra"]["tasks"]
    catalog = {row["task_id"] for row in load_manifest(_FIX / "manifest.mechanics_15.json")}
    assert len(set(astra["tasks"])) == 5 and set(astra["tasks"]) <= catalog
    check_run_config_expectations(cfg, _FIX / "manifest.mechanics_15.json", None)
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    agent, _ = _build_agent_from_spec("gpt6_astra", astra)
    assert agent.config.spend_cap_usd == astra["spend_cap_usd"] > 0
    assert (agent.config.service_tier, agent.config.reasoning_effort) == ("flex", "xhigh")


def test_arms_hash_apart():
    hashes = {
        _expected_run_hash(
            r1_spec(12), "gpt6_astra", 0, "minigrid",
            experiment_config=ExperimentConfig.from_dict(cfg["experiment_config"]),
            model_config=_runtime_model_config(CONFIGS["full_context"]["models"]["gpt6_astra"]),
        )
        for cfg in CONFIGS.values()
    }
    assert len(hashes) == 3


@pytest.mark.parametrize("arm", ARMS)
def test_pipeline_records_the_arm_end_to_end(arm, tmp_path):
    """run_from_config on one NL maze with a scripted agent: the arm reaches the
    sidecar and the hash, and episode.json carries the notes per query."""
    from interface.loader import default_maze_path
    from interface.smoke_tests.plans import v01_empty_room_trajectory
    from scorer.io import load_json
    from scripts.run_pipeline import run_from_config
    from test_run_pipeline import _MANIFEST, _STABLE_DIFFICULTY_MAX

    actions = iter(v01_empty_room_trajectory())

    class _NotesAgent:
        last_usage = None

        def __call__(self, messages):
            self.last_usage = {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}
            action = next(actions, "DONE")
            return f"<scratchpad>\nnext: {action.lower()}\n</scratchpad>\nFINAL_OUTPUT: {action}"

    cfg_path = tmp_path / "run_config.json"
    cfg_path.write_text(json.dumps({
        "experiment_config": CONFIGS[arm]["experiment_config"],
        "models": {"stub": {"provider": "openai", "model": "stub-model",
                            "tasks": [str(default_maze_path("V01_empty_room.json"))]}},
    }))
    run_from_config(
        run_config_path=cfg_path, manifest_path=_MANIFEST, seeds=[0],
        artifacts_root=tmp_path / "artifacts", run_set_id="memory",
        agent_factory=lambda name, cfg: (_NotesAgent(), cfg["model"]),
        difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
    )
    [run_dir] = [p.parent for p in (tmp_path / "artifacts").rglob("episode.json")]
    sidecar = load_json(run_dir / "run_inputs.json")
    assert sidecar["experiment_config"]["memory"] == arm
    episode = load_json(run_dir / "episode.json")
    assert episode["success"] is True
    queries = [r for r in episode["transcript"] if r["kind"] == "query"]
    if arm == "full_context":
        assert not any(k.startswith("scratchpad") for r in queries for k in r)
    else:
        assert all(r["scratchpad_status"] == "updated" for r in queries)
        assert queries[0]["scratchpad"].startswith("next: ")
        assert queries[1]["scratchpad_before"] == queries[0]["scratchpad"]
