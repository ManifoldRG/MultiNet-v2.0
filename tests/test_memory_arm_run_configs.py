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


class _NotesAgent:
    """Scripted V01 solver that keeps a scratchpad."""

    def __init__(self):
        from interface.smoke_tests.plans import v01_empty_room_trajectory

        self._actions = iter(v01_empty_room_trajectory())
        self.last_usage = None

    def __call__(self, messages):
        self.last_usage = {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}
        action = next(self._actions, "DONE")
        return f"<scratchpad>\nnext: {action.lower()}\n</scratchpad>\nFINAL_OUTPUT: {action}"


def _stub_run_config(tmp_path: Path, arm: str) -> Path:
    """The arm's cell with one stub model on the V01 maze."""
    from interface.loader import default_maze_path

    cfg_path = tmp_path / f"run_config_{arm}.json"
    cfg_path.write_text(json.dumps({
        "experiment_config": CONFIGS[arm]["experiment_config"],
        "models": {"stub": {"provider": "openai", "model": "stub-model", "group": "stub",
                            "tasks": [str(default_maze_path("V01_empty_room.json"))]}},
    }))
    return cfg_path


def _run_arm(tmp_path: Path, arm: str, root: str = "artifacts") -> None:
    from scripts.run_pipeline import run_from_config
    from test_run_pipeline import _MANIFEST, _STABLE_DIFFICULTY_MAX

    run_from_config(
        run_config_path=_stub_run_config(tmp_path, arm), manifest_path=_MANIFEST, seeds=[0],
        artifacts_root=tmp_path / root, run_set_id="memory",
        agent_factory=lambda name, cfg: (_NotesAgent(), cfg["model"]),
        difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
    )


@pytest.mark.parametrize("arm", ARMS)
def test_pipeline_records_the_arm_end_to_end(arm, tmp_path):
    """run_from_config on one NL maze with a scripted agent: the arm reaches the
    sidecar and the hash, and episode.json carries the notes per query."""
    from scorer.io import load_json

    _run_arm(tmp_path, arm)
    [run_dir] = [p.parent for p in (tmp_path / "artifacts").rglob("episode.json")]
    sidecar = load_json(run_dir / "run_inputs.json")
    episode = load_json(run_dir / "episode.json")
    assert episode["success"] is True
    queries = [r for r in episode["transcript"] if r["kind"] == "query"]
    if arm == "full_context":
        # Persisted as before the memory arms existed.
        for persisted in (sidecar["experiment_config"], episode["config"]):
            assert {"memory", "scratchpad_max_chars"}.isdisjoint(persisted)
        assert not any(k.startswith("scratchpad") for r in queries for k in r)
    else:
        for persisted in (sidecar["experiment_config"], episode["config"]):
            assert (persisted["memory"], persisted["scratchpad_max_chars"]) == (arm, 2000)
        assert all(r["scratchpad_status"] == "updated" for r in queries)
        assert queries[0]["scratchpad"].startswith("next: ")
        assert queries[1]["scratchpad_before"] == queries[0]["scratchpad"]


# The run directory's last component: the prompt variant, plus the arm for a
# scratchpad arm. The default arm's directory is the one it always had.
RUN_DIR_NAMES = {
    "full_context": "default",
    "scratchpad_window": "default__memory_scratchpad_window",
    "full_context_scratchpad": "default__memory_full_context_scratchpad",
}


def test_the_arms_keep_their_own_run_dirs_in_one_artifacts_root(tmp_path):
    """Run one after another into one root, no arm overwrites another."""
    from scorer.io import load_json

    for arm in ARMS:
        _run_arm(tmp_path, arm)
    episodes = {
        path.parent.name: load_json(path)
        for path in (tmp_path / "artifacts").rglob("episode.json")
    }
    assert set(episodes) == set(RUN_DIR_NAMES.values())
    for arm, name in RUN_DIR_NAMES.items():
        assert episodes[name]["config"].get("memory", "full_context") == arm
        assert episodes[name]["success"] is True
    [default_dir] = [p.parent for p in (tmp_path / "artifacts").rglob("default/episode.json")]
    assert default_dir.relative_to(tmp_path / "artifacts").parts[-2:] == ("seed_0", "default")


@pytest.mark.parametrize("arm", ARMS)
def test_a_fleet_unit_plans_and_writes_the_same_run_dir(arm, tmp_path):
    """coordinator-prepare's unit run_dir is where the worker writes and where
    the archive is packed from."""
    from scripts.distributed_run_pipeline import (
        CoordinatorStore,
        package_run_archive,
        prepare_job,
        run_assigned_unit,
    )
    from test_run_pipeline import _MANIFEST, _STABLE_DIFFICULTY_MAX

    prepare_job(
        run_config_path=_stub_run_config(tmp_path, arm), manifest_path=_MANIFEST, seeds=[0],
        conditions=None, artifacts_root=tmp_path / "coordinator", run_set_id="memory",
        difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
    )
    store = CoordinatorStore(tmp_path / "coordinator")
    worker_id = store.register({"worker_id": "w", "capabilities": {"model_group": "stub"}})["worker_id"]
    unit = store.assign(worker_id)["unit"]
    assert Path(unit["run_dir"]).name == RUN_DIR_NAMES[arm]
    worker_root = tmp_path / "worker"
    run_assigned_unit(
        unit, artifacts_root=worker_root,
        agent_factory=lambda name, cfg: (_NotesAgent(), cfg["model"]),
    )
    assert (worker_root / unit["run_dir"] / "episode.json").exists()
    archive = package_run_archive(unit, artifacts_root=worker_root)
    store.upload(worker_id, unit["unit_id"], archive.read_bytes())  # raises unless the hash matches
    assert store.load_state()["units"][unit["unit_id"]]["status"] == "verified"
