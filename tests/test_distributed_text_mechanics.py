"""NL smoke: a text-only episode on a PR #57/#61 mechanics maze runs on the fleet.

The text domain needs no fleet code of its own: the observation mode rides in
the unit's resolved experiment_config. This drives one unit over the M10
fixture (every new tile) through prepare -> serial worker -> upload verify,
and checks the model was told about the tiles in words.
"""

from __future__ import annotations

import json
from pathlib import Path

from gridworld.baselines import plan_bfs_path
from gridworld.task_validator import FragilityReport, TaskValidator
from interface.parser import ACTION_ORDER
from scorer.io import load_json, task_spec_from_payload
from scripts.distributed_run_pipeline import (
    CoordinatorStore,
    package_run_archive,
    prepare_job,
    run_assigned_unit,
)

_REPO = Path(__file__).resolve().parents[1]
_MANIFEST = _REPO / "gridworld" / "fixtures" / "manifest.json"
M10 = _REPO / "tests" / "fixtures" / "mechanics" / (
    "M10_18x18_dense_kr_sg_kb_portals_death_rotating_frozen_0.json"
)


_NO_FRAGILITY = FragilityReport(min_steps_to_break=-1, breaking_sequences=[], is_fragile=False)


class _PlanAgent:
    def __init__(self, actions):
        self._actions = iter(actions)
        self.last_usage = None
        self.prompts: list[str] = []

    def __call__(self, messages):
        self.prompts.append(json.dumps(messages))
        self.last_usage = {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}
        return f"FINAL_OUTPUT: {next(self._actions, 'DONE')}"


def test_text_only_mechanics_unit_runs_on_the_fleet(tmp_path, monkeypatch):
    # Static scoring's fragility search takes minutes on this 18x18 maze and has
    # no bearing on the fleet path; every other score dimension runs for real.
    monkeypatch.setattr(TaskValidator, "compute_fragility", lambda self, depth_limit=5: _NO_FRAGILITY)
    cfg = tmp_path / "run_config.json"
    cfg.write_text(json.dumps({
        "experiment_config": {"observation": "text_only", "querying": "step_by_step"},
        "models": {"stub": {"provider": "claude", "model": "stub-model", "group": "stub",
                            "tasks": [str(M10)]}},
    }), encoding="utf-8")
    artifacts = tmp_path / "coordinator"
    [planned] = prepare_job(
        run_config_path=cfg, manifest_path=_MANIFEST, seeds=[0], conditions=None,
        artifacts_root=artifacts, run_set_id="nl", difficulty_max_static_score=10000.0,  # M10 scores ~7840
    )["units"]
    assert planned["backend"] == "minigrid" and planned["experiment_config"]["observation"] == "text_only"

    store = CoordinatorStore(artifacts)
    worker_id = store.register({"worker_id": "w", "capabilities": {"model_group": "stub"}})["worker_id"]
    unit = store.assign(worker_id)["unit"]
    plan = plan_bfs_path(task_spec_from_payload(unit["task_payload"]))
    assert plan.success
    agent = _PlanAgent([ACTION_ORDER[a] for a in plan.actions])

    worker_root = tmp_path / "worker"
    run_assigned_unit(unit, artifacts_root=worker_root, agent_factory=lambda key, c: (agent, "stub-model"))
    run_dir = worker_root / unit["run_dir"]
    assert load_json(run_dir / "episode.json")["success"] is True
    store.upload(worker_id, unit["unit_id"], package_run_archive(unit, artifacts_root=worker_root).read_bytes())
    assert store.load_state()["units"][unit["unit_id"]]["status"] == "verified"

    first = agent.prompts[0]
    for phrase in ("death portal at", "frozen tile at", "rotating tile at", "teleporter between"):
        assert phrase in first
    assert '"type": "image_url"' not in first  # text only: no frame reaches the model
