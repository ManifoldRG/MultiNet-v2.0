"""The hex smoke run end to end, at zero API cost.

The real pipeline (Stage 2 canonical paths and static score, Stage 3 on the
MultiGrid backend with the R2 cell, Stage 4 runtime score) runs each hex maze
of manifest.mechanics_15_hex.json with run_config.mechanics_15_hex_2d_astra.json,
the model replaced by an agent replaying the planner's optimal actions. Every
maze must be solved in exactly its BFS step count with the artifacts the R2
run writes.
"""

import json
from pathlib import Path

import pytest

from gridworld.baselines import plan_bfs_path
from gridworld.render_settings import RenderSettings
from gridworld.task_spec import TaskSpecification
from interface.parser import ACTION_ORDER
from rulebook_test_utils import PlannerReplayAgent
from scripts.run_pipeline import run_from_config

_REPO = Path(__file__).resolve().parents[1]
_FIXTURES = _REPO / "gridworld" / "fixtures"
_MANIFEST = _FIXTURES / "manifest.mechanics_15_hex.json"
_RUN_CONFIG = _FIXTURES / "run_config.mechanics_15_hex_2d_astra.json"
_ROWS = json.loads(_MANIFEST.read_text())["tasks"]


def _row_params():
    for row in _ROWS:
        # M10's Stage 2 alone takes ~20 s (BFS over six facings and six-phase
        # turntables); the default suite runs the four single-mechanic mazes.
        marks = [pytest.mark.slow] if row["maze_category"] == "M10" else []
        yield pytest.param(row, id=row["task_id"], marks=marks)


@pytest.mark.parametrize("row", list(_row_params()))
def test_planner_actions_solve_the_hex_maze_in_bfs_steps(tmp_path, row):
    run_config = json.loads(_RUN_CONFIG.read_text())
    (model,) = run_config["models"].values()
    model["tasks"] = [row["task_id"]]
    config_path = tmp_path / "run_config.json"
    config_path.write_text(json.dumps(run_config))

    spec = TaskSpecification.from_json(str(_REPO / row["source"]))
    plan = [ACTION_ORDER[a] for a in plan_bfs_path(spec).actions]
    assert len(plan) == row["optimal_steps"]
    agent = PlannerReplayAgent([plan])
    artifacts = tmp_path / "artifacts"
    run_from_config(
        run_config_path=config_path,
        manifest_path=_MANIFEST,
        artifacts_root=artifacts,
        run_set_id="hex_scripted",
        difficulty_max_static_score=20000,
        agent_factory=lambda name, cfg: (agent, "scripted-bfs"),
    )
    assert agent.episodes == 1

    label = RenderSettings.from_run_config(run_config).label
    run_dir = artifacts / "runs" / row["task_id"] / label / "scripted-bfs" / "seed_0" / "default"
    canonical = json.loads((artifacts / "tasks" / row["task_id"] / "canonical_paths.json").read_text())
    assert canonical["bfs"]["optimal_steps"] == row["optimal_steps"]

    episode = json.loads((run_dir / "episode.json").read_text())
    steps = [r for r in episode["transcript"] if r.get("kind") == "step"]
    assert episode["success"] and episode["end_reason"] == "success"
    assert len(steps) == episode["steps_used"] == row["optimal_steps"]
    assert all(r["state_before"]["tiling"] == "hex" == r["state_after"]["tiling"] for r in steps)
    assert [r["action"] for r in steps] == plan
    assert episode["render"]["tiling"] == "hex"
    assert len(list((run_dir / "frames").glob("*.png"))) == len(steps) + 1  # reset + one per step

    run_inputs = json.loads((run_dir / "run_inputs.json").read_text())
    assert run_inputs["backend"] == label
    assert run_inputs["runtime_max_steps_cap"]["effective_max_steps"] == 3 * row["optimal_steps"]
    signals = json.loads((run_dir / "run_score.json").read_text())["signals"]
    assert signals["success"] and signals["steps"] == row["optimal_steps"]
    assert signals["step_ratio"] == 1.0 and signals["cell_overlap_bfs"] == 1.0

    # The model-facing prompt is R2's minimal one, word for word.
    query = json.loads(next((run_dir / "queries").glob("query_001/*.json")).read_text())
    system = query["agent_messages"][0]["content"]
    assert system == (
        "Task: Solve the maze by reaching the goal.\n"
        "Valid actions: TURN_LEFT, TURN_RIGHT, MOVE_FORWARD, PICKUP, DROP, TOGGLE, DONE."
    )
