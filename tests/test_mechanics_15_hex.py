"""The hex mechanics panel: converted mazes, their manifest, and run configs
that differ from the R2 2D run only in the tiling."""

import json
from pathlib import Path

import pytest

from gridworld.baselines import plan_bfs_path
from gridworld.render_settings import RenderSettings
from gridworld.task_spec import TaskSpecification
from gridworld.tiling import neighbor
from scripts import make_hex_mechanics
from scripts.run_pipeline import _build_agent_from_spec, check_run_config_expectations, load_run_config

_REPO = Path(__file__).resolve().parents[1]
_FIX = _REPO / "gridworld" / "fixtures"
_MANIFEST = json.loads((_FIX / "manifest.mechanics_15_hex.json").read_text())
_SQUARE = {t["task_id"]: t for t in json.loads((_FIX / "manifest.mechanics_15.json").read_text())["tasks"]}
_R2 = json.loads((_FIX / "run_config.mechanics_15_2d.json").read_text())
_HEX = json.loads((_FIX / "run_config.mechanics_15_hex_2d.json").read_text())
_SMOKE = json.loads((_FIX / "run_config.mechanics_15_hex_2d_astra.json").read_text())

pytestmark = pytest.mark.skipif(
    not (_REPO / next(iter(_SQUARE.values()))["source"]).exists(), reason="ogbench submodule not checked out"
)


def test_committed_mazes_and_manifest_are_the_converters_output():
    # Re-runs every check: solvable, BFS in R2's range, no key/switch bypassed.
    assert make_hex_mechanics.main(["--check"]) == 0


def test_one_maze_per_mechanic_and_one_with_all_four():
    combos = [frozenset(t["new_mechanisms"]) for t in _MANIFEST["tasks"]]
    assert combos == [
        {"portals"}, {"death"}, {"rotating"}, {"frozen"}, {"portals", "death", "rotating", "frozen"},
    ]


@pytest.mark.parametrize("row", _MANIFEST["tasks"], ids=lambda r: r["task_id"])
def test_hex_maze_keeps_the_square_layout_and_solves_in_range(row):
    hexed = TaskSpecification.from_json(str(_REPO / row["source"]))
    square = TaskSpecification.from_json(str(_REPO / _SQUARE[row["square_task_id"]]["source"]))
    assert hexed.maze.tiling == "hex" and hexed.metadata["tiling"] == "hex"
    assert hexed.maze.walls == square.maze.walls
    assert hexed.maze.start == square.maze.start and hexed.resolved_goal() == square.resolved_goal()
    for kind in ("keys", "doors", "switches", "gates", "teleporters", "kill_cells", "frozen_tiles", "rotating_tiles"):
        assert getattr(hexed.mechanisms, kind) == getattr(square.mechanisms, kind), kind
    # Each turntable first points at the cell its square arrow pointed at.
    for cell, hex_dir, square_dir in zip(
        hexed.mechanisms.rotating_tiles,
        hexed.mechanisms.rotating_initial_directions,
        square.mechanisms.rotating_initial_directions,
    ):
        assert neighbor("hex", cell.to_tuple(), hex_dir) == neighbor("square", cell.to_tuple(), square_dir)

    path = plan_bfs_path(hexed)
    assert path.success and len(path.action_labels) == row["optimal_steps"]
    assert 49 <= row["optimal_steps"] <= 161
    assert row["square_optimal_steps"] == _SQUARE[row["square_task_id"]]["optimal_steps"]
    assert (_REPO / row["source"]).parent == make_hex_mechanics.OUT_DIR


def test_hex_run_config_is_the_2d_run_on_the_hex_manifest():
    assert _HEX["experiment_config"] == _R2["experiment_config"]
    assert _HEX["render"] == {"backend": "multigrid"}
    assert RenderSettings.from_run_config(_HEX).backend == "multigrid"
    assert set(_HEX["models"]) == {"claude_opus55", "gpt6_astra"}
    for name, model in _HEX["models"].items():
        assert model == _R2["models"][name]
    check_run_config_expectations(
        load_run_config(_FIX / "run_config.mechanics_15_hex_2d.json"), _FIX / "manifest.mechanics_15_hex.json", None
    )


def test_astra_smoke_config_differs_only_in_its_spend_cap(monkeypatch):
    assert _SMOKE["experiment_config"] == _R2["experiment_config"]
    assert _SMOKE["render"] == _HEX["render"] and _SMOKE["manifest"] == _HEX["manifest"]
    (astra,) = _SMOKE["models"].values()
    assert {**astra, "spend_cap_usd": _R2["models"]["gpt6_astra"]["spend_cap_usd"]} == _R2["models"]["gpt6_astra"]
    assert astra["spend_cap_usd"] == 100
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    agent, label = _build_agent_from_spec("gpt6_astra", astra)
    assert label == "gpt-6-astra" and agent.config.spend_cap_usd == 100
