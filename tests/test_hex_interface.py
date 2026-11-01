"""A hex task through the interface and pipeline plumbing: facing names and
feedback on six directions, the multigrid render setting, the fail-fast
checks (wrong backend, cardinal actions, 3D) and the NL text."""

import dataclasses
import json

import pytest

from gridworld.actions import MiniGridActions as A
from gridworld.backends.multigrid_backend import MultiGridBackend
from gridworld.render2d import RENDER2D_VERSION
from gridworld.render3d.scene import UnsupportedSpecError, check_supported
from gridworld.render_settings import RenderSettings
from interface.config import ExperimentConfig
from interface.coords import agent_facing, direction_name, forward_cell
from interface.feedback import format_step_feedback
from interface.renderer import render_initial_maze_text, render_user_observation_text
from interface.runner import build_runner
from rulebook_test_utils import every_mechanic
from scripts.run_pipeline import run_from_config


def _backend(spec):
    backend = MultiGridBackend()
    backend.configure(spec)
    backend.reset(seed=spec.seed)
    return backend


# --------------------------------------------------------------------------- #
# Facing names and feedback
# --------------------------------------------------------------------------- #
def test_hex_facing_names_follow_the_six_directions():
    backend = _backend(every_mechanic("hex"))
    names = []
    for _ in range(6):
        names.append(agent_facing(backend.get_state()))
        backend.step(A.TURN_RIGHT)
    assert names == ["EAST", "SOUTHEAST", "SOUTHWEST", "WEST", "NORTHWEST", "NORTHEAST"]
    assert direction_name(3) == "NORTH" and direction_name(3, "hex") == "WEST"


def test_hex_forward_cell_is_the_hex_neighbour():
    backend = _backend(every_mechanic("hex"))
    state = backend.get_state()  # (1, 1) facing east
    assert forward_cell(state) == (1, 2)  # (row, col)
    *_, state, _ = backend.step(A.TURN_RIGHT)  # southeast from odd row 1
    assert forward_cell(state) == (2, 2)
    *_, state, _ = backend.step(A.TURN_RIGHT)  # southwest
    assert forward_cell(state) == (2, 1)


def test_hex_feedback_names_the_door_ahead_and_turntable_directions():
    spec = every_mechanic("hex")
    backend = _backend(spec)
    # Walk to (6, 1), face southeast: the red door at (7, 2) is ahead.
    for action in [A.MOVE_FORWARD] * 5 + [A.TURN_RIGHT]:
        backend.step(action)
    prev = backend.get_state()
    *_, reward, terminated, _trunc, state, _info = backend.step(A.MOVE_FORWARD)
    text, event = format_step_feedback("MOVE_FORWARD", prev, state, reward, terminated, spec, level="standard")
    assert event == "BLOCKED" and "red door" in text
    text, event = format_step_feedback(
        "MOVE_FORWARD", prev, prev, 0.0, False, spec, level="minimal",
        mechanic={"kind": "carried", "from_xy": (3, 5), "to_xy": (4, 5), "direction": 4},
    )
    assert (text, event) == ("CARRIED NORTHWEST", "CARRIED")


# --------------------------------------------------------------------------- #
# Action space
# --------------------------------------------------------------------------- #
def test_cardinal_actions_are_refused_on_hex():
    spec = every_mechanic("hex")
    backend = MultiGridBackend()
    backend.configure(spec)
    with pytest.raises(ValueError, match="cardinal action space .* square-only"):
        build_runner(ExperimentConfig(action_space="cardinal"), backend, spec)
    build_runner(ExperimentConfig(action_space="egocentric"), backend, spec)


# --------------------------------------------------------------------------- #
# Render settings
# --------------------------------------------------------------------------- #
def test_multigrid_render_setting():
    render = RenderSettings.from_run_config({"render": {"backend": "multigrid"}})
    hex_spec = every_mechanic("hex")
    assert render.label == f"multigrid_r{RENDER2D_VERSION}"
    assert render.backend_kwargs(hex_spec) == {"tiling": "hex"}
    provenance = render.provenance(hex_spec)
    assert provenance["tiling"] == "hex" and provenance["frame_shape"] == [280, 330]
    assert provenance["render2d_version"] == RENDER2D_VERSION
    # A distributed unit carries the whole dataclass and must parse back.
    block = dataclasses.asdict(render)
    assert RenderSettings.from_run_config({"render": block}) == render
    render.check_observation("text_only")  # a state backend: the NL run is fine
    for bad in ({"camera": "top_down"}, {"resolution": 256}, {"start_map": True}):
        with pytest.raises(ValueError, match="3D setting"):
            RenderSettings.from_run_config({"render": {"backend": "multigrid", **bad}})


def test_backends_check_the_tiling():
    hex_spec, square_spec = every_mechanic("hex"), every_mechanic("square")
    RenderSettings(backend="multigrid").check_tiling(hex_spec)
    RenderSettings(backend="multigrid").check_tiling(square_spec)
    RenderSettings().check_tiling(square_spec)
    for backend in ("minigrid", "mujoco3d"):
        with pytest.raises(ValueError, match='render: {"backend": "multigrid"}'):
            RenderSettings(backend=backend, camera="top_down" if backend == "mujoco3d" else None).check_tiling(hex_spec)
    with pytest.raises(UnsupportedSpecError, match="tiling='hex'"):
        check_supported(hex_spec)


def _hex_task(tmp_path):
    path = tmp_path / "hex_task.json"
    path.write_text(json.dumps(every_mechanic("hex").to_dict()))
    return path


class _NoSpendAgent:
    def __call__(self, messages):
        raise AssertionError("a run that cannot start must send no query")


def _run(tmp_path, run_config):
    cfg = tmp_path / "run_config.json"
    cfg.write_text(json.dumps(run_config))
    return run_from_config(
        run_config_path=cfg,
        manifest_path=tmp_path / "manifest.json",
        artifacts_root=tmp_path / "artifacts",
        agent_factory=lambda name, model_cfg: (_NoSpendAgent(), "stub"),
        difficulty_max_static_score=1000.0,
    )


@pytest.fixture
def hex_manifest(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"tasks": []}))
    return _hex_task(tmp_path)


def test_run_from_config_refuses_a_hex_task_on_minigrid_before_stage_2(tmp_path, hex_manifest):
    models = {"stub": {"provider": "claude", "model": "stub", "tasks": [str(hex_manifest)]}}
    with pytest.raises(ValueError, match="hex maze but the minigrid backend"):
        _run(tmp_path, {"models": models})
    assert not (tmp_path / "artifacts" / "tasks").exists()


def test_preflight_refuses_hex_on_minigrid_and_cardinal_on_hex(tmp_path, hex_manifest):
    from scripts.run_pipeline import _condition_configs, check_tasks_runnable, resolve_task_rows

    rows = resolve_task_rows([str(hex_manifest)], [], tmp_path / "manifest.json")
    cells = _condition_configs(None)
    with pytest.raises(ValueError, match="hex maze but the minigrid backend"):
        check_tasks_runnable(rows, tmp_path / "manifest.json", RenderSettings(), cells)
    multigrid = RenderSettings(backend="multigrid")
    check_tasks_runnable(rows, tmp_path / "manifest.json", multigrid, cells)
    cardinal = _condition_configs(None, base_overrides={"action_space": "cardinal"})
    with pytest.raises(ValueError, match="cardinal action space"):
        check_tasks_runnable(rows, tmp_path / "manifest.json", multigrid, cardinal)
    assert not (tmp_path / "artifacts").exists()


# --------------------------------------------------------------------------- #
# NL text
# --------------------------------------------------------------------------- #
def test_hex_coords_text_explains_hex_neighbours():
    spec = every_mechanic("hex")
    text = render_initial_maze_text(spec)
    assert text.startswith("The world is a hexagonal grid of 7 rows and 7 columns.")
    assert "every odd-numbered row sits half a cell further east" in text
    assert "rotating tile at (5,3) pointing SOUTHWEST" in text
    backend = _backend(spec)
    backend.step(A.TURN_LEFT)
    current = render_user_observation_text(spec, backend.get_state(), include_facing=True)
    assert current.startswith("You are at (1, 1) facing NORTHEAST.")


def test_hex_ascii_map_indents_odd_rows_and_json_names_the_tiling():
    spec = every_mechanic("hex")
    ascii_text = render_initial_maze_text(spec, observation_text_format="ascii", include_facing=True)
    assert "odd-numbered lines are indented" in ascii_text
    lines = ascii_text.split("\n")
    first = lines.index("  A  .  kR .  .  .  .")  # row 1 (odd): indented
    assert lines[first + 1].startswith("#  xx #")  # row 2 (even): not
    assert "SW = rotating tile pointing SOUTHWEST" in ascii_text
    assert "A = you, facing EAST" in ascii_text
    payload = json.loads(render_initial_maze_text(spec, observation_text_format="json"))
    assert payload["world"] == {"rows": 7, "cols": 7, "tiling": "hex"}
    assert "six neighbours" in payload["coordinates"]
