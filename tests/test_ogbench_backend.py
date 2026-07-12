"""Tests for OgbenchBackend (OGBench PointyEnv, continuous navigation).

Skipped wherever mujoco isn't importable (e.g. x86_64 Python on Apple
Silicon) — see gridworld/backends/ogbench_backend.py's module docstring.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("mujoco")

from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.backends.ogbench_backend import OgbenchBackend
from gridworld.task_spec import TaskSpecification
from interface.coords import agent_facing, agent_row_col, to_row_col
from interface.feedback import format_step_feedback

_POS_TOL = 1e-3
_DEG_TOL = 1.0


def _nav_task(**overrides):
    payload = {
        "task_id": "ogbench_nav",
        "seed": 5,
        "difficulty_tier": 1,
        "maze": {
            "dimensions": [8, 8],
            "walls": [[2, 1]],
            "start": [1, 1],
            "goal": [6, 6],
        },
        "mechanisms": {},
        "goal": {"type": "reach_position", "target": [6, 6]},
        "max_steps": 50,
    }
    payload.update(overrides)
    return TaskSpecification.from_dict(payload)


def _open_task():
    """A nav task with no walls near the start, for testing partial/continuous
    forward movement without a nearby wall's collision radius interfering."""
    return TaskSpecification.from_dict(
        {
            "task_id": "ogbench_open",
            "seed": 5,
            "difficulty_tier": 1,
            "maze": {
                "dimensions": [8, 8],
                "walls": [],
                "start": [3, 3],
                "goal": [6, 6],
            },
            "mechanisms": {},
            "goal": {"type": "reach_position", "target": [6, 6]},
            "max_steps": 50,
        }
    )


def _key_door_task():
    # Wall at [2, 2] closes the corridor above the door at [2, 1] so OGBench's
    # wall-distance-based door-geometry inference (maze.py::_compute_door_geometry)
    # centers/sizes the door sensibly on its own cell instead of degenerating to
    # spanning the whole open room (which happens with no walls anywhere nearby).
    return TaskSpecification.from_dict(
        {
            "task_id": "ogbench_key_door",
            "seed": 7,
            "difficulty_tier": 2,
            "maze": {
                "dimensions": [8, 8],
                "walls": [[2, 2]],
                "start": [1, 1],
                "goal": [6, 6],
            },
            "mechanisms": {
                "keys": [{"id": "k1", "position": [1, 1], "color": "red"}],
                "doors": [
                    {
                        "id": "d1",
                        "position": [2, 1],
                        "requires_key": "red",
                        "initial_state": "locked",
                    }
                ],
            },
            "goal": {"type": "reach_position", "target": [6, 6]},
            "max_steps": 50,
        }
    )


def _make_backend(task_spec) -> OgbenchBackend:
    backend = OgbenchBackend(render_mode="rgb_array")
    backend.configure(task_spec)
    return backend


def test_reset_lands_on_exact_start_cell_facing_east():
    spec = _nav_task()
    backend = _make_backend(spec)
    _, state, _ = backend.reset(seed=spec.seed)

    row, col = agent_row_col(state)
    assert (row, col) == to_row_col(spec.maze.start)
    assert agent_facing(state) == "EAST"
    assert state.step_count == 0


def test_move_forward_partial_and_full_accumulate():
    spec = _open_task()
    backend = _make_backend(spec)
    _, state, _ = backend.reset(seed=spec.seed)
    x0, y0 = state.agent_position

    action = backend.parse_action("MOVE_FORWARD(0.5)")
    _, _, _, _, state, _ = backend.step(action)
    x1, y1 = state.agent_position
    # Facing EAST (task-spec +x); half a step should move roughly half a cell.
    assert x1 - x0 == pytest.approx(0.5, abs=0.15)
    assert y1 == pytest.approx(y0, abs=_POS_TOL)

    action = backend.parse_action("MOVE_FORWARD(0.5)")
    _, _, _, _, state, _ = backend.step(action)
    x2, _ = state.agent_position
    assert x2 - x0 == pytest.approx(1.0, abs=0.2)


def test_turn_default_and_explicit_magnitude():
    spec = _nav_task()
    backend = _make_backend(spec)
    _, state, _ = backend.reset(seed=spec.seed)

    action = backend.parse_action("TURN_LEFT")
    _, _, _, _, state, _ = backend.step(action)
    assert agent_facing(state) == "NORTH"

    action = backend.parse_action("TURN_RIGHT(45)")
    _, _, _, _, state2, _ = backend.step(action)
    assert state2.agent_direction == pytest.approx(state.agent_direction - 45.0, abs=_DEG_TOL)


def test_move_into_wall_is_blocked_and_position_unchanged():
    spec = _nav_task()  # wall directly east of start at [2, 1]
    backend = _make_backend(spec)
    _, prev_state, _ = backend.reset(seed=spec.seed)

    action = backend.parse_action("MOVE_FORWARD(1.0)")
    _, reward, terminated, truncated, curr_state, info = backend.step(action)

    assert info.get("movement_blocked") is True
    px, py = prev_state.agent_position
    cx, cy = curr_state.agent_position
    assert cx == pytest.approx(px, abs=_POS_TOL)
    assert cy == pytest.approx(py, abs=_POS_TOL)

    _, event_type = format_step_feedback(
        "MOVE_FORWARD(1.0)", prev_state, curr_state, reward, terminated, spec, info
    )
    assert event_type == "BLOCKED"


def test_pickup_and_toggle_door():
    # Key k1 sits on the start cell [1,1]; door d1 is one cell east at [2,1]
    # (interact radii are ~0.5-0.6 * maze_unit, smaller than a full cell, so
    # the agent must approach before TOGGLE reaches the door panel).
    spec = _key_door_task()
    backend = _make_backend(spec)
    _, state, _ = backend.reset(seed=spec.seed)
    assert "k1" not in state.collected_keys
    assert "d1" not in state.open_doors

    _, _, _, _, state, _ = backend.step(backend.parse_action("PICKUP"))
    assert "k1" in state.collected_keys
    assert state.agent_carrying == "red"

    # Approach the (still-locked) door in small increments — a single full
    # MOVE_FORWARD(1.0) would sweep the collision disk into the closed door's
    # panel and get rejected outright (movement_blocked), leaving the agent
    # too far away to toggle; smaller steps get as close as collision allows.
    for _ in range(4):
        backend.step(backend.parse_action("MOVE_FORWARD(0.3)"))
    _, _, _, _, state, _ = backend.step(backend.parse_action("TOGGLE"))
    assert "d1" in state.open_doors


def test_done_wrong_then_correct():
    spec = _nav_task()
    backend = _make_backend(spec)
    _, prev_state, _ = backend.reset(seed=spec.seed)

    action = backend.parse_action("DONE")
    _, reward, terminated, truncated, curr_state, info = backend.step(action)
    _, event_type = format_step_feedback(
        "DONE", prev_state, curr_state, reward, terminated, spec, info
    )
    assert event_type == "WRONG_DONE"
    assert not terminated

    # Teleport (via repeated set_xy) straight to the goal cell center, then DONE.
    goal_xy = backend.env.unwrapped.ij_to_xy(backend._goal_ij)
    backend.env.unwrapped.set_xy(goal_xy)
    prev_state = backend.get_state()
    action = backend.parse_action("DONE")
    _, reward, terminated, truncated, curr_state, info = backend.step(action)
    _, event_type = format_step_feedback(
        "DONE", prev_state, curr_state, reward, terminated, spec, info
    )
    assert event_type == "DONE"
    assert terminated
    assert curr_state.goal_reached


def test_unsupported_mechanisms_rejected_at_configure():
    spec = TaskSpecification.from_dict(
        {
            "task_id": "ogbench_teleporters_unsupported",
            "seed": 1,
            "difficulty_tier": 1,
            "maze": {"dimensions": [8, 8], "walls": [], "start": [1, 1], "goal": [6, 6]},
            "mechanisms": {
                "teleporters": [
                    {
                        "id": "t1",
                        "position_a": [2, 2],
                        "position_b": [5, 5],
                    }
                ]
            },
            "goal": {"type": "reach_position", "target": [6, 6]},
            "max_steps": 50,
        }
    )
    backend = OgbenchBackend(render_mode="rgb_array")
    with pytest.raises(ValueError):
        backend.configure(spec)


def test_keys_without_doors_or_switches_rejected_at_configure():
    spec = TaskSpecification.from_dict(
        {
            "task_id": "ogbench_orphan_key",
            "seed": 1,
            "difficulty_tier": 1,
            "maze": {"dimensions": [8, 8], "walls": [], "start": [1, 1], "goal": [6, 6]},
            "mechanisms": {"keys": [{"id": "k1", "position": [2, 2], "color": "red"}]},
            "goal": {"type": "reach_position", "target": [6, 6]},
            "max_steps": 50,
        }
    )
    backend = OgbenchBackend(render_mode="rgb_array")
    with pytest.raises(ValueError):
        backend.configure(spec)


def test_minigrid_regression_start_cell_and_facing_unaffected():
    """The shared parser.py/coords.py/feedback.py/runner.py seam changes made
    for OgbenchBackend must not alter MiniGridBackend's discrete behavior."""
    spec = _nav_task()
    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(spec)
    _, state, _ = backend.reset(seed=spec.seed)

    assert agent_row_col(state) == to_row_col(spec.maze.start)
    assert agent_facing(state) == "EAST"
    assert backend.parse_action("TURN_LEFT") == backend.parse_action("TURN_LEFT")
    assert isinstance(backend.parse_action("MOVE_FORWARD"), int)
