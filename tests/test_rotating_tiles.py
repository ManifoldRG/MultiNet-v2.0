from pathlib import Path

import pytest

from gridworld.actions import MiniGridActions
from gridworld.baselines import plan_bfs_path
from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.custom_env import RotatingTile
from gridworld.task_spec import TaskSpecification
from gridworld.task_validator import TaskValidator
from gridworld.world_model import TaskPlanningContext, apply


_M7_ROTATE = (
    Path(__file__).resolve().parent.parent
    / "M7-10 New Mechanisms"
    / "M7-10 New Mechanisms"
    / "M7"
    / "18x18_dense_kr_sg_kb_rotating_0.json"
)


def _spec(**overrides):
    d = {
        "task_id": "rotate_test",
        "seed": 1,
        "difficulty_tier": 1,
        "maze": {
            "dimensions": [8, 8],
            "walls": [[1, 2], [2, 2], [3, 2], [4, 2], [5, 2], [6, 2]],
            "start": [1, 1],
            "goal": [6, 1],
        },
        "mechanisms": {
            "rotating_tiles": [[2, 1]],
            "rotating_initial_directions": [0],
        },
        "goal": {"type": "reach_position", "target": [6, 1]},
        "max_steps": 50,
    }
    d.update(overrides)
    return TaskSpecification.from_dict(d)


def test_turns_wait_for_the_arrow_then_forward_rides_it():
    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(_spec())
    backend.reset(seed=1)
    tile = backend.env.grid.get(2, 1)
    assert isinstance(tile, RotatingTile)
    assert tile.direction == 0

    backend.env.step(MiniGridActions.MOVE_FORWARD)
    assert tuple(backend.env.agent_pos) == (2, 1)
    assert tile.direction == 1

    backend.env.step(MiniGridActions.TURN_LEFT)
    assert tuple(backend.env.agent_pos) == (2, 1)
    assert tile.direction == 2

    backend.env.step(MiniGridActions.TURN_LEFT)
    backend.env.step(MiniGridActions.TURN_LEFT)
    assert tile.direction == 0

    backend.env.step(MiniGridActions.MOVE_FORWARD)
    assert tuple(backend.env.agent_pos) == (3, 1)


def test_illegal_action_still_spins():
    ctx = TaskPlanningContext(_spec())
    state = ctx.initial_state()
    assert state.rotator_dirs == (0,)
    state = apply(ctx, state, MiniGridActions.PICKUP)
    assert state.agent_pos == (1, 1)
    assert state.rotator_dirs == (1,)


def test_rotating_dirs_must_match_tiles():
    spec = _spec(mechanisms={"rotating_tiles": [[2, 1]], "rotating_initial_directions": []})
    ok, errors = spec.validate()
    assert not ok
    assert any("rotating_tiles and rotating_initial_directions" in e for e in errors)


def test_blocked_arrow_keeps_you_on_the_tile():
    ctx = TaskPlanningContext(_spec())
    state = ctx.initial_state()
    state = apply(ctx, state, MiniGridActions.MOVE_FORWARD)
    assert state.agent_pos == (2, 1)
    assert state.rotator_dirs == (1,)
    state = apply(ctx, state, MiniGridActions.MOVE_FORWARD)
    assert state.agent_pos == (2, 1)


def test_toggle_opens_a_door_from_a_rotator():
    spec = _spec(
        mechanisms={
            "rotating_tiles": [[2, 1]],
            "rotating_initial_directions": [0],
            "keys": [{"id": "kB", "position": [1, 1], "color": "blue"}],
            "doors": [{
                "id": "DB",
                "position": [3, 1],
                "requires_key": "blue",
                "initial_state": "locked",
            }],
        }
    )
    ctx = TaskPlanningContext(spec)
    state = ctx.initial_state()
    state = apply(ctx, state, MiniGridActions.PICKUP)
    state = apply(ctx, state, MiniGridActions.MOVE_FORWARD)
    assert state.agent_pos == (2, 1)
    assert state.carrying_key == "kB"
    state = apply(ctx, state, MiniGridActions.TOGGLE)
    assert state.agent_pos == (2, 1)
    assert "DB" in state.open_doors
    assert state.carrying_key is None


def test_bfs_can_cross_a_rotator():
    path = plan_bfs_path(_spec())
    assert path.success
    assert (2, 1) in path.positions
    assert path.positions[-1] == (6, 1)


def test_validator_finds_a_route_across_a_rotator():
    ok, positions, message = TaskValidator(_spec()).validate()
    assert ok, message
    assert (2, 1) in (positions or [])


@pytest.mark.skipif(not _M7_ROTATE.exists(), reason="local M7 example not present")
def test_m7_rotating_maze_is_beatable():
    spec = TaskSpecification.from_json(str(_M7_ROTATE))
    ok, errors = spec.validate()
    assert ok, errors
    assert len(spec.mechanisms.rotating_tiles) == 3
    path = plan_bfs_path(spec)
    assert path.success
    ok, positions, message = TaskValidator(spec).validate()
    assert ok, message
