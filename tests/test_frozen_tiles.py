from pathlib import Path

import pytest

from gridworld.actions import MiniGridActions
from gridworld.baselines import plan_bfs_path
from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.custom_env import FrozenTile
from gridworld.task_spec import TaskSpecification
from gridworld.task_validator import TaskValidator
from gridworld.world_model import TaskPlanningContext, apply


_M7_FROZEN = (
    Path(__file__).resolve().parent.parent
    / "M7-10 New Mechanisms"
    / "M7-10 New Mechanisms"
    / "M7"
    / "18x18_dense_kr_sg_kb_frozen_0.json"
)


def _spec(**overrides):
    d = {
        "task_id": "freeze_test",
        "seed": 1,
        "difficulty_tier": 1,
        "maze": {
            "dimensions": [8, 8],
            "walls": [[1, 2], [2, 2], [3, 2], [4, 2], [5, 2], [6, 2]],
            "start": [1, 1],
            "goal": [6, 1],
        },
        "mechanisms": {
            "frozen_tiles": [[2, 1]],
            "freeze_steps": 5,
        },
        "goal": {"type": "reach_position", "target": [6, 1]},
        "max_steps": 50,
    }
    d.update(overrides)
    return TaskSpecification.from_dict(d)


def test_any_action_ticks_freeze_without_moving():
    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(_spec())
    backend.reset(seed=1)
    assert isinstance(backend.env.grid.get(2, 1), FrozenTile)
    backend.env.step(MiniGridActions.MOVE_FORWARD)
    assert tuple(backend.env.agent_pos) == (2, 1)
    assert backend.env.agent_dir == 0
    assert backend.env.freeze_remaining == 5

    for action in (
        MiniGridActions.MOVE_FORWARD,
        MiniGridActions.TURN_LEFT,
        MiniGridActions.PICKUP,
        MiniGridActions.TOGGLE,
        MiniGridActions.DROP,
    ):
        before = backend.env.freeze_remaining
        backend.env.step(action)
        assert tuple(backend.env.agent_pos) == (2, 1)
        assert backend.env.agent_dir == 0
        assert backend.env.freeze_remaining == before - 1

    assert backend.env.freeze_remaining == 0
    backend.env.step(MiniGridActions.MOVE_FORWARD)
    assert tuple(backend.env.agent_pos) == (3, 1)


def test_leaving_and_reentering_freezes_again():
    ctx = TaskPlanningContext(_spec())
    state = ctx.initial_state()
    state = apply(ctx, state, MiniGridActions.MOVE_FORWARD)
    assert state.agent_pos == (2, 1)
    assert state.freeze_remaining == 5
    for _ in range(5):
        state = apply(ctx, state, MiniGridActions.MOVE_FORWARD)
    assert state.freeze_remaining == 0
    state = apply(ctx, state, MiniGridActions.MOVE_FORWARD)
    assert state.agent_pos == (3, 1)
    state = apply(ctx, state, MiniGridActions.TURN_LEFT)
    state = apply(ctx, state, MiniGridActions.TURN_LEFT)
    state = apply(ctx, state, MiniGridActions.MOVE_FORWARD)
    assert state.agent_pos == (2, 1)
    assert state.freeze_remaining == 5


def test_bfs_pays_freeze_steps_on_the_ice():
    path = plan_bfs_path(_spec())
    assert path.success
    assert path.positions.count((2, 1)) == 6
    freeze_ticks = sum(1 for label in path.action_labels if label == "freeze")
    assert freeze_ticks == 5
    assert path.actions.count(int(MiniGridActions.DONE)) == 0


def test_validator_finds_a_route_across_ice():
    ok, positions, message = TaskValidator(_spec()).validate()
    assert ok, message
    assert (2, 1) in (positions or [])


@pytest.mark.skipif(not _M7_FROZEN.exists(), reason="local M7 example not present")
def test_m7_frozen_maze_is_beatable():
    spec = TaskSpecification.from_json(str(_M7_FROZEN))
    ok, errors = spec.validate()
    assert ok, errors
    assert len(spec.mechanisms.frozen_tiles) == 5
    assert spec.mechanisms.freeze_steps == 5
    path = plan_bfs_path(spec)
    assert path.success
    ok, positions, message = TaskValidator(spec).validate()
    assert ok, message
