from pathlib import Path

import pytest

from gridworld.actions import MiniGridActions
from gridworld.baselines import plan_bfs_path
from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.custom_env import KillCell, Key
from gridworld.task_spec import TaskSpecification
from gridworld.task_validator import TaskValidator


_M7_DEATH = (
    Path(__file__).resolve().parent.parent
    / "M7-10 New Mechanisms"
    / "M7-10 New Mechanisms"
    / "M7"
    / "18x18_dense_kr_sg_kb_death_0.json"
)


def _spec(**overrides):
    d = {
        "task_id": "kill_test",
        "seed": 1,
        "difficulty_tier": 1,
        "maze": {"dimensions": [8, 8], "walls": [], "start": [1, 1], "goal": [6, 6]},
        "mechanisms": {
            "keys": [{"id": "kR", "position": [2, 1], "color": "red"}],
            "death_portals": [[3, 1]],
        },
        "goal": {"type": "reach_position", "target": [6, 6]},
        "max_steps": 50,
    }
    d.update(overrides)
    return TaskSpecification.from_dict(d)


def test_stepping_on_a_kill_cell_returns_to_start():
    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(_spec())
    backend.reset(seed=1)
    assert isinstance(backend.env.grid.get(3, 1), KillCell)
    backend.env.step(MiniGridActions.MOVE_FORWARD)
    backend.env.step(MiniGridActions.PICKUP)
    assert backend.env.carrying is not None
    before_steps = backend.env.step_count
    _, _, terminated, truncated, _info = backend.env.step(MiniGridActions.MOVE_FORWARD)

    assert tuple(backend.env.agent_pos) == (1, 1)
    assert backend.env.agent_dir == 0
    assert backend.env.carrying is None
    assert backend.env.collected_keys == set()
    assert backend.env.step_count == before_steps + 1
    assert terminated is False
    assert truncated is False
    keys = {
        getattr(backend.env.grid.get(x, y), "key_id", None)
        for x in range(backend.env.width)
        for y in range(backend.env.height)
        if isinstance(backend.env.grid.get(x, y), Key)
    }
    assert "kR" in keys


def test_bfs_avoids_kill_cells_on_the_optimal_path():
    spec = _spec()
    path = plan_bfs_path(spec)
    assert path.success
    assert (3, 1) not in path.positions


def test_validator_finds_a_safe_route():
    ok, positions, message = TaskValidator(_spec()).validate()
    assert ok, message
    assert (3, 1) not in (positions or [])


@pytest.mark.skipif(not _M7_DEATH.exists(), reason="local M7 example not present")
def test_m7_death_maze_is_beatable():
    spec = TaskSpecification.from_json(str(_M7_DEATH))
    ok, errors = spec.validate()
    assert ok, errors
    assert len(spec.mechanisms.kill_cells) == 5
    path = plan_bfs_path(spec)
    assert path.success
    kill = {p.to_tuple() for p in spec.mechanisms.kill_cells}
    assert kill.isdisjoint(path.positions)
    ok, positions, message = TaskValidator(spec).validate()
    assert ok, message
