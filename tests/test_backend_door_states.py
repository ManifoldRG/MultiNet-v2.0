"""door_states() = physical open/closed; GridState.open_doors = "unlocked"."""

from __future__ import annotations

import pytest

from gridworld.actions import MiniGridActions as A
from gridworld.backends.base import AbstractGridBackend, GridState
from gridworld.task_spec import TaskSpecification
from render3d_test_utils import CORRIDOR

TWO_DOORS = {
    **CORRIDOR,
    "mechanisms": {
        "doors": [
            {"id": "d1", "position": [3, 1], "requires_key": "red"},
            {"id": "d2", "position": [4, 1], "requires_key": "blue"},
        ]
    },
}


class _StateOnlyBackend(AbstractGridBackend):
    """Minimal backend exposing a fixed GridState (exercises the ABC default)."""

    def __init__(self, spec, open_doors):
        super().__init__()
        self.task_spec = spec
        self._configured = True
        self._open = set(open_doors)

    def configure(self, task_spec):
        self.task_spec = task_spec

    def reset(self, seed=None):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def get_mission_text(self):
        return ""

    def get_state(self):
        return GridState(agent_position=(1, 1), agent_direction=0, open_doors=set(self._open))


def test_default_door_states_derive_from_open_doors():
    backend = _StateOnlyBackend(TaskSpecification.from_dict(TWO_DOORS), open_doors={"d1"})
    assert backend.door_states() == {"d1": True, "d2": False}


def test_default_door_states_empty_when_unconfigured():
    backend = _StateOnlyBackend(None, open_doors=set())
    assert backend.door_states() == {}


def test_frame_is_grid_aligned_defaults_true():
    backend = _StateOnlyBackend(TaskSpecification.from_dict(TWO_DOORS), open_doors=set())
    assert backend.frame_is_grid_aligned is True


def test_minigrid_door_states_track_physical_open_close():
    pytest.importorskip("minigrid")
    from gridworld.backends.minigrid_backend import MiniGridBackend

    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(TaskSpecification.from_dict(CORRIDOR))
    backend.reset(seed=0)
    assert backend.door_states() == {"d1": False}

    for action in (A.MOVE_FORWARD, A.PICKUP, A.TOGGLE):
        backend.step(int(action))
    assert backend.door_states() == {"d1": True}

    # Re-close the unlocked door: physically closed, still "unlocked" for scoring.
    *_, state, _info = backend.step(int(A.TOGGLE))
    assert backend.door_states() == {"d1": False}
    assert "d1" in state.open_doors

    backend.step(int(A.TOGGLE))
    assert backend.door_states() == {"d1": True}
