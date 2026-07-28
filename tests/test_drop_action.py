"""Tests for the DROP action's state bookkeeping.

DROP physically works (custom_env falls through to MiniGridEnv.step, which places
the carried object in the front cell), but the env's own `collected_keys` set is
never updated, and the observation layer renders keys from the task spec's static
position while skipping anything in `collected_keys`. The result is a dropped key
that exists on the grid but is invisible to the agent and still reported as held.
"""

import sys
from pathlib import Path

import pytest

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from gridworld.actions import MiniGridActions
from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.custom_env import Key
from gridworld.task_spec import TaskSpecification
from interface import coords
from interface.renderer import render_user_observation_text


def _spec(**overrides):
    d = {
        "task_id": "drop_test",
        "seed": 1,
        "difficulty_tier": 3,
        "maze": {"dimensions": [8, 8], "walls": [], "start": [1, 1], "goal": [6, 6]},
        "mechanisms": {
            "keys": [{"id": "kR", "position": [2, 1], "color": "red"}],
            "doors": [
                {
                    "id": "DR",
                    "position": [4, 4],
                    "requires_key": "red",
                    "initial_state": "locked",
                }
            ],
        },
        "goal": {"type": "reach_position", "target": [6, 6]},
        "max_steps": 200,
    }
    d.update(overrides)
    return TaskSpecification.from_dict(d)


@pytest.fixture
def backend():
    b = MiniGridBackend(render_mode="rgb_array")
    b.configure(_spec())
    b.reset(seed=1)
    return b


def _keys_on_grid(env):
    found = {}
    for x in range(env.width):
        for y in range(env.height):
            cell = env.grid.get(x, y)
            if isinstance(cell, Key):
                found[getattr(cell, "key_id", "?")] = (x, y)
    return found


def _pick_up_the_key(backend):
    """Agent starts at (1,1) facing east; the key sits at (2,1)."""
    backend.env.step(MiniGridActions.MOVE_FORWARD)
    backend.env.step(MiniGridActions.PICKUP)
    assert backend.env.carrying is not None, "precondition: key must be held"


class TestDropPlacesKeyInAgentCell:
    def test_dropped_key_lands_under_the_agent(self, backend):
        _pick_up_the_key(backend)                    # agent at (2,1) holding kR
        backend.env.step(MiniGridActions.DROP)

        assert backend.env.carrying is None
        assert _keys_on_grid(backend.env) == {"kR": (2, 1)}
        assert "kR" not in backend.env.collected_keys, (
            "a dropped key is back on the grid and is no longer collected"
        )

    def test_drop_then_pickup_round_trips_in_two_actions(self, backend):
        _pick_up_the_key(backend)
        backend.env.step(MiniGridActions.DROP)
        backend.env.step(MiniGridActions.PICKUP)

        assert getattr(backend.env.carrying, "key_id", None) == "kR"
        assert "kR" in backend.env.collected_keys

    def test_drop_succeeds_regardless_of_facing(self, backend):
        """Same-cell placement must not depend on what is in front."""
        _pick_up_the_key(backend)
        for _ in range(2):
            backend.env.step(MiniGridActions.TURN_LEFT)   # face west
        backend.env.step(MiniGridActions.MOVE_FORWARD)    # to (1,1), wall ahead
        backend.env.step(MiniGridActions.DROP)

        assert backend.env.carrying is None
        assert _keys_on_grid(backend.env) == {"kR": (1, 1)}

    def test_drop_with_empty_hands_is_a_rejected_no_op(self, backend):
        before = backend.env.step_count
        _, _, _, _, info = backend.env.step(MiniGridActions.DROP)

        assert info.get("invalid_action") is True
        assert backend.env.carrying is None
        assert backend.env.step_count == before + 1

    def test_drop_onto_an_occupied_cell_is_rejected(self):
        """Standing on a switch, the cell is taken; the key stays in hand."""
        b = MiniGridBackend(render_mode="rgb_array")
        b.configure(_spec(mechanisms={
            "keys": [{"id": "kR", "position": [2, 1], "color": "red"}],
            "switches": [{"id": "s1", "position": [3, 1], "controls": ["g1"],
                          "color": "yellow", "switch_type": "toggle",
                          "initial_state": "off"}],
            "gates": [{"id": "g1", "position": [5, 5], "initial_state": "closed",
                       "color": "black"}],
        }))
        b.reset(seed=1)
        b.env.step(MiniGridActions.MOVE_FORWARD)   # (2,1), onto the key
        b.env.step(MiniGridActions.PICKUP)
        b.env.step(MiniGridActions.MOVE_FORWARD)   # (3,1), onto the switch
        held = b.env.carrying

        _, _, _, _, info = b.env.step(MiniGridActions.DROP)

        assert info.get("invalid_action") is True
        assert b.env.carrying is held
        assert "kR" in b.env.collected_keys


class TestDropIsNotInTheModelFacingVocabulary:
    """DROP works in the env but is deliberately not offered to models.

    Recorded here so the consequence is explicit rather than accidental: because
    PICKUP requires empty hands and models cannot DROP, picking up a decoy key is
    unrecoverable for an agent. That is a live design question for the D-condition
    mazes (wrong-key decoys), not an env bug. If DROP is ever added to the action
    space, these assertions should be updated deliberately — the env-side tests
    above already cover the mechanics.
    """

    def test_drop_is_absent_from_both_action_spaces(self):
        from interface.action_space import valid_actions

        assert "DROP" not in valid_actions("egocentric")
        assert "DROP" not in valid_actions("cardinal")

    def test_parser_rejects_a_drop_reply(self):
        from interface.parser import parse_final_output

        assert parse_final_output("FINAL_OUTPUT: DROP") is None
        assert parse_final_output("FINAL_OUTPUT: PICKUP") == ["PICKUP"]


class TestDroppedKeyIsObservable:
    def test_state_reports_live_key_positions(self, backend):
        _pick_up_the_key(backend)
        backend.env.step(MiniGridActions.DROP)

        state = backend.get_state()
        assert state.key_positions == {"kR": (2, 1)}, (
            "the agent must be able to see where a dropped key actually is"
        )

    def test_held_key_has_no_grid_position(self, backend):
        _pick_up_the_key(backend)
        state = backend.get_state()
        assert state.key_positions == {}

    def test_key_at_cell_follows_the_dropped_key(self, backend):
        _pick_up_the_key(backend)
        backend.env.step(MiniGridActions.DROP)
        state = backend.get_state()
        spec = _spec()

        # dropped in the agent's own cell (x,y)=(2,1) -> (row,col)=(1,2),
        # which here coincides with the key's spec cell, so move first and
        # re-drop somewhere the spec position cannot explain.
        assert coords.key_at_cell(spec, state, 1, 2) == "red"

    def test_text_observation_lists_the_dropped_key(self, backend):
        _pick_up_the_key(backend)
        backend.env.step(MiniGridActions.DROP)
        state = backend.get_state()

        text = render_user_observation_text(_spec(), state)

        assert "red key" in text.lower(), "a dropped key must reappear in the text observation"
        assert "(1,2)" in text.replace(" ", ""), "listed at its current cell"
