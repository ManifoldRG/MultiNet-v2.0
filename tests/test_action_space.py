"""Cardinal action-space vocabulary and translation to egocentric primitives."""

from __future__ import annotations

import pytest

from interface.action_space import (
    CARDINAL_ACTIONS,
    EGOCENTRIC_ACTIONS,
    actions_hint,
    cardinal_to_primitives,
    synonyms,
    valid_actions,
)


def test_valid_actions_per_space():
    assert valid_actions("egocentric") == set(EGOCENTRIC_ACTIONS)
    assert valid_actions("cardinal") == set(CARDINAL_ACTIONS)


def test_cardinal_vocabulary_is_the_agreed_set():
    assert CARDINAL_ACTIONS == (
        "MOVE_NORTH",
        "MOVE_SOUTH",
        "MOVE_EAST",
        "MOVE_WEST",
        "PICKUP",
        "DROP",
        "INTERACT",
        "DONE",
    )


def test_actions_hint_lists_space_tokens_in_order():
    assert actions_hint("egocentric") == ", ".join(EGOCENTRIC_ACTIONS)
    assert actions_hint("cardinal") == ", ".join(CARDINAL_ACTIONS)


def test_synonyms_cover_cardinal_phrases_and_interact():
    cardinal = synonyms("cardinal")
    assert cardinal["move north"] == "MOVE_NORTH"
    assert cardinal["north"] == "MOVE_NORTH"
    assert cardinal["interact"] == "INTERACT"
    # Egocentric turn synonyms must not leak into the cardinal vocabulary.
    assert "turn left" not in cardinal


@pytest.mark.parametrize(
    "facing, token, expected",
    [
        # Already facing the target direction: just step forward.
        ("EAST", "MOVE_EAST", ["MOVE_FORWARD"]),
        ("NORTH", "MOVE_NORTH", ["MOVE_FORWARD"]),
        # One clockwise quarter-turn.
        ("EAST", "MOVE_SOUTH", ["TURN_RIGHT", "MOVE_FORWARD"]),
        ("NORTH", "MOVE_EAST", ["TURN_RIGHT", "MOVE_FORWARD"]),
        # One counter-clockwise quarter-turn (shorter than three rights).
        ("EAST", "MOVE_NORTH", ["TURN_LEFT", "MOVE_FORWARD"]),
        ("NORTH", "MOVE_WEST", ["TURN_LEFT", "MOVE_FORWARD"]),
        # Opposite direction: two turns then forward.
        ("EAST", "MOVE_WEST", ["TURN_RIGHT", "TURN_RIGHT", "MOVE_FORWARD"]),
        ("NORTH", "MOVE_SOUTH", ["TURN_RIGHT", "TURN_RIGHT", "MOVE_FORWARD"]),
    ],
)
def test_cardinal_move_expands_relative_to_facing(facing, token, expected):
    assert cardinal_to_primitives(token, facing) == expected


def test_interact_maps_to_toggle():
    assert cardinal_to_primitives("INTERACT", "NORTH") == ["TOGGLE"]


def test_pickup_and_done_pass_through():
    assert cardinal_to_primitives("PICKUP", "EAST") == ["PICKUP"]
    assert cardinal_to_primitives("DONE", "EAST") == ["DONE"]


def test_egocentric_primitives_are_idempotent():
    # Injected/expanded primitives must survive a second translation pass
    # unchanged so the runner can push them back onto the queue safely.
    for token in ("TURN_LEFT", "TURN_RIGHT", "MOVE_FORWARD", "TOGGLE"):
        assert cardinal_to_primitives(token, "NORTH") == [token]
