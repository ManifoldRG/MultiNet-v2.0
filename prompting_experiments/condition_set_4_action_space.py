"""Condition set 4: action space."""

from __future__ import annotations

from .core import ConditionSet, Variant


CONDITION_SET = ConditionSet(
    name="Action space",
    comparisons=(
        "Egocentric: TURN_LEFT, TURN_RIGHT, MOVE_FORWARD, PICKUP, TOGGLE, DONE",
        "Cardinal: MOVE_NORTH/SOUTH/EAST/WEST, PICKUP, INTERACT, DONE",
    ),
    decision=(
        "Determine whether an absolute (cardinal) move interface improves "
        "performance over the egocentric turn-and-move interface."
    ),
    variants={
        "egocentric": Variant(
            name="egocentric",
            description="Turn-and-move interface - same as the standard prompt (the default).",
        ),
        "cardinal": Variant(
            name="cardinal",
            description=(
                "Absolute moves (MOVE_NORTH/SOUTH/EAST/WEST) plus PICKUP, INTERACT, "
                "DONE; each move is expanded into egocentric primitives at runtime."
            ),
            config_overrides={"action_space": "cardinal"},
        ),
    },
)
