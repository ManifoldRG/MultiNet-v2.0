"""Condition set 4: action space."""

from __future__ import annotations

from .core import ConditionSet, Variant


CONDITION_SET = ConditionSet(
    name="Action space",
    comparisons=(
        "Egocentric: TURN_LEFT, TURN_RIGHT, MOVE_FORWARD, PICKUP, TOGGLE, DONE",
        "Cardinal: MOVE_NORTH/SOUTH/EAST/WEST plus interaction action",
    ),
    decision="If cardinal actions materially improve performance, add a runtime action-space axis.",
    variants={
        "egocentric": Variant(
            name="egocentric",
            description="Current interface action space.",
            config_overrides={},
        ),
        "cardinal": Variant(
            name="cardinal",
            description="PR #12 design axis; no cardinal action map exists in the interface yet.",
            implemented=False,
        ),
    },
    implemented=False,
    notes="The current runner only supports the egocentric NLU action map.",
)
