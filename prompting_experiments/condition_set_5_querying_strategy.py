"""Condition set 5: querying strategy."""

from __future__ import annotations

from .core import ConditionSet, Variant


CONDITION_SET = ConditionSet(
    name="Querying strategy",
    comparisons=(
        "Step-by-step: one action per query",
        "Subgoal planning: model outputs a subgoal and action chunk",
        "Full trajectory: model outputs a complete plan once",
    ),
    decision="Determine whether chunked or one-shot planning improves performance.",
    variants={
        "step_by_step": Variant(
            name="step_by_step",
            description="Ask for one action each query.",
            config_overrides={"querying": "step_by_step"},
        ),
        "subgoal": Variant(
            name="subgoal",
            description="Ask for a short subgoal and action chunk.",
            config_overrides={"querying": "subgoal"},
        ),
        "full_trajectory": Variant(
            name="full_trajectory",
            description="Ask once for a complete action trajectory.",
            config_overrides={"querying": "full_trajectory"},
        ),
    },
)
