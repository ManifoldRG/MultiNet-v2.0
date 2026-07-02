"""Condition set 6: in-context learning."""

from __future__ import annotations

from .core import ConditionSet, Variant


CONDITION_SET = ConditionSet(
    name="In-context learning",
    comparisons=(
        "Standard zero-shot: no examples",
        "1-shot: one example trajectory from a different maze",
    ),
    decision=(
        "If 1-shot dramatically improves performance, the bottleneck is likely "
        "task understanding rather than navigation capability."
    ),
    variants={
        "standard": Variant(
            name="zero_shot",
            description="No examples — same as the standard prompt.",
        ),
        "one_shot": Variant(
            name="one_shot",
            description=(
                "One example trajectory prepended to each user message "
                "(14x14 dense kr_sg_kb maze, separate from evaluation mazes)."
            ),
            config_overrides={"in_context_learning": "one_shot"},
        ),
    },
    notes="ICL examples must not use evaluation mazes.",
)
