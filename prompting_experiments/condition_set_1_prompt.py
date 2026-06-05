"""Condition set 1: prompt verbosity."""

from __future__ import annotations

from .core import ConditionSet, Variant


CONDITION_SET = ConditionSet(
    name="Prompt",
    comparisons=(
        "Standard: goal + mechanism descriptions + action list",
        "Verbose: standard + explicit rules",
    ),
    decision="If delta < 5%, use standard. If > 5%, use verbose.",
    variants={
        "standard": Variant(
            name="standard",
            description="Standard task prompt with mechanism descriptions.",
            config_overrides={"prompting": "standard"},
        ),
        "verbose": Variant(
            name="verbose",
            description="Standard prompt plus explicit domain rules.",
            config_overrides={"prompting": "verbose"},
        ),
    },
)
