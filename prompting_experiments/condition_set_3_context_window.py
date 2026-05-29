"""Condition set 3: context window."""

from __future__ import annotations

from .core import ConditionSet, Variant


CONDITION_SET = ConditionSet(
    name="Context window",
    comparisons=(
        "0 history: current observation only",
        "Last 3 executed steps",
        "Current observation + text summary of prior actions",
    ),
    decision="Compare current-state-only prompting against recent history.",
    variants={
        "current": Variant(
            name="current",
            description="Prompt only with the current observation.",
            config_overrides={"context_window": "current"},
        ),
        "last3": Variant(
            name="last3",
            description="Include up to the last three executed steps.",
            config_overrides={"context_window": "last3"},
        ),
        "text_summary": Variant(
            name="text_summary",
            description="PR #12 design axis; no ExperimentConfig summary mode exists yet.",
            implemented=False,
        ),
    },
)
