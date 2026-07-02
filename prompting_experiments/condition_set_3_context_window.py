"""Condition set 3: context window."""

from __future__ import annotations

from .core import ConditionSet, Variant


CONDITION_SET = ConditionSet(
    name="Context window",
    comparisons=(
        "Standard 0 history: current observation only",
        "Last 3 executed steps",
        "Current observation + text summary of prior actions",
    ),
    decision="Compare current-state-only prompting against recent history.",
    variants={
        "standard": Variant(
            name="current",
            description="Prompt only with the current observation-same as the standard prompt.",
        ),
        "last3": Variant(
            name="last3",
            description="Include up to the last three executed steps.",
            config_overrides={"context_window": "last3"},
        ),
        "text_summary": Variant(
            name="text_summary",
            description="One-sentence summary of all prior mechanism events or path waypoints.",
            config_overrides={"context_window": "text_summary"},
            preview_steps=10,
            preview_rollout_seed=5,
            preview_move_only=True,
        ),
    },
)
