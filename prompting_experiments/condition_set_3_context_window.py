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
            description="Current observation only, no history (the 0-history ablation of the last3 default).",
            config_overrides={"context_window": "current", "chat_history": "stateless"},
        ),
        "last3": Variant(
            name="last3",
            description="Last three executed steps rendered in one stateless message - same as the fair default.",
            config_overrides={"context_window": "last3", "chat_history": "stateless"},
        ),
        "text_summary": Variant(
            name="text_summary",
            description="One-sentence summary of all prior mechanism events/path waypoints, in one stateless message.",
            config_overrides={"context_window": "text_summary", "chat_history": "stateless"},
            preview_steps=10,
            preview_rollout_seed=5,
            preview_move_only=True,
        ),
        "text_summary_and_last3": Variant(
            name="text_summary_and_last3",
            description="One-sentence summary of all prior mechanism events/path waypoints, in one stateless message, and last three executed steps rendered as 3 images-each with the action taken in that step.",
            config_overrides={"context_window": "text_summary_and_last3", "chat_history": "stateless"},
            preview_steps=10,
            preview_rollout_seed=5,
            preview_move_only=True,
        ),
    },
)
