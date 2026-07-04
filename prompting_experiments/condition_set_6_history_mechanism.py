"""Condition set 7: history mechanism (single-message vs multi-turn chat).

Context window (set 3) tests the *amount* of history; this set isolates the
*mechanism*: rendering that history inside one stateless message (the fair
default) versus carrying it as a real multi-turn rolling conversation. The two
knobs are coupled here so we never double-count history (in-prompt + turns).
"""

from __future__ import annotations

from .core import ConditionSet, Variant


CONDITION_SET = ConditionSet(
    name="History mechanism",
    comparisons=(
        "Single-message: last3 history rendered in one stateless message",
        "Multi-turn: rolling 3-turn chat, one observation image per prior turn",
    ),
    decision=(
        "Isolate whether a real multi-turn chat structure beats cramming the "
        "same recent history into a single stateless message."
    ),
    variants={
        "single_message": Variant(
            name="single_message",
            description="History in one stateless message (last3) - same as the fair default.",
            config_overrides={"chat_history": "stateless", "context_window": "last3"},
        ),
        "multiturn": Variant(
            name="multiturn",
            description=(
                "Rolling 3-turn chat; each prior turn carries its own observation "
                "image (context_window=current so history comes from turns, not "
                "in-prompt, avoiding duplication)."
            ),
            config_overrides={
                "chat_history": "rolling",
                "context_window": "current",
                "chat_turns_max": 3,
            },
        ),
    },
)
