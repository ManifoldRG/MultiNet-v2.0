"""Step feedback strings for the episode loop — independent of prompt strategy."""

from __future__ import annotations

from typing import Any, Literal

ObservationKind = Literal["text_only", "image_text", "image_only"]


def action_feedback_for_prompt(observation: ObservationKind, text: str) -> str:
    """Step outcomes for ``Last result:`` / history; cleared for ``image_only``. Parse failures stay visible in runner."""
    if observation == "image_only":
        return ""
    return text


def format_step_feedback(
    action: str, event_type: str, event_message: str, prev_pos: Any
) -> str:
    """Format env step for ``Last result:`` (branches match ``StepEvent.type`` in ``env``)."""
    if event_type == "BLOCKED":
        return f"BLOCKED — {action}: {event_message} You remain at {prev_pos}."
    if event_type == "TURNED":
        return f"TURNED — {action}: {event_message}"
    if event_type == "MOVED":
        return f"MOVED — {action}: {event_message}"
    if event_type == "DONE":
        return f"SUCCESS — {action}: {event_message}"
    if event_type == "PICKUP":
        return f"PICKUP — {action}: {event_message}"
    if event_type == "NOTHING":
        return f"NOTHING — {action}: {event_message} You remain at {prev_pos}."
    if event_type == "TOGGLED":
        return f"TOGGLED — {action}: {event_message}"
    if event_type == "WRONG_DONE":
        return f"WRONG DONE — {action}: {event_message} You remain at {prev_pos}."
    if event_type == "INVALID":
        return f"INVALID — {action}: {event_message} You remain at {prev_pos}."
    return f"{event_type} — {action}: {event_message}"
