"""Observation formatting for the NLU interface.

History when ``context_window == "last3"`` (last 3 executed steps, oldest first):

* **text_only** — full text history only (position, facing, action, feedback).
* **image_only** — prior decision-frame PNGs + inventory/action labels (no text history).
* **image_text** — full text history **and** prior decision-frame PNGs.

History is derived from enriched ``transcript`` step records.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from gridworld.backends.base import GridState
from gridworld.task_spec import TaskSpecification

from interface.renderer import (
    render_current_inventory_text,
    render_user_observation_text,
    rgb_to_image_block,
)
from prompting_experiments.prompt_templates import observation as observation_templates
from prompting_experiments.prompt_templates import user as user_templates

ObservationMode = Literal["text_only", "image_text", "image_only"]
ContextWindow = Literal["current", "last3", "text_summary"]


def history_steps(transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        rec
        for rec in transcript
        if rec.get("kind") == "step" and rec.get("event_type") != "INVALID"
    ]


def recent_history_steps(
    transcript: list[dict[str, Any]], context_window: ContextWindow
) -> list[dict[str, Any]]:
    if context_window != "last3":
        return []
    return history_steps(transcript)[-3:]


def history_text(
    observation: ObservationMode,
    context_window: ContextWindow,
    transcript: list[dict[str, Any]],
) -> str:
    if context_window == "text_summary":
        return text_summary_history(transcript)
    if observation not in ("text_only", "image_text"):
        return ""
    recs = recent_history_steps(transcript, context_window)
    if not recs:
        return ""

    lines = [observation_templates.RECENT_HISTORY_HEADER]
    for rec in recs:
        row, col = rec["position_after"]
        lines.append(
            observation_templates.RECENT_HISTORY_STEP.format(
                row=int(row),
                col=int(col),
                facing=rec["facing_after"],
                action=rec["action"],
                feedback=rec["prompt_feedback"],
            )
        )
    return "\n".join(lines)


def text_summary_history(transcript: list[dict[str, Any]]) -> str:
    """Build a one-sentence summary of all prior mechanism events or path waypoints."""
    steps = history_steps(transcript)
    mechanism_events = _extract_mechanism_events(steps)

    if mechanism_events:
        summary = _format_summary_chain(mechanism_events)
    else:
        move_steps = [rec for rec in steps if rec.get("event_type") == "MOVED"]
        if move_steps:
            waypoints = _pick_waypoints(move_steps, 3)
            nav_parts: list[str] = []
            for i, (row, col) in enumerate(waypoints):
                if i == len(waypoints) - 1:
                    nav_parts.append(user_templates.TEXT_SUMMARY_PASSED.format(row=row, col=col))
                else:
                    nav_parts.append(user_templates.TEXT_SUMMARY_NAV_TO.format(row=row, col=col))
            summary = _format_summary_chain(nav_parts)
        else:
            return user_templates.TEXT_SUMMARY_EMPTY

    return f"{user_templates.TEXT_SUMMARY_BLOCK_HEADER}\n{summary}"


def _extract_mechanism_events(steps: list[dict[str, Any]]) -> list[str]:
    events: list[str] = []
    for rec in steps:
        event_type = rec.get("event_type", "")
        sb = rec.get("state_before") or {}
        sa = rec.get("state_after") or {}

        if event_type == "PICKUP":
            before_keys = set(sb.get("collected_keys") or [])
            after_keys = set(sa.get("collected_keys") or [])
            new_keys = after_keys - before_keys
            if new_keys:
                key_id = sorted(new_keys)[0]
            else:
                key_id = sa.get("agent_carrying") or sb.get("agent_carrying") or "a"
            events.append(user_templates.TEXT_SUMMARY_PICKUP_KEY.format(key_id=key_id))

        elif event_type == "OPENED":
            before_doors = set(sb.get("open_doors") or [])
            after_doors = set(sa.get("open_doors") or [])
            new_doors = after_doors - before_doors
            door_id = sorted(new_doors)[0] if new_doors else "a"
            events.append(user_templates.TEXT_SUMMARY_OPEN_DOOR.format(door_id=door_id))

        elif event_type == "TOGGLED":
            before_gates = set(sb.get("open_gates") or [])
            after_gates = set(sa.get("open_gates") or [])
            opened = after_gates - before_gates
            closed = before_gates - after_gates
            if opened:
                events.append(user_templates.TEXT_SUMMARY_OPEN_GATE.format(gate_id=sorted(opened)[0]))
            elif closed:
                events.append(user_templates.TEXT_SUMMARY_CLOSE_GATE.format(gate_id=sorted(closed)[0]))

    return events


def _format_summary_chain(events: list[str]) -> str:
    if not events:
        return ""
    if len(events) == 1:
        return f"first you {events[0]}"
    parts = [f"first you {events[0]}"]
    parts.extend(f"then you {e}" for e in events[1:-1])
    parts.append(f"finally you {events[-1]}")
    return ", ".join(parts)


def _pick_waypoints(steps: list[dict[str, Any]], count: int) -> list[tuple[int, int]]:
    n = len(steps)
    if n <= count:
        return [tuple(rec["position_after"]) for rec in steps]  # type: ignore[return-value]
    indices = [round(i * (n - 1) / (count - 1)) for i in range(count)]
    seen: list[tuple[int, int]] = []
    for i in indices:
        pos: tuple[int, int] = tuple(steps[i]["position_after"])  # type: ignore[assignment]
        if pos not in seen:
            seen.append(pos)
    return seen


def history_content_blocks(
    observation: ObservationMode,
    context_window: ContextWindow,
    transcript: list[dict[str, Any]],
) -> list[dict]:
    if observation not in ("image_only", "image_text"):
        return []
    recs = recent_history_steps(transcript, context_window)
    if not recs:
        return []

    blocks: list[dict] = []
    for rec in recs:
        rgb = rec.get("_decision_frame_rgb")
        if rgb is None:
            continue
        blocks.append(rgb_to_image_block(rgb))
        inventory = _history_record_inventory(rec)
        text = (
            observation_templates.IMAGE_HISTORY_INVENTORY_ACTION.format(
                inventory=inventory,
                action=rec["action"],
            )
            if observation == "image_only"
            else observation_templates.IMAGE_HISTORY_INVENTORY.format(inventory=inventory)
        )
        blocks.append({"type": "text", "text": text})

    if not blocks:
        return []

    intro = (
        observation_templates.IMAGE_ONLY_HISTORY_INTRO
        if observation == "image_only"
        else observation_templates.IMAGE_TEXT_HISTORY_INTRO
    )
    return [{"type": "text", "text": intro}] + blocks


def current_observation_text(
    observation: ObservationMode,
    task_spec: TaskSpecification,
    state: GridState,
    *,
    include_description: bool = False,
    include_facing: bool = False,
) -> str:
    if observation == "image_only":
        return render_current_inventory_text(state)
    if not include_description:
        return ""
    return render_user_observation_text(task_spec, state, include_facing=include_facing)


def current_image_blocks(observation: ObservationMode, rgb: np.ndarray | None) -> list[dict]:
    if observation == "text_only" or rgb is None:
        return []
    return [rgb_to_image_block(rgb)]


def _history_record_inventory(rec: dict[str, Any]) -> str:
    state_before = rec.get("state_before")
    if isinstance(state_before, dict):
        inventory = state_before.get("inventory")
        if isinstance(inventory, list):
            return ", ".join(str(item) for item in inventory) or "empty"

    state_after = rec.get("state_after")
    if isinstance(state_after, dict):
        inventory = state_after.get("inventory")
        if isinstance(inventory, list):
            return ", ".join(str(item) for item in inventory) or "empty"

    return "unknown"
