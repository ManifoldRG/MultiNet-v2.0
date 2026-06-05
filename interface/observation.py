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

ObservationMode = Literal["text_only", "image_text", "image_only"]
ContextWindow = Literal["current", "last3"]


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
