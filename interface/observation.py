"""Observation formatting for the NLU interface.

History when ``context_window == "last3"`` (last 3 executed steps, oldest first):

* **text_only** — full text history only (position, facing, action, feedback).
* **image_only** — prior decision-frame PNGs + ``Action: …`` labels (no text history).
* **image_text** — full text history **and** prior decision-frame PNGs.

History is derived from enriched ``transcript`` step records.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from gridworld.backends.base import GridState
from gridworld.task_spec import TaskSpecification

from interface.renderer import render_user_observation_text, rgb_to_image_block

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

    lines = ["Recent history (last 3 steps, oldest first):"]
    for rec in recs:
        row, col = rec["position_after"]
        lines.append(
            f"  ({int(row)}, {int(col)}) facing {rec['facing_after']} -> {rec['action']} -> {rec['prompt_feedback']}"
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
        if observation == "image_only":
            blocks.append({"type": "text", "text": f"Action: {rec['action']}\n\n"})

    if not blocks:
        return []

    intro = (
        "Recent steps (oldest first). Each image is the maze view from which the "
        "following action was chosen; infer pose and environment state from the image.\n\n"
        if observation == "image_only"
        else "Recent step views (oldest first):\n\n"
    )
    return [{"type": "text", "text": intro}] + blocks


def current_observation_text(
    observation: ObservationMode,
    task_spec: TaskSpecification,
    state: GridState,
) -> str:
    if observation == "image_only":
        return ""
    return render_user_observation_text(task_spec, state)


def current_image_blocks(observation: ObservationMode, rgb: np.ndarray | None) -> list[dict]:
    if observation == "text_only" or rgb is None:
        return []
    return [rgb_to_image_block(rgb)]
