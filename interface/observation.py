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
    render_user_observation_text,
    rgb_to_image_block,
)
from prompting_experiments.prompt_templates import observation as observation_templates
from prompting_experiments.prompt_templates import user as user_templates

ObservationMode = Literal["text_only", "image_text", "image_only"]
ContextWindow = Literal["current", "last3", "text_summary", "text_summary_and_last3"]


def history_steps(transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        rec
        for rec in transcript
        if rec.get("kind") == "step" and rec.get("event_type") != "INVALID"
    ]


def recent_history_steps(
    transcript: list[dict[str, Any]], context_window: ContextWindow
) -> list[dict[str, Any]]:
    if context_window not in ("last3", "text_summary_and_last3"):
        return []
    return history_steps(transcript)[-3:]


def history_text(
    observation: ObservationMode,
    context_window: ContextWindow,
    transcript: list[dict[str, Any]],
    task_spec: TaskSpecification | None = None,
) -> str:
    if context_window == "text_summary":
        return text_summary_history(transcript, task_spec)
    if context_window == "text_summary_and_last3":
        if observation not in ("text_only", "image_text"):
            # image_only: the summary is emitted separately (see
            # leading_summary_blocks) so it can precede the last3 image
            # blocks in the message content list.
            return ""
        recent_text = _last3_history_text(context_window, transcript)
        if observation == "image_text":
            # Same reason: the summary precedes the last3 image blocks, so
            # it is supplied by leading_summary_blocks instead of here.
            return recent_text
        summary = text_summary_history(transcript, task_spec)
        if not recent_text:
            return summary
        return f"{summary}\n\n{recent_text}"
    if observation not in ("text_only", "image_text"):
        return ""
    return _last3_history_text(context_window, transcript)


def leading_summary_blocks(
    observation: ObservationMode,
    context_window: ContextWindow,
    transcript: list[dict[str, Any]],
    task_spec: TaskSpecification | None = None,
) -> list[dict]:
    """Text-summary content block that must precede the last3 image blocks.

    For ``text_summary_and_last3`` with an image-bearing observation mode,
    the last3 steps are rendered as separate image blocks (see
    ``history_content_blocks``) rather than folded into the main prompt
    text, so the summary has to be surfaced here to keep summary-then-last3
    ordering in the final message content list.
    """
    if context_window != "text_summary_and_last3" or observation not in (
        "image_only",
        "image_text",
    ):
        return []
    summary = text_summary_history(transcript, task_spec)
    if not summary:
        return []
    return [{"type": "text", "text": summary}]


def _history_record_action(rec: dict[str, Any]) -> str:
    """The action to attribute to a history step in the model's own vocabulary.

    Cardinal runs expand one model action into primitives; each step record
    keeps the primitive in ``action`` and the model's emission in
    ``cardinal_action`` (None for egocentric runs). History renders under the
    ``FINAL_OUTPUT:`` delimiter, so it must show an action the model is
    actually allowed to output.
    """
    return rec.get("cardinal_action") or rec["action"]


def _last3_history_text(
    context_window: ContextWindow,
    transcript: list[dict[str, Any]],
) -> str:
    recs = recent_history_steps(transcript, context_window)
    if not recs:
        return ""

    lines = [observation_templates.RECENT_HISTORY_HEADER]
    for rec in recs:
        row, col = rec["position_after_row_col"]
        lines.append(
            observation_templates.RECENT_HISTORY_STEP.format(
                row=int(row),
                col=int(col),
                facing=rec["facing_after"],
                action=_history_record_action(rec),
                feedback=rec["prompt_feedback"],
            )
        )
    return "\n".join(lines)


def _agent_start_pose(
    transcript: list[dict[str, Any]],
) -> tuple[int, int, str] | None:
    """(row, col, facing) of the agent's starting cell, in prompt coordinates.

    Read from the transcript's ``reset`` record (``state.position_row_col`` +
    ``state.facing``); falls back to the first step's before-pose. Returns None
    if neither is available (older/partial transcripts) so the caller can omit
    the grounding line rather than crash.
    """
    for rec in transcript:
        if rec.get("kind") == "reset":
            st = rec.get("state") or {}
            rc = st.get("position_row_col")
            facing = st.get("facing")
            if rc and facing:
                return int(rc[0]), int(rc[1]), str(facing)
            break
    for rec in transcript:
        if rec.get("kind") == "step":
            pb = rec.get("position_before_row_col")
            fb = rec.get("facing_before")
            if pb and fb:
                return int(pb[0]), int(pb[1]), str(fb)
            break
    return None


def text_summary_history(
    transcript: list[dict[str, Any]],
    task_spec: TaskSpecification | None = None,
) -> str:
    """Summarize prior mechanism events plus the movement trail since the last one.

    The trail is essential: an events-only summary carried zero spatial
    information from the first pickup onward, exactly when a keyed maze turns
    back into a navigation problem (29/45 sweep episodes).

    The summary is prefixed with a persistent start-pose line ("You started at
    (r, c) facing DIR.") so the model has a fixed coordinate anchor to reason
    the trail against — important under image_only, where the observation gives
    no textual position and the model otherwise loses track of where it began.
    """
    steps = history_steps(transcript)
    mechanism_events = _extract_mechanism_events(steps, task_spec)
    parts = [text for _, text in mechanism_events]

    trail_start = mechanism_events[-1][0] + 1 if mechanism_events else 0
    move_steps = [rec for rec in steps[trail_start:] if rec.get("event_type") == "MOVED"]
    if move_steps:
        waypoints = _pick_waypoints(move_steps, 3)
        for i, (row, col) in enumerate(waypoints):
            if i == len(waypoints) - 1:
                parts.append(
                    observation_templates.TEXT_SUMMARY_PASSED.format(row=row, col=col)
                )
            else:
                parts.append(
                    observation_templates.TEXT_SUMMARY_NAV_TO.format(row=row, col=col)
                )

    start = _agent_start_pose(transcript)
    start_line = (
        observation_templates.TEXT_SUMMARY_START.format(
            row=start[0], col=start[1], facing=start[2]
        )
        if start
        else None
    )

    if not parts:
        body = observation_templates.TEXT_SUMMARY_EMPTY
    else:
        body = (
            f"{observation_templates.TEXT_SUMMARY_BLOCK_HEADER}\n"
            f"{_format_summary_chain(parts)}"
        )
    return f"{start_line}\n{body}" if start_line else body


def _extract_mechanism_events(
    steps: list[dict[str, Any]],
    task_spec: TaskSpecification | None = None,
) -> list[tuple[int, str]]:
    """Return (index-into-steps, event text) pairs, in step order."""
    events: list[tuple[int, str]] = []
    key_colors = {
        key.id: key.color for key in task_spec.mechanisms.keys
    } if task_spec else {}
    door_colors = {
        door.id: door.requires_key for door in task_spec.mechanisms.doors
    } if task_spec else {}
    gate_colors = {
        gate.id: getattr(gate, "color", "grey") for gate in task_spec.mechanisms.gates
    } if task_spec else {}
    for index, rec in enumerate(steps):
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
            events.append(
                (
                    index,
                    observation_templates.TEXT_SUMMARY_PICKUP_KEY.format(
                        key_color=key_colors.get(key_id, sa.get("agent_carrying") or key_id)
                    ),
                )
            )

        elif event_type == "DROP":
            # The dropped key is the one that leaves collected_keys (mirror of
            # the PICKUP diff above); its colour comes from the spec map. Fall
            # back to the colour held beforehand (agent_carrying is a colour,
            # not a key id), then to colourless phrasing.
            before_keys = set(sb.get("collected_keys") or [])
            after_keys = set(sa.get("collected_keys") or [])
            gone_keys = before_keys - after_keys
            if gone_keys:
                key_id = sorted(gone_keys)[0]
                key_color = key_colors.get(key_id, sb.get("agent_carrying") or key_id)
            else:
                key_color = sb.get("agent_carrying") or sa.get("agent_carrying")
            events.append(
                (
                    index,
                    observation_templates.TEXT_SUMMARY_DROP_KEY.format(key_color=key_color)
                    if key_color
                    else observation_templates.TEXT_SUMMARY_DROP_KEY_UNKNOWN,
                )
            )

        elif event_type == "OPENED":
            before_doors = set(sb.get("open_doors") or [])
            after_doors = set(sa.get("open_doors") or [])
            new_doors = after_doors - before_doors
            door_id = sorted(new_doors)[0] if new_doors else "a"
            events.append(
                (
                    index,
                    observation_templates.TEXT_SUMMARY_OPEN_DOOR.format(
                        door_color=door_colors.get(door_id, door_id)
                    ),
                )
            )

        elif event_type == "TOGGLED":
            before_gates = set(sb.get("open_gates") or [])
            after_gates = set(sa.get("open_gates") or [])
            opened = after_gates - before_gates
            closed = before_gates - after_gates
            if opened:
                events.append(
                    (
                        index,
                        observation_templates.TEXT_SUMMARY_OPEN_GATE.format(
                            gate_color=gate_colors.get(sorted(opened)[0], sorted(opened)[0])
                        ),
                    )
                )
            elif closed:
                events.append(
                    (
                        index,
                        observation_templates.TEXT_SUMMARY_CLOSE_GATE.format(
                            gate_color=gate_colors.get(sorted(closed)[0], sorted(closed)[0])
                        ),
                    )
                )

    return events


def _format_summary_chain(events: list[str]) -> str:
    if not events:
        return ""
    if len(events) == 1:
        return observation_templates.TEXT_SUMMARY_FIRST_EVENT.format(event=events[0])
    parts = [observation_templates.TEXT_SUMMARY_FIRST_EVENT.format(event=events[0])]
    parts.extend(
        observation_templates.TEXT_SUMMARY_THEN_EVENT.format(event=e)
        for e in events[1:-1]
    )
    parts.append(
        observation_templates.TEXT_SUMMARY_FINAL_EVENT.format(event=events[-1])
    )
    return ", ".join(parts)


def _pick_waypoints(steps: list[dict[str, Any]], count: int) -> list[tuple[int, int]]:
    n = len(steps)
    if n <= count:
        return [tuple(rec["position_after_row_col"]) for rec in steps]  # type: ignore[return-value]
    indices = [round(i * (n - 1) / (count - 1)) for i in range(count)]
    seen: list[tuple[int, int]] = []
    for i in indices:
        pos: tuple[int, int] = tuple(steps[i]["position_after_row_col"])  # type: ignore[assignment]
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
            user_templates.LAST3_USER_PROMPT["image_only_step"].format(
                inventory=inventory,
                action=_history_record_action(rec),
            )
            if observation == "image_only"
            else user_templates.LAST3_USER_PROMPT["image_text_step"].format(
                inventory=inventory,
                action=_history_record_action(rec),
            )
        )
        blocks.append({"type": "text", "text": text})

    if not blocks:
        return []

    return [{"type": "text", "text": user_templates.LAST3_USER_PROMPT["header"]}] + blocks


def current_observation_text(
    observation: ObservationMode,
    task_spec: TaskSpecification,
    state: GridState,
    *,
    include_description: bool = False,
    include_facing: bool = False,
) -> str:
    if observation == "image_only":
        return ""
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
