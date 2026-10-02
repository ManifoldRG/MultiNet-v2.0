"""Step feedback strings for the episode loop."""

from __future__ import annotations

from typing import Literal

from gridworld.backends.base import GridState
from gridworld.task_spec import TaskSpecification

from interface.coords import (
    agent_facing,
    agent_row_col,
    compact_ids,
    forward_cell,
    gate_at_cell,
    goal_row_col,
    switch_at_cell,
    switches_controlling_gate,
    door_at_cell,
)
from prompting_experiments.prompt_templates import feedback as feedback_templates


def infer_step_outcome(
    action: str,
    prev: GridState,
    curr: GridState,
    reward: float,
    terminated: bool,
    task_spec: TaskSpecification,
    level: Literal["minimal", "standard", "causal"] = "minimal",
) -> tuple[str, str]:
    goal = goal_row_col(task_spec)
    prev_pos = agent_row_col(prev)
    curr_pos = agent_row_col(curr)
    newly_open = curr.open_doors - prev.open_doors

    if newly_open:
        door_id = sorted(newly_open)[0]
        door = next((d for d in task_spec.mechanisms.doors if d.id == door_id), None)
        color = door.requires_key if door else "matching"
        if action == "MOVE_FORWARD" and prev_pos != curr_pos:
            return "OPENED", feedback_templates.OPENED_AND_MOVED.format(
                color=color,
                door_id=door_id,
            )
        return "OPENED", feedback_templates.OPENED_DOOR.format(color=color, door_id=door_id)

    if action in ("TURN_LEFT", "TURN_RIGHT"):
        if prev.agent_direction != curr.agent_direction:
            return "TURNED", feedback_templates.NOW_FACING.format(facing=agent_facing(curr))
        return "NOTHING", feedback_templates.ACTION_NO_EFFECT.format(action=action)

    if action == "MOVE_FORWARD":
        if prev_pos == curr_pos:
            fwd = forward_cell(prev)
            gate = gate_at_cell(task_spec, prev, fwd[0], fwd[1])
            if gate and not gate["open"]:
                gates, switches = compact_ids(task_spec)
                gate_id = gates[str(gate["id"])]
                controllers = [
                    switches[sid]
                    for sid in switches_controlling_gate(task_spec, str(gate["id"]))
                ]
                if controllers and level == "causal":
                    return (
                        "BLOCKED",
                        feedback_templates.MOVE_BLOCKED_BY_GATE_WITH_SWITCHES.format(
                            gate_id=gate_id,
                            switches=", ".join(controllers),
                        ),
                    )
                return (
                    "BLOCKED",
                    feedback_templates.MOVE_BLOCKED_BY_GATE.format(
                        gate_id=gate_id,
                    ),
                )
            door = door_at_cell(task_spec, prev, fwd[0], fwd[1])
            if door and not door["open"]:
                color = door["requires_key"]
                msg = feedback_templates.MOVE_BLOCKED_BY_DOOR.format(color=color)
                if level == "causal":
                    hint = (
                        feedback_templates.DOOR_USE_TOGGLE
                        if prev.agent_carrying == color
                        else feedback_templates.DOOR_NEEDS_KEY
                    )
                    msg += " " + hint.format(color=color)
                return "BLOCKED", msg
            return "BLOCKED", feedback_templates.MOVE_BLOCKED_GENERIC
        if terminated and reward > 0 and curr_pos == goal:
            return "DONE", feedback_templates.REACHED_GOAL
        return "MOVED", feedback_templates.MOVED_TO

    if action == "PICKUP":
        if (
            prev.agent_carrying != curr.agent_carrying
            or len(curr.collected_keys) > len(prev.collected_keys)
        ):
            carried = curr.agent_carrying or "a"
            return "PICKUP", feedback_templates.PICKED_UP_KEY.format(key_color=carried)
        return "NOTHING", feedback_templates.NOTHING_TO_PICK_UP

    if action == "DROP":
        if prev.agent_carrying and not curr.agent_carrying:
            return "DROP", feedback_templates.DROPPED_KEY.format(
                key_color=prev.agent_carrying,
            )
        if not prev.agent_carrying:
            return "NOTHING", feedback_templates.NOTHING_TO_DROP
        tmpl = (
            feedback_templates.DROP_BLOCKED
            if level == "causal"
            else feedback_templates.DROP_BLOCKED_STANDARD
        )
        return "NOTHING", tmpl

    if action == "TOGGLE":
        if (
            prev.active_switches != curr.active_switches
            or prev.open_gates != curr.open_gates
        ):
            return "TOGGLED", feedback_templates.TOGGLED_STATE_CHANGED
        fwd = forward_cell(prev)
        switch_ahead = switch_at_cell(task_spec, fwd[0], fwd[1])
        switch_here = switch_at_cell(task_spec, prev_pos[0], prev_pos[1])
        gate_ahead = gate_at_cell(task_spec, prev, fwd[0], fwd[1])
        door_ahead = door_at_cell(task_spec, prev, fwd[0], fwd[1])
        if switch_ahead and not switch_here:
            if level != "causal":
                return "NOTHING", feedback_templates.TOGGLE_NO_EFFECT_STANDARD
            if switch_ahead["switch_type"] == "hold":
                return (
                    "NOTHING",
                    feedback_templates.TOGGLE_HOLD_SWITCH_HINT,
                )
            return (
                "NOTHING",
                feedback_templates.TOGGLE_SWITCH_HINT,
            )
        if gate_ahead and not gate_ahead["open"]:
            msg = feedback_templates.GATE_TOGGLE_STANDARD
            if level == "causal":
                msg += " " + feedback_templates.GATE_TOGGLE_HINT
            return "NOTHING", msg
        if door_ahead and not door_ahead["open"]:
            msg = feedback_templates.DOOR_TOGGLE_STANDARD
            if level == "causal":
                msg += " " + feedback_templates.DOOR_TOGGLE_HINT
            return "NOTHING", msg
        return (
            "NOTHING",
            feedback_templates.TOGGLE_NO_EFFECT if level == "causal"
            else feedback_templates.TOGGLE_NO_EFFECT_STANDARD,
        )

    if action == "DONE":
        if terminated and reward > 0 and curr_pos == goal:
            return "DONE", feedback_templates.TASK_COMPLETE
        return "WRONG_DONE", feedback_templates.WRONG_DONE

    return "INVALID", feedback_templates.UNKNOWN_ACTION.format(action=action)


def format_step_feedback(
    action: str,
    prev: GridState,
    curr: GridState,
    reward: float,
    terminated: bool,
    task_spec: TaskSpecification,
    level: Literal["minimal", "standard", "causal"] = "minimal",
) -> tuple[str, str]:
    event_type, event_message = infer_step_outcome(
        action, prev, curr, reward, terminated, task_spec, level=level
    )
    if level == "minimal":
        return event_type, event_type
    if event_type == "BLOCKED":
        return feedback_templates.BLOCKED_FEEDBACK.format(action=action, message=event_message), event_type
    if event_type == "TURNED":
        return feedback_templates.TURNED_FEEDBACK.format(action=action, message=event_message), event_type
    if event_type == "MOVED":
        return feedback_templates.MOVED_FEEDBACK.format(action=action, message=event_message), event_type
    if event_type == "DONE":
        return feedback_templates.SUCCESS_FEEDBACK.format(action=action, message=event_message), event_type
    if event_type == "PICKUP":
        return feedback_templates.PICKUP_FEEDBACK.format(action=action, message=event_message), event_type
    if event_type == "DROP":
        return feedback_templates.DROP_FEEDBACK.format(action=action, message=event_message), event_type
    if event_type == "NOTHING":
        return feedback_templates.NOTHING_FEEDBACK.format(action=action, message=event_message), event_type
    if event_type == "OPENED":
        return feedback_templates.OPENED_FEEDBACK.format(action=action, message=event_message), event_type
    if event_type == "TOGGLED":
        return feedback_templates.TOGGLED_FEEDBACK.format(action=action, message=event_message), event_type
    if event_type == "WRONG_DONE":
        return feedback_templates.WRONG_DONE_FEEDBACK.format(action=action, message=event_message), event_type
    if event_type == "INVALID":
        return feedback_templates.INVALID_FEEDBACK.format(action=action, message=event_message), event_type
    return feedback_templates.DEFAULT_FEEDBACK.format(
        event_type=event_type,
        action=action,
        message=event_message,
    ), event_type
