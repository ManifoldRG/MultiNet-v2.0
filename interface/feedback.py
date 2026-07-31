"""Step feedback strings for the episode loop."""

from __future__ import annotations

from gridworld.backends.base import GridState
from gridworld.task_spec import TaskSpecification

from interface.coords import (
    agent_facing,
    agent_row_col,
    forward_cell,
    gate_at_cell,
    goal_row_col,
    key_at_cell,
    switch_at_cell,
    switches_controlling_gate,
)
from prompting_experiments.prompt_templates import feedback as feedback_templates


def infer_step_outcome(
    action: str,
    prev: GridState,
    curr: GridState,
    reward: float,
    terminated: bool,
    task_spec: TaskSpecification,
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
                position=curr_pos,
            )
        return "OPENED", feedback_templates.OPENED_DOOR.format(color=color, door_id=door_id)

    if action in ("TURN_LEFT", "TURN_RIGHT"):
        if prev.agent_direction != curr.agent_direction:
            return "TURNED", feedback_templates.NOW_FACING.format(facing=agent_facing(curr))
        return "NOTHING", feedback_templates.ACTION_NO_EFFECT.format(action=action)

    if action == "MOVE_FORWARD":
        if prev_pos == curr_pos:
            fwd = forward_cell(prev)
            key_color = key_at_cell(task_spec, prev, fwd[0], fwd[1])
            if key_color:
                return (
                    "BLOCKED",
                    feedback_templates.MOVE_BLOCKED_BY_KEY.format(
                        key_color=key_color,
                        position=fwd,
                    ),
                )
            gate = gate_at_cell(task_spec, prev, fwd[0], fwd[1])
            if gate and not gate["open"]:
                controllers = switches_controlling_gate(task_spec, str(gate["id"]))
                if controllers:
                    switch_list = ", ".join(controllers)
                    return (
                        "BLOCKED",
                        feedback_templates.MOVE_BLOCKED_BY_GATE_WITH_SWITCHES.format(
                            gate_id=gate["id"],
                            position=fwd,
                            switches=switch_list,
                        ),
                    )
                return (
                    "BLOCKED",
                    feedback_templates.MOVE_BLOCKED_BY_GATE.format(
                        gate_id=gate["id"],
                        position=fwd,
                    ),
                )
            return "BLOCKED", feedback_templates.MOVE_BLOCKED_GENERIC
        if terminated and reward > 0 and curr_pos == goal:
            return "DONE", feedback_templates.REACHED_GOAL.format(goal=goal)
        return "MOVED", feedback_templates.MOVED_TO.format(position=curr_pos)

    if action == "PICKUP":
        if (
            prev.agent_carrying != curr.agent_carrying
            or len(curr.collected_keys) > len(prev.collected_keys)
        ):
            carried = curr.agent_carrying or "a"
            return "PICKUP", feedback_templates.PICKED_UP_KEY.format(key_color=carried)
        return "NOTHING", feedback_templates.NOTHING_TO_PICK_UP

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
        if switch_ahead and not switch_here:
            if switch_ahead["switch_type"] == "hold":
                return (
                    "NOTHING",
                    feedback_templates.TOGGLE_HOLD_SWITCH_HINT.format(position=fwd),
                )
            return (
                "NOTHING",
                feedback_templates.TOGGLE_SWITCH_HINT.format(position=fwd),
            )
        if gate_ahead and not gate_ahead["open"]:
            controllers = switches_controlling_gate(task_spec, str(gate_ahead["id"]))
            if controllers:
                switch_list = ", ".join(controllers)
                return (
                    "NOTHING",
                    feedback_templates.GATE_TOGGLE_WITH_SWITCHES.format(
                        switches=switch_list,
                    ),
                )
            return "NOTHING", feedback_templates.GATE_TOGGLE_GENERIC
        return (
            "NOTHING",
            feedback_templates.TOGGLE_NO_EFFECT,
        )

    if action == "DONE":
        if terminated and reward > 0 and curr_pos == goal:
            return "DONE", feedback_templates.TASK_COMPLETE.format(goal=goal)
        return "WRONG_DONE", feedback_templates.WRONG_DONE.format(goal=goal)

    return "INVALID", feedback_templates.UNKNOWN_ACTION.format(action=action)


def format_step_feedback(
    action: str,
    prev: GridState,
    curr: GridState,
    reward: float,
    terminated: bool,
    task_spec: TaskSpecification,
) -> tuple[str, str]:
    event_type, event_message = infer_step_outcome(
        action, prev, curr, reward, terminated, task_spec
    )
    prev_pos = agent_row_col(prev)
    if event_type == "BLOCKED":
        return feedback_templates.BLOCKED_FEEDBACK.format(action=action, message=event_message, position=prev_pos), event_type
    if event_type == "TURNED":
        return feedback_templates.TURNED_FEEDBACK.format(action=action, message=event_message), event_type
    if event_type == "MOVED":
        return feedback_templates.MOVED_FEEDBACK.format(action=action, message=event_message), event_type
    if event_type == "DONE":
        return feedback_templates.SUCCESS_FEEDBACK.format(action=action, message=event_message), event_type
    if event_type == "PICKUP":
        return feedback_templates.PICKUP_FEEDBACK.format(action=action, message=event_message), event_type
    if event_type == "NOTHING":
        return feedback_templates.NOTHING_FEEDBACK.format(action=action, message=event_message, position=prev_pos), event_type
    if event_type == "OPENED":
        return feedback_templates.OPENED_FEEDBACK.format(action=action, message=event_message), event_type
    if event_type == "TOGGLED":
        return feedback_templates.TOGGLED_FEEDBACK.format(action=action, message=event_message), event_type
    if event_type == "WRONG_DONE":
        return feedback_templates.WRONG_DONE_FEEDBACK.format(action=action, message=event_message, position=prev_pos), event_type
    if event_type == "INVALID":
        return feedback_templates.INVALID_FEEDBACK.format(action=action, message=event_message, position=prev_pos), event_type
    return feedback_templates.DEFAULT_FEEDBACK.format(
        event_type=event_type,
        action=action,
        message=event_message,
    ), event_type
