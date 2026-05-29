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
            return "OPENED", f"Opened {color} door {door_id} and moved to {curr_pos}."
        return "OPENED", f"Opened {color} door {door_id}."

    if action in ("TURN_LEFT", "TURN_RIGHT"):
        if prev.agent_direction != curr.agent_direction:
            return "TURNED", f"Now facing {agent_facing(curr)}."
        return "NOTHING", f"{action} had no effect."

    if action == "MOVE_FORWARD":
        if prev_pos == curr_pos:
            fwd = forward_cell(prev)
            key_color = key_at_cell(task_spec, prev, fwd[0], fwd[1])
            if key_color:
                return (
                    "BLOCKED",
                    f"MOVE_FORWARD blocked by a {key_color} key at {fwd}. "
                    "Keys occupy their cell; you cannot walk onto them. "
                    "Face the key and use PICKUP from your current cell.",
                )
            gate = gate_at_cell(task_spec, prev, fwd[0], fwd[1])
            if gate and not gate["open"]:
                controllers = switches_controlling_gate(task_spec, str(gate["id"]))
                if controllers:
                    switch_list = ", ".join(controllers)
                    return (
                        "BLOCKED",
                        f"MOVE_FORWARD blocked by closed gate {gate['id']} at {fwd}. "
                        f"Activate switch(es) {switch_list} to open it.",
                    )
                return (
                    "BLOCKED",
                    f"MOVE_FORWARD blocked by closed gate {gate['id']} at {fwd}.",
                )
            return "BLOCKED", "MOVE_FORWARD blocked by wall or closed door/gate."
        if terminated and reward > 0 and curr_pos == goal:
            return "DONE", f"Reached goal at {goal}."
        return "MOVED", f"Moved to {curr_pos}."

    if action == "PICKUP":
        if (
            prev.agent_carrying != curr.agent_carrying
            or len(curr.collected_keys) > len(prev.collected_keys)
        ):
            carried = curr.agent_carrying or "a"
            return "PICKUP", f"Picked up {carried} key."
        return "NOTHING", "Nothing to pick up here."

    if action == "TOGGLE":
        if (
            prev.active_switches != curr.active_switches
            or prev.open_gates != curr.open_gates
        ):
            return "TOGGLED", "Toggled switch or gate state changed."
        fwd = forward_cell(prev)
        switch_ahead = switch_at_cell(task_spec, fwd[0], fwd[1])
        switch_here = switch_at_cell(task_spec, prev_pos[0], prev_pos[1])
        gate_ahead = gate_at_cell(task_spec, prev, fwd[0], fwd[1])
        if switch_ahead and not switch_here:
            if switch_ahead["switch_type"] == "hold":
                return (
                    "NOTHING",
                    f"TOGGLE had no effect. MOVE_FORWARD onto the switch at {fwd} "
                    "(hold switches activate while you stand on them).",
                )
            return (
                "NOTHING",
                f"TOGGLE had no effect. MOVE_FORWARD onto the switch at {fwd}, then TOGGLE.",
            )
        if gate_ahead and not gate_ahead["open"]:
            controllers = switches_controlling_gate(task_spec, str(gate_ahead["id"]))
            if controllers:
                switch_list = ", ".join(controllers)
                return (
                    "NOTHING",
                    "Gates cannot be toggled directly. "
                    f"Activate switch(es) {switch_list} instead.",
                )
            return "NOTHING", "Gates cannot be toggled directly. Activate a linked switch instead."
        return (
            "NOTHING",
            "TOGGLE had no effect. Stand on a switch and TOGGLE, or use PICKUP/keys for doors.",
        )

    if action == "DONE":
        if terminated and reward > 0 and curr_pos == goal:
            return "DONE", f"Task complete at {goal}."
        return "WRONG_DONE", f"DONE called but not at goal {goal}."

    return "INVALID", f"Unknown or unsupported action {action}."


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
        return f"BLOCKED — {action}: {event_message} You remain at {prev_pos}.", event_type
    if event_type == "TURNED":
        return f"TURNED — {action}: {event_message}", event_type
    if event_type == "MOVED":
        return f"MOVED — {action}: {event_message}", event_type
    if event_type == "DONE":
        return f"SUCCESS — {action}: {event_message}", event_type
    if event_type == "PICKUP":
        return f"PICKUP — {action}: {event_message}", event_type
    if event_type == "NOTHING":
        return f"NOTHING — {action}: {event_message} You remain at {prev_pos}.", event_type
    if event_type == "OPENED":
        return f"OPENED — {action}: {event_message}", event_type
    if event_type == "TOGGLED":
        return f"TOGGLED — {action}: {event_message}", event_type
    if event_type == "WRONG_DONE":
        return f"WRONG DONE — {action}: {event_message} You remain at {prev_pos}.", event_type
    if event_type == "INVALID":
        return f"INVALID — {action}: {event_message} You remain at {prev_pos}.", event_type
    return f"{event_type} — {action}: {event_message}", event_type
