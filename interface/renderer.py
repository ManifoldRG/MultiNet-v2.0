"""Maze text layout and MiniGrid RGB rendering for NLU observations."""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from interface.coords import (
    agent_facing,
    agent_row_col,
    goal_row_col,
    inventory_list,
    maze_rows_cols,
    to_row_col,
    wall_cells,
)

if TYPE_CHECKING:
    from gridworld.backends.base import GridState
    from gridworld.task_spec import TaskSpecification

#TODO: Move to utils.py
def rgb_to_png_bytes(rgb: np.ndarray) -> bytes:
    img = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def rgb_to_image_block(rgb: np.ndarray) -> dict:
    b64 = base64.b64encode(rgb_to_png_bytes(rgb)).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


def _static_layout_lines(task_spec: TaskSpecification) -> list[str]:
    rows, cols = maze_rows_cols(task_spec)
    walls = wall_cells(task_spec)
    wall_str = ", ".join(f"({r},{c})" for r, c in sorted(walls)) or "none"
    start = to_row_col(task_spec.maze.start)
    goal = goal_row_col(task_spec)
    return [
        f"The world is a {rows} by {cols} grid.",
        "Coordinates: JSON lists use ``[x, y]`` (east, south) from the **top-left** corner ``(1, 1)``;"
        " tuples in this text use ``(row, column)`` matching env state (row southward, column east)."
        " So ``x`` = column index, ``y`` = row index.",
        f"The start is at {start}.",
        f"The goal is at {goal}.",
        f"The following cells are walls: {wall_str}.",
    ]


def _mechanism_lines(task_spec: TaskSpecification, state: GridState | None = None) -> list[str]:
    parts: list[str] = []
    collected = state.collected_keys if state else set()
    open_doors = state.open_doors if state else set()
    active = state.active_switches if state else set()
    open_gates = state.open_gates if state else set()

    for key in task_spec.mechanisms.keys:
        if key.id in collected:
            continue
        row, col = to_row_col(key.position)
        parts.append(f"There is a {key.color} key at ({row},{col}).")

    for door in task_spec.mechanisms.doors:
        row, col = to_row_col(door.position)
        status = "open" if door.id in open_doors else door.initial_state
        parts.append(
            f"There is a {status} {door.requires_key} door at ({row},{col})."
            f" It requires the {door.requires_key} key to open."
        )

    for switch in task_spec.mechanisms.switches:
        row, col = to_row_col(switch.position)
        on_off = "on" if switch.id in active else switch.initial_state
        controls = ", ".join(switch.controls)
        parts.append(
            f"There is a {switch.switch_type} switch at ({row},{col}) (currently {on_off})."
            f" It controls: {controls}."
        )

    for gate in task_spec.mechanisms.gates:
        row, col = to_row_col(gate.position)
        cur = "open" if gate.id in open_gates else gate.initial_state
        parts.append(
            f"There is a gate ({gate.id}) at ({row},{col})."
            f" It is currently {cur} (initially {gate.initial_state})."
        )
    return parts


def render_initial_maze_text(task_spec: TaskSpecification) -> str:
    return "\n".join(_static_layout_lines(task_spec) + _mechanism_lines(task_spec))


def render_user_observation_text(task_spec: TaskSpecification, state: GridState) -> str:
    goal = goal_row_col(task_spec)
    pos = agent_row_col(state)
    inv = ", ".join(inventory_list(state)) or "empty"
    head = [
        "Current situation (this step):",
        f"The goal is at {goal}.",
        f"You are at {pos} facing {agent_facing(state)}.",
        f"Environment steps used so far: {state.step_count} (max {state.max_steps} before timeout).",
        f"Your inventory: {inv}.",
        "",
        "Map contents as of this step (keys on the ground, doors, switches, gates):",
    ]
    mech = _mechanism_lines(task_spec, state)
    if mech:
        head.extend(mech)
    else:
        head.append("(No keys on the ground, doors, switches, or gates in the current state description.)")
    return "\n".join(head)

