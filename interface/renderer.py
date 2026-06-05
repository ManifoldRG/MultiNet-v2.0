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
from prompting_experiments.prompt_templates import observation as observation_templates

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
        observation_templates.WORLD_SIZE_LINE.format(rows=rows, cols=cols),
        observation_templates.COORDINATE_EXPLANATION,
        observation_templates.START_LINE.format(start=start),
        observation_templates.GOAL_LINE.format(goal=goal),
        observation_templates.WALLS_LINE.format(walls=wall_str)
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
        parts.append(
            observation_templates.KEY_LINE.format(color=key.color, row=row, col=col)
        )

    for door in task_spec.mechanisms.doors:
        row, col = to_row_col(door.position)
        status = "open" if door.id in open_doors else door.initial_state
        parts.append(
            observation_templates.DOOR_LINE.format(
                status=status,
                requires_key=door.requires_key,
                row=row,
                col=col,
            )
        )

    for switch in task_spec.mechanisms.switches:
        row, col = to_row_col(switch.position)
        on_off = "on" if switch.id in active else switch.initial_state
        controls = ", ".join(switch.controls)
        parts.append(
            observation_templates.SWITCH_LINE.format(
                switch_type=switch.switch_type,
                row=row,
                col=col,
                state=on_off,
                controls=controls,
            )
        )

    for gate in task_spec.mechanisms.gates:
        row, col = to_row_col(gate.position)
        cur = "open" if gate.id in open_gates else gate.initial_state
        parts.append(
            observation_templates.GATE_LINE.format(
                gate_id=gate.id,
                row=row,
                col=col,
                state=cur,
                initial_state=gate.initial_state,
            )
        )
    return parts


def render_initial_maze_text(task_spec: TaskSpecification) -> str:
    return "\n".join(_static_layout_lines(task_spec) + _mechanism_lines(task_spec))


def render_user_observation_text(
    task_spec: TaskSpecification,
    state: GridState,
    *,
    include_facing: bool = False,
) -> str:
    goal = goal_row_col(task_spec)
    pos = agent_row_col(state)
    inv = ", ".join(inventory_list(state)) or "empty"
    agent_line = (
        observation_templates.CURRENT_AGENT_LINE.format(
            position=pos,
            facing=agent_facing(state),
        )
        if include_facing
        else observation_templates.CURRENT_AGENT_POSITION_LINE.format(position=pos)
    )
    head = [
        observation_templates.CURRENT_SITUATION_HEADER,
        observation_templates.CURRENT_GOAL_LINE.format(goal=goal),
        agent_line,
        observation_templates.CURRENT_INVENTORY_LINE.format(inventory=inv),
        "",
        observation_templates.CURRENT_MAP_CONTENTS_HEADER,
    ]
    mech = _mechanism_lines(task_spec, state)
    if mech:
        head.extend(mech)
    else:
        head.append(observation_templates.NO_MECHANISMS_LINE)
    return "\n".join(head)
