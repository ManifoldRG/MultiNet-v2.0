"""Maze text layout and MiniGrid RGB rendering for NLU observations."""

from __future__ import annotations

import base64
import io
import json
from typing import TYPE_CHECKING, Literal

import numpy as np
from PIL import Image

from interface.coords import (
    agent_facing,
    agent_row_col,
    goal_row_col,
    inventory_list,
    live_key_position,
    maze_rows_cols,
    to_row_col,
    wall_cells,
)
from prompting_experiments.prompt_templates import observation as observation_templates

if TYPE_CHECKING:
    from gridworld.backends.base import GridState
    from gridworld.task_spec import TaskSpecification

ObservationTextFormat = Literal["coords", "json", "ascii"]
_FACING = {"NORTH": "^", "EAST": ">", "SOUTH": "v", "WEST": "<"}
_ASCII_MAP_HEADER = (
    "Map. Every cell is a two-character token and tokens are separated by a"
    " space, so row N is the Nth line and column N is the Nth token on it."
    " No border is drawn: the first token of the first line is cell (1,1)."
)


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
        # Live position, not the spec's: a dropped key moves (see coords.live_key_position)
        row, col = to_row_col(live_key_position(key, state) if state else key.position)
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


def _dumps(obj, indent: int = 0) -> str:
    """json.dumps(indent=2) with [row, col] lists kept on one line."""
    pad = " " * indent
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        body = ",\n".join(
            f"{pad}  {json.dumps(k)}: {_dumps(v, indent + 2)}" for k, v in obj.items()
        )
        return f"{{\n{body}\n{pad}}}"
    if isinstance(obj, list):
        if not obj or all(isinstance(x, int) for x in obj) or all(
            isinstance(x, list) and all(isinstance(i, int) for i in x) for x in obj
        ):
            return "[" + ", ".join(json.dumps(v) for v in obj) + "]"
        body = ",\n".join(f"{pad}  {_dumps(v, indent + 2)}" for v in obj)
        return f"[\n{body}\n{pad}]"
    return json.dumps(obj)


def _key_status(key, state: GridState | None) -> dict:
    collected = state.collected_keys if state else set()
    row, col = to_row_col(live_key_position(key, state) if state else key.position)
    if state and key.id in collected:
        carried = state.agent_carrying == key.color
        return {"color": key.color, "status": "carried" if carried else "used"}
    dropped = state is not None and to_row_col(key.position) != (row, col)
    return {
        "color": key.color,
        "status": "dropped" if dropped else "on_ground",
        "row": row,
        "col": col,
    }


def _mechanism_payload(task_spec: TaskSpecification, state: GridState | None = None) -> dict:
    open_doors = state.open_doors if state else set()
    active = state.active_switches if state else set()
    open_gates = state.open_gates if state else set()
    return {
        "keys": [_key_status(key, state) for key in task_spec.mechanisms.keys],
        "doors": [
            {
                "status": "open" if door.id in open_doors else door.initial_state,
                "requires_key": door.requires_key,
                "row": to_row_col(door.position)[0],
                "col": to_row_col(door.position)[1],
            }
            for door in task_spec.mechanisms.doors
        ],
        "switches": [
            {
                "switch_type": switch.switch_type,
                "row": to_row_col(switch.position)[0],
                "col": to_row_col(switch.position)[1],
                "status": "on" if switch.id in active else switch.initial_state,
                "controls": list(switch.controls),
            }
            for switch in task_spec.mechanisms.switches
        ],
        "gates": [
            {
                "id": gate.id,
                "row": to_row_col(gate.position)[0],
                "col": to_row_col(gate.position)[1],
                "status": "open" if gate.id in open_gates else gate.initial_state,
                "initial_status": gate.initial_state,
            }
            for gate in task_spec.mechanisms.gates
        ],
    }


def _gate_ids(task_spec: TaskSpecification) -> tuple[dict[str, int], dict[str, int]]:
    gates = {g.id: i for i, g in enumerate(task_spec.mechanisms.gates, start=1)}
    switches = {s.id: min(gates[g] for g in s.controls) for s in task_spec.mechanisms.switches}
    return gates, switches


def _ascii_grid(
    task_spec: TaskSpecification,
    state: GridState | None,
    include_facing: bool,
) -> tuple[list[str], list[tuple[str, str]]]:
    rows, cols = maze_rows_cols(task_spec)
    walls = wall_cells(task_spec)
    gates, switches = _gate_ids(task_spec)
    open_doors = state.open_doors if state else set()
    open_gates = state.open_gates if state else set()
    collected = state.collected_keys if state else set()
    cells: dict[tuple[int, int], tuple[str, str]] = {}

    def place(row, col, token, desc):
        cells.setdefault((row, col), (token, desc))

    if state is not None:
        pos, facing = agent_row_col(state), agent_facing(state)
    else:
        pos, facing = to_row_col(task_spec.maze.start), "EAST"
    place(*pos, _FACING[facing] if include_facing else "A",
          f"you, facing {facing}" if include_facing else "you")
    place(*goal_row_col(task_spec), "G", "the goal")

    for key in task_spec.mechanisms.keys:
        if key.id in collected:
            continue
        row, col = to_row_col(live_key_position(key, state) if state else key.position)
        place(row, col, f"k{key.color[0].upper()}", f"{key.color} key")

    for door in task_spec.mechanisms.doors:
        row, col = to_row_col(door.position)
        if door.id in open_doors:
            place(row, col, "dO", "open door (passable)")
        else:
            place(row, col, f"d{door.requires_key[0].upper()}",
                  f"{door.initial_state} door, needs the {door.requires_key} key")

    gate_token = {
        g.id: "gO" if g.id in open_gates else f"g{gates[g.id]}"
        for g in task_spec.mechanisms.gates
    }
    for gate in task_spec.mechanisms.gates:
        row, col = to_row_col(gate.position)
        if gate.id in open_gates:
            place(row, col, "gO", "open gate (passable)")
        else:
            openers = ", ".join(
                f"s{switches[s.id]}" for s in task_spec.mechanisms.switches
                if gate.id in s.controls
            )
            place(row, col, gate_token[gate.id], f"closed gate, opened by {openers}")

    for switch in task_spec.mechanisms.switches:
        row, col = to_row_col(switch.position)
        place(row, col, f"s{switches[switch.id]}",
              f"{switch.switch_type} switch, controls "
              + ", ".join(gate_token[g] for g in switch.controls))

    grid, legend = [], []
    used_wall = used_open = False
    for row in range(1, rows + 1):
        line = []
        for col in range(1, cols + 1):
            if (row, col) in cells:
                line.append(cells[(row, col)][0])
            elif (row, col) in walls:
                line.append("#")
                used_wall = True
            else:
                line.append(".")
                used_open = True
        grid.append(" ".join(t.ljust(2) for t in line).rstrip())
    if used_wall:
        legend.append(("#", "wall"))
    if used_open:
        legend.append((".", "open floor"))
    seen = {t for t, _ in legend}
    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            entry = cells.get((row, col))
            if entry and entry[0] not in seen:
                seen.add(entry[0])
                legend.append(entry)
    return grid, legend


def _ascii_status(task_spec: TaskSpecification, state: GridState | None) -> list[str]:
    _, switches = _gate_ids(task_spec)
    collected = state.collected_keys if state else set()
    active = state.active_switches if state else set()
    carrying = inventory_list(state) if state else []
    lines = ["Status:", f"  Carrying: {', '.join(f'{c} key' for c in carrying) or 'nothing'}"]
    spent = [
        k.color for k in task_spec.mechanisms.keys
        if k.id in collected and k.color not in carrying
    ]
    if spent:
        lines.append(f"  Keys used up: {', '.join(spent)}")
    if task_spec.mechanisms.switches:
        on = [
            f"s{switches[s.id]}" for s in task_spec.mechanisms.switches
            if (s.id in active if state else s.initial_state == "on")
        ]
        lines.append(f"  Switches on: {', '.join(on) or 'none'}")
    return lines


def _ascii_block(task_spec, state, include_facing: bool) -> str:
    grid, legend = _ascii_grid(task_spec, state, include_facing)
    return "\n".join([
        _ASCII_MAP_HEADER, "", *grid, "",
        "Legend:",
        *(f"  {token} = {desc}" for token, desc in legend),
        "", *_ascii_status(task_spec, state),
    ])


def render_initial_maze_text(
    task_spec: TaskSpecification,
    *,
    observation_text_format: ObservationTextFormat = "coords",
    include_facing: bool = False,
) -> str:
    if observation_text_format == "json":
        rows, cols = maze_rows_cols(task_spec)
        return _dumps({
            "world": {"rows": rows, "cols": cols},
            "coordinates": observation_templates.COORDINATE_EXPLANATION,
            "start": list(to_row_col(task_spec.maze.start)),
            "goal": list(goal_row_col(task_spec)),
            "walls": [list(c) for c in sorted(wall_cells(task_spec))],
            "mechanisms": _mechanism_payload(task_spec),
        })
    if observation_text_format == "ascii":
        return (
            observation_templates.COORDINATE_EXPLANATION
            + "\n\n"
            + _ascii_block(task_spec, None, include_facing)
        )
    return "\n".join(_static_layout_lines(task_spec) + _mechanism_lines(task_spec))


def render_user_observation_text(
    task_spec: TaskSpecification,
    state: GridState,
    *,
    include_facing: bool = False,
    observation_text_format: ObservationTextFormat = "coords",
) -> str:
    pos = agent_row_col(state)
    inv = inventory_list(state)
    if observation_text_format == "json":
        agent: dict = {"row": pos[0], "col": pos[1]}
        if include_facing:
            agent["facing"] = agent_facing(state)
        return _dumps({
            "agent": agent,
            "inventory": inv,
            "map_contents": _mechanism_payload(task_spec, state),
        })
    if observation_text_format == "ascii":
        return _ascii_block(task_spec, state, include_facing)

    agent_line = (
        observation_templates.CURRENT_AGENT_LINE.format(position=pos, facing=agent_facing(state))
        if include_facing
        else observation_templates.CURRENT_AGENT_POSITION_LINE.format(position=pos)
    )
    head = [
        agent_line,
        observation_templates.CURRENT_INVENTORY_LINE.format(inventory=", ".join(inv) or "empty"),
    ]
    collected = [k.color for k in task_spec.mechanisms.keys if k.id in state.collected_keys]
    if collected:
        head.append(observation_templates.CURRENT_KEYS_COLLECTED_LINE.format(keys=", ".join(collected)))
    head += ["", observation_templates.CURRENT_MAP_CONTENTS_HEADER]
    head += _mechanism_lines(task_spec, state) or [observation_templates.NO_MECHANISMS_LINE]
    return "\n".join(head)


def render_current_inventory_text(state: GridState) -> str:
    inv = ", ".join(inventory_list(state)) or "empty"
    return observation_templates.CURRENT_INVENTORY_LINE.format(inventory=inv)
