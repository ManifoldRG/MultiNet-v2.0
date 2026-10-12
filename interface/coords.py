"""Coordinate and direction helpers for NLU prompts over gridworld state."""

from __future__ import annotations

from gridworld.backends.base import GridState
from gridworld.task_spec import Position, TaskSpecification
from prompting_experiments.prompt_templates import observation as observation_templates

FACING_ORDER = ["NORTH", "EAST", "SOUTH", "WEST"]

FACING_TO_DELTA: dict[str, tuple[int, int]] = {
    "NORTH": (-1, 0),
    "EAST": (0, 1),
    "SOUTH": (1, 0),
    "WEST": (0, -1),
}

_DIR_TO_FACING = {
    0: "EAST",
    1: "SOUTH",
    2: "WEST",
    3: "NORTH",
}


def to_row_col(pos: Position | tuple[int, int]) -> tuple[int, int]:
    """Gridworld ``(x, y)`` or ``Position`` → 1-based ``(row, column)`` with row southward."""
    if isinstance(pos, Position):
        return (int(pos.y), int(pos.x))
    x, y = pos
    return (int(y), int(x))


def agent_row_col(state: GridState) -> tuple[int, int]:
    return to_row_col(state.agent_position)


def agent_facing(state: GridState) -> str:
    return direction_name(state.agent_direction)


def direction_name(direction: int) -> str:
    return _DIR_TO_FACING.get(int(direction) % 4, "NORTH")


def rotating_dirs(task_spec: TaskSpecification, state: GridState | None) -> tuple[int, ...]:
    if state is None:
        return tuple(int(d) for d in task_spec.mechanisms.rotating_initial_directions)
    return tuple(int(d) for d in state.rotator_dirs)


def teleporter_link(
    task_spec: TaskSpecification,
    row: int,
    col: int,
) -> tuple[tuple[int, int], str] | None:
    """Where stepping on ``(row, col)`` lands, and that teleporter's color.

    One-way teleporters only send from the first end to the second.
    """
    for teleporter in task_spec.mechanisms.teleporters:
        end_a = to_row_col(teleporter.position_a)
        end_b = to_row_col(teleporter.position_b)
        if (row, col) == end_a:
            return end_b, teleporter.color
        if teleporter.bidirectional and (row, col) == end_b:
            return end_a, teleporter.color
    return None


def rotating_pointing(
    task_spec: TaskSpecification,
    state: GridState | None,
    row: int,
    col: int,
) -> str | None:
    dirs = rotating_dirs(task_spec, state)
    for i, cell in enumerate(task_spec.mechanisms.rotating_tiles):
        if to_row_col(cell) == (row, col):
            return direction_name(dirs[i])
    return None


def goal_row_col(task_spec: TaskSpecification) -> tuple[int, int]:
    return to_row_col(task_spec.resolved_goal())


def maze_rows_cols(task_spec: TaskSpecification) -> tuple[int, int]:
    # ``dimensions`` counts the border wall both backends ring the grid with, so
    # the playable interior is two smaller on each axis.
    width, height = task_spec.maze.dimensions
    return height - 2, width - 2


def wall_cells(task_spec: TaskSpecification) -> set[tuple[int, int]]:
    return {to_row_col(w) for w in task_spec.maze.walls}


def inventory_list(state: GridState) -> list[str]:
    items: list[str] = []
    if state.agent_carrying:
        items.append(str(state.agent_carrying))
    return items


def forward_cell(state: GridState) -> tuple[int, int]:
    row, col = agent_row_col(state)
    dr, dc = FACING_TO_DELTA[agent_facing(state)]
    return (row + dr, col + dc)


def entered_cell(task_spec: TaskSpecification, state: GridState) -> tuple[int, int]:
    """Cell a MOVE_FORWARD from ``state`` enters, in ``(row, col)``.

    Standing on a rotating tile enters the arrow's cell. Otherwise, the cell ahead.
    """
    row, col = agent_row_col(state)
    pointing = rotating_pointing(task_spec, state, row, col)
    if pointing is None:
        return forward_cell(state)
    dr, dc = FACING_TO_DELTA[pointing]
    return (row + dr, col + dc)


def move_landing(
    task_spec: TaskSpecification,
    state: GridState,
) -> tuple[tuple[int, int], tuple[tuple[int, int], str] | None]:
    """Where a MOVE_FORWARD from ``state`` lands, and the teleporter it used.

    An active teleporter end lands on the other end.
    """
    entered = entered_cell(task_spec, state)
    link = teleporter_link(task_spec, *entered)
    return (link[0] if link is not None else entered), link


def live_key_position(key, state: GridState):
    """Where a key actually is now.

    Keys move: DROP puts a held key back on the grid in the agent's own cell, so
    the task spec's position is only correct until the first drop.
    `state.key_positions` is scanned from the live grid; fall back to the spec
    position for states built before that field existed.
    """
    pos = (state.key_positions or {}).get(key.id)
    if pos is None:
        return key.position
    return Position(x=pos[0], y=pos[1])


# Backward-compatible alias for the pre-promotion private name.
_live_key_position = live_key_position


def key_at_cell(
    task_spec: TaskSpecification,
    state: GridState,
    row: int,
    col: int,
) -> str | None:
    for key in task_spec.mechanisms.keys:
        if key.id in state.collected_keys:
            continue
        if to_row_col(live_key_position(key, state)) == (row, col):
            return key.color
    return None


def switch_at_cell(
    task_spec: TaskSpecification,
    row: int,
    col: int,
) -> dict[str, str] | None:
    for switch in task_spec.mechanisms.switches:
        if to_row_col(switch.position) == (row, col):
            return {"id": switch.id, "switch_type": switch.switch_type}
    return None


def door_at_cell(
    task_spec: TaskSpecification,
    state: GridState,
    row: int,
    col: int,
) -> dict[str, str | bool] | None:
    for door in task_spec.mechanisms.doors:
        if to_row_col(door.position) == (row, col):
            return {
                "id": door.id,
                "open": door.id in state.open_doors,
                "requires_key": door.requires_key,
            }
    return None


def gate_at_cell(
    task_spec: TaskSpecification,
    state: GridState,
    row: int,
    col: int,
) -> dict[str, str | bool] | None:
    for gate in task_spec.mechanisms.gates:
        if to_row_col(gate.position) == (row, col):
            return {
                "id": gate.id,
                "open": gate.id in state.open_gates,
            }
    return None


def compact_ids(task_spec: TaskSpecification) -> tuple[dict[str, str], dict[str, str]]:
    gates = {g.id: f"g{i}" for i, g in enumerate(task_spec.mechanisms.gates, start=1)}
    switches = {s.id: f"s{i}" for i, s in enumerate(task_spec.mechanisms.switches, start=1)}
    return gates, switches


def switches_controlling_gate(task_spec: TaskSpecification, gate_id: str) -> list[str]:
    return [
        switch.id
        for switch in task_spec.mechanisms.switches
        if gate_id in switch.controls
    ]


def describe_cell(
    task_spec: TaskSpecification,
    state: GridState,
    row: int,
    col: int,
    *,
    walls: set[tuple[int, int]],
    goal: tuple[int, int],
    rows: int,
    cols: int,
) -> str:
    if row < 1 or row > rows or col < 1 or col > cols:
        return observation_templates.CELL_OUT_OF_BOUNDS
    if (row, col) in walls:
        return observation_templates.CELL_WALL
    if (row, col) == goal:
        return observation_templates.CELL_GOAL.format(row=row, col=col)

    key_color = key_at_cell(task_spec, state, row, col)
    if key_color:
        return observation_templates.CELL_KEY.format(
            key_color=key_color,
            row=row,
            col=col,
        )

    for door in task_spec.mechanisms.doors:
        if to_row_col(door.position) == (row, col):
            status = "open" if door.id in state.open_doors else door.initial_state
            return observation_templates.CELL_DOOR.format(
                status=status,
                requires_key=door.requires_key,
                row=row,
                col=col,
            )

    for gate in task_spec.mechanisms.gates:
        if to_row_col(gate.position) == (row, col):
            cur = "open" if gate.id in state.open_gates else gate.initial_state
            return observation_templates.CELL_GATE.format(state=cur, row=row, col=col)

    for switch in task_spec.mechanisms.switches:
        if to_row_col(switch.position) == (row, col):
            on_off = "on" if switch.id in state.active_switches else switch.initial_state
            return observation_templates.CELL_SWITCH.format(
                state=on_off,
                row=row,
                col=col,
            )

    for cell in task_spec.mechanisms.kill_cells:
        if to_row_col(cell) == (row, col):
            return observation_templates.CELL_DEATH_PORTAL.format(row=row, col=col)

    for cell in task_spec.mechanisms.frozen_tiles:
        if to_row_col(cell) == (row, col):
            return observation_templates.CELL_FROZEN_TILE.format(row=row, col=col)

    pointing = rotating_pointing(task_spec, state, row, col)
    if pointing:
        return observation_templates.CELL_ROTATING_TILE.format(
            direction=pointing, row=row, col=col
        )

    for teleporter in task_spec.mechanisms.teleporters:
        ends = (
            to_row_col(teleporter.position_a),
            to_row_col(teleporter.position_b),
        )
        if (row, col) in ends:
            return observation_templates.CELL_TELEPORTER.format(
                color=teleporter.color, row=row, col=col
            )

    return observation_templates.CELL_OPEN.format(row=row, col=col)
