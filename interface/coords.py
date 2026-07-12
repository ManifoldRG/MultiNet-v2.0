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
    """Gridworld ``(x, y)`` or ``Position`` → 1-based ``(row, column)`` with row southward.

    Rounds to the nearest integer cell. This is a no-op for the already-integer
    coordinates discrete-grid backends (MiniGrid/MultiGrid) produce, and gives a
    "nearest cell" approximation for continuous-navigation backends (e.g.
    OgbenchBackend), so mechanism-adjacency/feedback logic elsewhere in this
    module can keep comparing exact ``(row, col)`` tuples either way.
    """
    if isinstance(pos, Position):
        return (int(pos.y), int(pos.x))
    x, y = pos
    return (round(y), round(x))


def agent_row_col(state: GridState) -> tuple[int, int]:
    return to_row_col(state.agent_position)


def agent_facing(state: GridState) -> str:
    direction = state.agent_direction
    if isinstance(direction, int):
        return _DIR_TO_FACING.get(direction, "NORTH")
    # Continuous heading in degrees (task-spec frame: 0=EAST, 90=NORTH,
    # 180=WEST, 270=SOUTH) — quantized to a compass label for display only,
    # never used to decide movement outcomes.
    quadrant = round(float(direction) / 90.0) % 4
    return {0: "EAST", 1: "NORTH", 2: "WEST", 3: "SOUTH"}[quadrant]


def goal_row_col(task_spec: TaskSpecification) -> tuple[int, int]:
    target = task_spec.goal.target or task_spec.maze.goal
    return to_row_col(target)


def maze_rows_cols(task_spec: TaskSpecification) -> tuple[int, int]:
    width, height = task_spec.maze.dimensions
    return height, width


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


def key_at_cell(
    task_spec: TaskSpecification,
    state: GridState,
    row: int,
    col: int,
) -> str | None:
    for key in task_spec.mechanisms.keys:
        if key.id in state.collected_keys:
            continue
        if to_row_col(key.position) == (row, col):
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

    return observation_templates.CELL_OPEN.format(row=row, col=col)
