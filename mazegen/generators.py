from __future__ import annotations

from typing import List, Set

from gridworld.baselines import plan_bfs_path
from gridworld.task_spec import (
    GoalSpec,
    MazeLayout,
    MechanismSet,
    Position,
    Rules,
    TaskSpecification,
)

from .models import (
    Backbone,
    Coord,
    DenseMazeParams,
    LogicChain,
    MazeGenSpec,
    WindingCorridorParams,
)


def neighbors4(c: Coord) -> List[Coord]:
    x, y = c
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def build_walls_from_open(width: int, height: int, open_cells: Set[Coord]) -> Set[Coord]:
    return {(x, y) for x in range(width) for y in range(height) if (x, y) not in open_cells}


def _task_from_geometry(
    spec: MazeGenSpec,
    width: int,
    height: int,
    walls: Set[Coord],
    start: Coord,
    goal: Coord,
    metadata: dict | None = None,
) -> TaskSpecification:
    info = {
        "backbone": spec.backbone.value,
        "logic_chain": spec.logic_chain.value,
    }
    if metadata:
        info.update(metadata)
    goal_pos = Position(goal[0], goal[1])
    return TaskSpecification(
        task_id=f"{spec.backbone.value}_{spec.logic_chain.value}_{spec.seed}",
        seed=spec.seed,
        difficulty_tier=spec.difficulty_tier,
        maze=MazeLayout(
            dimensions=(width, height),
            walls=[Position(x, y) for x, y in sorted(walls)],
            start=Position(start[0], start[1]),
            goal=goal_pos,
        ),
        mechanisms=MechanismSet(),
        rules=Rules(),
        goal=GoalSpec(goal_type="reach_position", target=goal_pos),
        max_steps=1000,
        metadata=info,
        version="2.0",
        description=f"{spec.backbone.value} {spec.logic_chain.value}",
    )


def _navigation_cells(task: TaskSpecification) -> list[Coord]:
    """Cell sequence of the gridworld plan, with in-place turns collapsed."""
    planned = plan_bfs_path(task)
    if not planned.success:
        return []
    cells: list[Coord] = []
    for pos in planned.positions:
        if not cells or cells[-1] != pos:
            cells.append(pos)
    return cells


def _winding_corridor_geometry(size: int, vertical: bool) -> tuple[set[Coord], Coord, Coord]:
    """Fill a square with the M1–M6 winding corridor.

    Lanes are one cell wide and run the full interior. Each divider keeps a
    single opening, and the opening alternates ends. 8×8 and 10×10 open the
    first divider at the far end; 14×14 opens it beside the start, matching
    those maps. The vertical layout swaps x and y. On 14×14 that layout starts at
    the far end of the first lane, as the benchmark map does.
    """
    if size < 6:
        raise ValueError(f"Winding corridor size {size} is too small")
    far = size - 2
    near = 1
    far_first = size != 14
    dividers = list(range(2, size - 1, 2))

    def gap_at(index: int) -> int:
        use_far = far_first if index % 2 == 0 else not far_first
        return far if use_far else near

    open_cells: set[Coord] = {
        (x, y)
        for x in range(1, size - 1)
        for y in range(1, size - 1)
    }
    for index, divider in enumerate(dividers):
        gap = gap_at(index)
        for along in range(1, size - 1):
            if along == gap:
                continue
            cell = (divider, along) if vertical else (along, divider)
            open_cells.discard(cell)

    last_gap = gap_at(len(dividers) - 1)
    last_divider = dividers[-1]
    if vertical:
        start = (1, far) if size == 14 else (1, 1)
        goal = (last_divider, last_gap)
    else:
        start = (1, 1)
        goal = (last_gap, last_divider)
    return open_cells, start, goal


def generate_winding_corridor(spec: MazeGenSpec) -> TaskSpecification:
    assert spec.backbone == Backbone.WINDING_CORRIDOR
    if spec.grid_width != spec.grid_height:
        raise ValueError("Winding corridors must be square")
    params: WindingCorridorParams = spec.backbone_params or WindingCorridorParams()
    size = spec.grid_width
    open_cells, start, goal = _winding_corridor_geometry(size, params.vertical)
    walls = build_walls_from_open(size, size, open_cells)
    return _task_from_geometry(
        spec,
        size,
        size,
        walls,
        start,
        goal,
        {"orientation": "vertical" if params.vertical else "horizontal"},
    )

def _carve_dense_maze_grid(size: int, rng) -> tuple[set[Coord], int, int]:
    """Carve the M1–M6 dense maze: a perfect maze plus a corner notch.

    Rooms sit on odd coordinates and stop one cell short of the far border.
    That extra row and column stay walls, except a two-cell dead end in the
    bottom-right corner, which is how those maps meet the far edge.
    """
    cell_w = (size - 1) // 2
    cell_h = (size - 1) // 2
    if cell_w < 2 or cell_h < 2:
        raise ValueError(f"Dense maze size {size} is too small")
    width = size
    height = size

    open_cells: set[Coord] = set()

    for cx in range(cell_w):
        for cy in range(cell_h):
            open_cells.add((2 * cx + 1, 2 * cy + 1))

    visited = set()
    stack = [(0, 0)]
    visited.add((0, 0))

    while stack:
        cx, cy = stack[-1]
        neighbors = []
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < cell_w and 0 <= ny < cell_h and (nx, ny) not in visited:
                neighbors.append((nx, ny, dx, dy))

        if not neighbors:
            stack.pop()
            continue

        nx, ny, dx, dy = rng.choice(neighbors)
        wall_x = 2 * cx + 1 + dx
        wall_y = 2 * cy + 1 + dy
        open_cells.add((wall_x, wall_y))

        visited.add((nx, ny))
        stack.append((nx, ny))

    # Two-cell dead end in the bottom-right corner.
    open_cells.add((size - 2, size - 3))
    open_cells.add((size - 2, size - 2))

    return open_cells, width, height


def _add_dense_maze_loops(open_cells: set[Coord], width: int, height: int, rng, loop_count: int) -> None:
    candidates = []
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            if (x, y) in open_cells:
                continue
            # candidate interior wall between two open cells
            horiz = (x - 1, y) in open_cells and (x + 1, y) in open_cells
            vert = (x, y - 1) in open_cells and (x, y + 1) in open_cells
            if horiz or vert:
                candidates.append((x, y))

    rng.shuffle(candidates)
    for c in candidates[:loop_count]:
        open_cells.add(c)


def generate_dense_maze(spec: MazeGenSpec) -> TaskSpecification:
    assert spec.backbone == Backbone.DENSE_MAZE
    rng = spec.rng()
    p: DenseMazeParams = spec.backbone_params

    if spec.grid_width != spec.grid_height:
        raise ValueError("Dense mazes must be square")
    open_cells, width, height = _carve_dense_maze_grid(spec.grid_width, rng)

    if p.add_loops and p.loop_count > 0:
        _add_dense_maze_loops(open_cells, width, height, rng, p.loop_count)

    # Same corners as the benchmark maps: start at the top-left room, goal
    # on the top row at the last room.
    start = (1, 1)
    goal = (width - 3, 1)
    walls = build_walls_from_open(width, height, open_cells)
    return _task_from_geometry(
        spec,
        width,
        height,
        walls,
        start,
        goal,
        {"dense_maze_cells": [(width - 1) // 2, (height - 1) // 2]},
    )


def _spaced_indices(path_len: int, count: int) -> list[int]:
    """count strictly increasing indices strictly between the start and goal cells."""
    if path_len < count + 2:
        raise ValueError(f"Path of {path_len} cells is too short for {count} mechanism slots")
    raw = [max(1, (i * path_len) // (count + 1)) for i in range(1, count + 1)]
    fixed: list[int] = []
    previous = 0
    for idx in raw:
        idx = min(max(idx, previous + 1), path_len - 2)
        fixed.append(idx)
        previous = idx
    if len(set(fixed)) < count or fixed[-1] >= path_len - 1:
        raise ValueError(f"Path of {path_len} cells is too short for {count} mechanism slots")
    return fixed


def slots_for_logic(task: TaskSpecification, logic: LogicChain) -> dict:
    """Place mechanism slots along the gridworld path. The layout does not choose the chain."""
    if logic == LogicChain.NONE:
        return {}
    path = _navigation_cells(task)
    if not path:
        raise ValueError("Layout has no path from start to goal")
    if logic in {LogicChain.KD, LogicChain.SG}:
        n_slots = 2
    elif logic == LogicChain.KSK:
        n_slots = 6
    else:
        n_slots = 4
    indices = _spaced_indices(len(path), n_slots)
    cells = [path[idx] for idx in indices]
    if len(set(cells)) < n_slots:
        raise ValueError("Mechanism slots landed on the same cell")
    slots = {
        "pickup_1_candidates": [cells[0]],
        "blocker_1_candidates": [cells[1]],
    }
    if n_slots >= 4:
        slots["pickup_2_candidates"] = [cells[2]]
        slots["blocker_2_candidates"] = [cells[3]]
    if n_slots == 6:
        slots["pickup_3_candidates"] = [cells[4]]
        slots["blocker_3_candidates"] = [cells[5]]
    return slots


def generate_from_spec(spec: MazeGenSpec) -> TaskSpecification:
    if spec.backbone == Backbone.WINDING_CORRIDOR:
        return generate_winding_corridor(spec)
    if spec.backbone == Backbone.DENSE_MAZE:
        return generate_dense_maze(spec)
    raise ValueError(f"Unsupported backbone: {spec.backbone}")

