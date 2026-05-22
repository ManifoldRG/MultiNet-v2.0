from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from automatic_maze_generation.mazegen.models import Door, Gate, Key, MazeInstance, Switch

from nlu_benchmark.env import GridState, GridWorldEnv


def _swap_validation_v04_dimensions_if_raw(maze: dict[str, Any], task_id: str) -> None:
    """``validation_10_v04_single_key.json`` lists ``dimensions`` as ``[cols, rows]`` = ``[14, 12]``; normalize once."""
    if str(task_id) != "validation_10_v04_single_key":
        return
    dims = maze.get("dimensions")
    if isinstance(dims, list) and len(dims) == 2 and dims[0] == 14 and dims[1] == 12:
        maze["dimensions"] = [12, 14]


def _json_cell_to_pos(pair: list | tuple) -> tuple[int, int]:
    """JSON cell ``[x, y]`` with origin at **top-left** ``(1, 1)``: ``x`` east (column), ``y`` south (row).

    Same as ``[column, row]``. Internal env tuple is ``(row, column)``.
    """
    col, row = int(pair[0]), int(pair[1])
    return (row, col)


def _normalize_mechanisms_from_json(mechs: dict[str, Any] | None) -> dict[str, Any]:
    """Deep-copy mechanisms; JSON ``position`` is ``[x, y]`` / ``[column, row]``, stored as ``[row, column]`` internally."""
    m = copy.deepcopy(mechs or {})
    for name in ("keys", "doors", "switches", "gates"):
        for item in m.get(name, []):
            pos = item.get("position")
            if isinstance(pos, (list, tuple)) and len(pos) == 2:
                r, c = _json_cell_to_pos(pos)
                item["position"] = [r, c]
    return m


def _task_dict_to_env(data: dict[str, Any]) -> GridWorldEnv:
    maze = data["maze"]
    _swap_validation_v04_dimensions_if_raw(maze, str(data.get("task_id", "")))
    rows, cols = maze["dimensions"]
    walls = {_json_cell_to_pos(w) for w in maze["walls"]}
    start = _json_cell_to_pos(maze["start"])
    goal = _json_cell_to_pos(maze["goal"])
    max_steps = data.get("max_steps", 100)
    mechanisms = _normalize_mechanisms_from_json(data.get("mechanisms", {}))
    return GridWorldEnv(
        rows=rows,
        cols=cols,
        walls=walls,
        start=start,
        goal=goal,
        max_steps=max_steps,
        mechanisms=mechanisms,
    )


def load_maze(path) -> GridWorldEnv:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return _task_dict_to_env(data)


def load_maze_from_dict(data: dict[str, Any]) -> GridWorldEnv:
    """Build env from a parsed task dict (same schema as ``load_maze`` JSON files)."""
    return _task_dict_to_env(data)


def grid_state_to_maze_instance(st: GridState) -> MazeInstance:
    def rc_to_xy(pos):
        row, col = pos
        # Mazegen ``y`` increases south from the north edge; NLU row 1 is north (top) → ``y = row - 1``.
        return (col - 1, row - 1)

    return MazeInstance(
        width=st.cols,
        height=st.rows,
        walls={rc_to_xy(w) for w in st.walls},
        start=rc_to_xy(st.start),
        goal=rc_to_xy(st.goal),
        keys=[
            Key(id=k.get("id", f"key_{i}"), position=rc_to_xy(tuple(k["position"])), color=k["color"])
            for i, k in enumerate(st.keys)
        ],
        doors=[
            Door(
                id=d.get("id", f"door_{i}"),
                position=rc_to_xy(tuple(d["position"])),
                requires_key=d["requires_key"],
                initial_state=d.get("initial_state", "locked"),
            )
            for i, d in enumerate(st.doors)
        ],
        switches=[
            Switch(
                id=s.get("id", f"switch_{i}"),
                position=rc_to_xy(tuple(s["position"])),
                controls=list(s.get("controls", [])),
                switch_type=s.get("switch_type", "toggle"),
                initial_state=s.get("initial_state", "off"),
            )
            for i, s in enumerate(st.switches)
        ],
        gates=[
            Gate(
                id=g.get("id", f"gate_{i}"),
                position=rc_to_xy(tuple(g["position"])),
                initial_state=g.get("initial_state", "closed"),
            )
            for i, g in enumerate(st.gates)
        ],
    )


def load_maze_instance(path) -> MazeInstance:
    """Parse task JSON like :func:`load_maze`, reset env once, and build a :class:`MazeInstance` for mazegen."""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    return maze_instance_from_task_dict(data)


def maze_instance_from_task_dict(data: dict[str, Any]) -> MazeInstance:
    """Same as :func:`load_maze_instance` but from an already-parsed task dict (avoids a second disk read)."""
    inst = grid_state_to_maze_instance(_task_dict_to_env(data).reset())
    return replace(inst, metadata=dict(data.get("metadata", {})))


def task_dict_shrink_dimensions_minus_two(data: dict[str, Any]) -> dict[str, Any]:
    """
    Return a deep copy whose ``maze.dimensions`` are each reduced by 2 (e.g. ``[10, 10] -> [8, 8]``).

    JSON coordinates are 1-based ``[x, y]`` with origin at the **top-left** cell ``(1, 1)``: ``x`` east (column),
    ``y`` south (row). Same as ``[column, row]``. They are not rewritten—only ``dimensions`` shrink.

    Raises ``ValueError`` if the new size would be <2 or any coordinate lies outside the shrunk grid.
    """
    out = copy.deepcopy(data)
    maze = out["maze"]
    _swap_validation_v04_dimensions_if_raw(maze, str(out.get("task_id", "")))
    rows, cols = maze["dimensions"]
    if rows < 2 or cols < 2:
        raise ValueError("maze dimensions must be at least 2 to shrink by 2")
    nr, nc = rows - 2, cols - 2

    def bad_cell(col: int, row: int) -> bool:
        return not (1 <= row <= nr and 1 <= col <= nc)

    scol, srow = int(maze["start"][0]), int(maze["start"][1])
    gcol, grow = int(maze["goal"][0]), int(maze["goal"][1])
    if bad_cell(scol, srow) or bad_cell(gcol, grow):
        raise ValueError(f"start/goal outside shrunk grid x 1..{nc}, y 1..{nr}: start={maze['start']} goal={maze['goal']}")

    for w in maze["walls"]:
        wc, wr = int(w[0]), int(w[1])
        if bad_cell(wc, wr):
            raise ValueError(f"wall {w} outside shrunk grid ({nr}x{nc})")

    mech = out.get("mechanisms", {})
    for name in ("keys", "doors", "switches", "gates"):
        for item in mech.get(name, []):
            pos = item.get("position")
            if pos is None:
                continue
            wc, wr = int(pos[0]), int(pos[1])
            if bad_cell(wc, wr):
                raise ValueError(f"{name} position {pos} outside shrunk grid ({nr}x{nc})")

    g = out.get("goal")
    if isinstance(g, dict) and g.get("type") == "reach_position":
        t = g.get("target")
        if isinstance(t, (list, tuple)) and len(t) == 2:
            tc, tr = int(t[0]), int(t[1])
            if bad_cell(tc, tr):
                raise ValueError(f"goal.target {t} outside shrunk grid ({nr}x{nc})")

    maze["dimensions"] = [nr, nc]
    return out
