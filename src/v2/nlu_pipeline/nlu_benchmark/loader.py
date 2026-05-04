from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from automatic_maze_generation.mazegen.models import Door, Gate, Key, MazeInstance, Switch

from nlu_benchmark.env import GridState, GridWorldEnv


def _task_dict_to_env(data: dict[str, Any]) -> GridWorldEnv:
    maze = data["maze"]
    rows, cols = maze["dimensions"]
    walls = {tuple(w) for w in maze["walls"]}
    start = tuple(maze["start"])
    goal = tuple(maze["goal"])
    max_steps = data.get("max_steps", 100)
    mechanisms = data.get("mechanisms", {})
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


def grid_state_to_maze_instance(st: GridState) -> MazeInstance:
    def rc_to_xy(pos):
        r, c = pos
        # NLU grids are 1-based (row, col); mazegen solver uses 0-based (x, y).
        return (c - 1, r - 1)

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

    Row/column coordinates are **unchanged**. Use when the JSON names a larger grid than the
    coordinates actually use (common in the ogbench-style exports this repo ingests).

    Raises ``ValueError`` if the new size would be <2 or any coordinate lies outside the shrunk grid.
    """
    out = copy.deepcopy(data)
    maze = out["maze"]
    rows, cols = maze["dimensions"]
    if rows < 2 or cols < 2:
        raise ValueError("maze dimensions must be at least 2 to shrink by 2")
    nr, nc = rows - 2, cols - 2

    def bad_rc(r: int, c: int) -> bool:
        return not (1 <= r <= nr and 1 <= c <= nc)

    sr, sc = int(maze["start"][0]), int(maze["start"][1])
    gr, gc = int(maze["goal"][0]), int(maze["goal"][1])
    if bad_rc(sr, sc) or bad_rc(gr, gc):
        raise ValueError(f"start/goal outside shrunk grid 1..{nr} x 1..{nc}: start={maze['start']} goal={maze['goal']}")

    for w in maze["walls"]:
        r, c = int(w[0]), int(w[1])
        if bad_rc(r, c):
            raise ValueError(f"wall {w} outside shrunk grid ({nr}x{nc})")

    mech = out.get("mechanisms", {})
    for name in ("keys", "doors", "switches", "gates"):
        for item in mech.get(name, []):
            pos = item.get("position")
            if pos is None:
                continue
            r, c = int(pos[0]), int(pos[1])
            if bad_rc(r, c):
                raise ValueError(f"{name} position {pos} outside shrunk grid ({nr}x{nc})")

    g = out.get("goal")
    if isinstance(g, dict) and g.get("type") == "reach_position":
        t = g.get("target")
        if isinstance(t, (list, tuple)) and len(t) == 2:
            r, c = int(t[0]), int(t[1])
            if bad_rc(r, c):
                raise ValueError(f"goal.target {t} outside shrunk grid ({nr}x{nc})")

    maze["dimensions"] = [nr, nc]
    return out
