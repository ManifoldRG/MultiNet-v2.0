from __future__ import annotations

from typing import List

import random

from gridworld.baselines import plan_bfs_path
from gridworld.task_spec import (
    DependencyChain,
    DependencyStep,
    Distractor,
    DoorSpec,
    GateSpec,
    KeySpec,
    Position,
    SwitchSpec,
    TaskSpecification,
)

from .generators import in_bounds, neighbors4
from .models import Coord


def _pos(cell: Coord) -> Position:
    return Position(cell[0], cell[1])


def _walls(task: TaskSpecification) -> set[Coord]:
    return {wall.to_tuple() for wall in task.maze.walls}


def _set_walls(task: TaskSpecification, walls: set[Coord]) -> None:
    task.maze.walls = [Position(x, y) for x, y in sorted(walls)]


def _dims(task: TaskSpecification) -> tuple[int, int]:
    return task.maze.dimensions


def _interior(cell: Coord, width: int, height: int) -> bool:
    x, y = cell
    return 0 < x < width - 1 and 0 < y < height - 1


def _slot(slots: dict, name: str) -> Coord:
    values = slots.get(name) or []
    if not values or values[0] is None:
        raise ValueError(f"Missing {name} slot")
    return values[0]


def _set_chain(task: TaskSpecification, notation: str, steps: list[tuple[str, str, str]]) -> None:
    task.dependency_chain = DependencyChain(
        depth=len(steps),
        sequence=[
            DependencyStep(step=index + 1, type=kind, element=element, unlocks=unlocks)
            for index, (kind, element, unlocks) in enumerate(steps)
        ],
        notation=notation,
    )
    task.metadata["chain_pattern"] = notation


def place_key_door_pair(task: TaskSpecification, slots: dict, color: str = "red") -> None:
    key_pos = _slot(slots, "pickup_1_candidates")
    door_pos = _slot(slots, "blocker_1_candidates")
    key_id = f"k{color[0].upper()}"
    door_id = f"D{color[0].upper()}"
    task.mechanisms.keys.append(KeySpec(id=key_id, position=_pos(key_pos), color=color))
    task.mechanisms.doors.append(
        DoorSpec(id=door_id, position=_pos(door_pos), requires_key=color)
    )
    _set_chain(task, "kd", [("key-door", key_id, door_id)])


def place_switch_gate_pair(task: TaskSpecification, slots: dict, switch_id: str = "1") -> None:
    switch_pos = _slot(slots, "pickup_1_candidates")
    gate_pos = _slot(slots, "blocker_1_candidates")
    switch_name = f"s{switch_id}"
    gate_name = f"g{switch_id}"
    task.mechanisms.switches.append(
        SwitchSpec(id=switch_name, position=_pos(switch_pos), controls=[gate_name])
    )
    task.mechanisms.gates.append(GateSpec(id=gate_name, position=_pos(gate_pos)))
    _set_chain(task, "sg", [("switch-gate", switch_name, gate_name)])


def place_sequential_chain(task: TaskSpecification, slots: dict, chain_type: str, palette: dict) -> None:
    p1 = _slot(slots, "pickup_1_candidates")
    b1 = _slot(slots, "blocker_1_candidates")
    p2 = _slot(slots, "pickup_2_candidates")
    b2 = _slot(slots, "blocker_2_candidates")

    if chain_type == "ks":
        key_color = palette.get("keys", ["red"])[0]
        switch_id = palette.get("switches", ["1"])[0]
        key_id = f"k{key_color[0].upper()}"
        door_id = f"D{key_color[0].upper()}"
        switch_name = f"s{switch_id}"
        gate_name = f"g{switch_id}"
        task.mechanisms.keys.append(KeySpec(id=key_id, position=_pos(p1), color=key_color))
        task.mechanisms.doors.append(DoorSpec(id=door_id, position=_pos(b1), requires_key=key_color))
        task.mechanisms.switches.append(
            SwitchSpec(id=switch_name, position=_pos(p2), controls=[gate_name])
        )
        task.mechanisms.gates.append(GateSpec(id=gate_name, position=_pos(b2)))
        _set_chain(
            task,
            "ks",
            [("key-door", key_id, door_id), ("switch-gate", switch_name, gate_name)],
        )
    elif chain_type == "sk":
        key_color = palette.get("keys", ["red"])[0]
        switch_id = palette.get("switches", ["1"])[0]
        key_id = f"k{key_color[0].upper()}"
        door_id = f"D{key_color[0].upper()}"
        switch_name = f"s{switch_id}"
        gate_name = f"g{switch_id}"
        task.mechanisms.switches.append(
            SwitchSpec(id=switch_name, position=_pos(p1), controls=[gate_name])
        )
        task.mechanisms.gates.append(GateSpec(id=gate_name, position=_pos(b1)))
        task.mechanisms.keys.append(KeySpec(id=key_id, position=_pos(p2), color=key_color))
        task.mechanisms.doors.append(DoorSpec(id=door_id, position=_pos(b2), requires_key=key_color))
        _set_chain(
            task,
            "sk",
            [("switch-gate", switch_name, gate_name), ("key-door", key_id, door_id)],
        )
    elif chain_type == "kk":
        colors = palette.get("keys", ["red", "blue"])
        c1, c2 = colors[0], colors[1]
        first_key, first_door = f"k{c1[0].upper()}", f"D{c1[0].upper()}"
        second_key, second_door = f"k{c2[0].upper()}", f"D{c2[0].upper()}"
        task.mechanisms.keys.append(KeySpec(id=first_key, position=_pos(p1), color=c1))
        task.mechanisms.doors.append(DoorSpec(id=first_door, position=_pos(b1), requires_key=c1))
        task.mechanisms.keys.append(KeySpec(id=second_key, position=_pos(p2), color=c2))
        task.mechanisms.doors.append(DoorSpec(id=second_door, position=_pos(b2), requires_key=c2))
        _set_chain(
            task,
            "kk",
            [("key-door", first_key, first_door), ("key-door", second_key, second_door)],
        )
    else:
        raise ValueError(f"Unsupported chain_type: {chain_type}")


def _open_cells(task: TaskSpecification) -> set[Coord]:
    width, height = _dims(task)
    walls = _walls(task)
    return {(x, y) for x in range(width) for y in range(height) if (x, y) not in walls}


def _occupied_cells(task: TaskSpecification) -> set[Coord]:
    occupied = {task.maze.start.to_tuple(), task.maze.goal.to_tuple()}
    for group in (
        task.mechanisms.keys,
        task.mechanisms.doors,
        task.mechanisms.switches,
        task.mechanisms.gates,
    ):
        occupied.update(obj.position.to_tuple() for obj in group)
    return occupied


def _find_free_open_cells(task: TaskSpecification) -> list[Coord]:
    occupied = _occupied_cells(task)
    return [cell for cell in _open_cells(task) if cell not in occupied]


def _find_dead_end_attachment_candidates(task: TaskSpecification) -> list[Coord]:
    open_cells = _open_cells(task)
    occupied = _occupied_cells(task)
    width, height = _dims(task)
    walls = _walls(task)
    candidates = []
    for cell in open_cells:
        if cell in occupied:
            continue
        open_nbs = [nb for nb in neighbors4(cell) if nb in open_cells]
        wall_nbs = [
            nb
            for nb in neighbors4(cell)
            if in_bounds(nb, width, height) and nb in walls and _interior(nb, width, height)
        ]
        if len(open_nbs) >= 1 and len(wall_nbs) >= 1:
            candidates.append(cell)
    return candidates


def _carve_dead_end_branch(task: TaskSpecification, attach: Coord, length: int = 2) -> bool:
    """Carve a straight interior dead-end off an open cell. The border stays walled."""
    walls = _walls(task)
    open_cells = _open_cells(task)
    width, height = _dims(task)
    x, y = attach

    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        branch = []
        cx, cy = x, y
        ok = True
        for _ in range(length):
            cx += dx
            cy += dy
            cell = (cx, cy)
            if not _interior(cell, width, height) or cell not in walls or cell in open_cells:
                ok = False
                break
            branch.append(cell)
        if ok:
            for cell in branch:
                walls.discard(cell)
            _set_walls(task, walls)
            return True
    return False


def add_dead_end_distractors(
    task: TaskSpecification,
    count: int = 1,
    branch_length: int = 2,
) -> None:
    rng = random.Random(task.seed + 101)
    candidates = _find_dead_end_attachment_candidates(task)
    rng.shuffle(candidates)

    added = 0
    for attach in candidates:
        if added >= count:
            break
        if _carve_dead_end_branch(task, attach, length=branch_length):
            added += 1

    task.metadata["dead_end_distractors"] = {
        "count": added,
        "branch_length": branch_length,
    }


def add_wrong_key_distractors(task: TaskSpecification, colors: List[str]) -> None:
    """Place keys whose colors match no door, off the current gridworld plan."""
    rng = random.Random(task.seed + 117)
    planned = plan_bfs_path(task)
    optimal_path = set(planned.positions) if planned.success else set()

    candidates = [cell for cell in _find_free_open_cells(task) if cell not in optimal_path]
    rng.shuffle(candidates)

    real_required_colors = {door.requires_key for door in task.mechanisms.doors}
    distractor_colors = [color for color in colors if color not in real_required_colors]
    if task.distractors is None:
        task.distractors = []

    added_colors: list[str] = []
    for color, cell in zip(distractor_colors, candidates):
        key_id = f"k{color[0].upper()}"
        task.mechanisms.keys.append(KeySpec(id=key_id, position=_pos(cell), color=color))
        task.distractors.append(
            Distractor(
                type="wrong_color_key",
                element_id=key_id,
                description=f"unused {color} key",
            )
        )
        added_colors.append(color)

    task.metadata["wrong_key_distractors"] = added_colors


def add_distractor_chain(
    task: TaskSpecification,
    chain_type: str = "kd",
    color: str = "green",
) -> None:
    """Add a side key-door branch that is not part of the required chain."""
    if chain_type != "kd":
        raise ValueError("V1 distractor_chain only supports chain_type='kd'")

    rng = random.Random(task.seed + 129)
    candidates = _find_dead_end_attachment_candidates(task)
    rng.shuffle(candidates)

    real_colors = {door.requires_key for door in task.mechanisms.doors}
    if color in real_colors:
        return

    width, height = _dims(task)
    walls = _walls(task)
    for attach in candidates:
        x, y = attach
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            c1 = (x + dx, y + dy)
            c2 = (x + 2 * dx, y + 2 * dy)
            c3 = (x + 3 * dx, y + 3 * dy)
            branch = [c1, c2, c3]
            if not all(_interior(cell, width, height) and cell in walls for cell in branch):
                continue

            for cell in branch:
                walls.discard(cell)
            _set_walls(task, walls)

            key_id = f"k{color[0].upper()}"
            door_id = f"D{color[0].upper()}"
            task.mechanisms.keys.append(KeySpec(id=key_id, position=_pos(c1), color=color))
            task.mechanisms.doors.append(
                DoorSpec(id=door_id, position=_pos(c2), requires_key=color)
            )
            if task.distractors is None:
                task.distractors = []
            task.distractors.append(
                Distractor(
                    type="distractor_chain",
                    element_id="distractor_kd",
                    description=f"unused {color} key-door branch",
                )
            )
            task.metadata["distractor_chain"] = {
                "type": "kd",
                "color": color,
                "cells": [list(c1), list(c2), list(c3)],
            }
            return
