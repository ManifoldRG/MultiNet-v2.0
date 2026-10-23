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

from .generators import neighbors4
from .models import Coord


def _pos(cell: Coord) -> Position:
    return Position(cell[0], cell[1])


def _walls(task: TaskSpecification) -> set[Coord]:
    return {wall.to_tuple() for wall in task.maze.walls}


def _dims(task: TaskSpecification) -> tuple[int, int]:
    return task.maze.dimensions


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
    elif chain_type == "ksk":
        colors = palette.get("keys", ["red", "blue"])
        c1, c2 = colors[0], colors[1]
        switch_id = palette.get("switches", ["1"])[0]
        first_key, first_door = f"k{c1[0].upper()}", f"D{c1[0].upper()}"
        second_key, second_door = f"k{c2[0].upper()}", f"D{c2[0].upper()}"
        switch_name = f"s{switch_id}"
        gate_name = f"g{switch_id}"
        p3 = _slot(slots, "pickup_3_candidates")
        b3 = _slot(slots, "blocker_3_candidates")
        task.mechanisms.keys.append(KeySpec(id=first_key, position=_pos(p1), color=c1))
        task.mechanisms.doors.append(DoorSpec(id=first_door, position=_pos(b1), requires_key=c1))
        task.mechanisms.switches.append(
            SwitchSpec(id=switch_name, position=_pos(p2), controls=[gate_name], color="yellow")
        )
        task.mechanisms.gates.append(GateSpec(id=gate_name, position=_pos(b2)))
        task.mechanisms.keys.append(KeySpec(id=second_key, position=_pos(p3), color=c2))
        task.mechanisms.doors.append(DoorSpec(id=second_door, position=_pos(b3), requires_key=c2))
        _set_chain(
            task,
            "ksk",
            [
                ("key-door", first_key, first_door),
                ("switch-gate", switch_name, gate_name),
                ("key-door", second_key, second_door),
            ],
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


def _path_cells(task: TaskSpecification) -> list[Coord]:
    planned = plan_bfs_path(task)
    if not planned.success:
        return []
    cells: list[Coord] = []
    for pos in planned.positions:
        if not cells or cells[-1] != pos:
            cells.append(pos)
    return cells


def add_inactive_switch_distractor(task: TaskSpecification) -> None:
    """Place a blue switch that controls no gate, further along the path than the real switch.

    This is the benchmark D2 switch. A wrong-color key is added separately.
    """
    real = [switch for switch in task.mechanisms.switches if switch.controls]
    if not real:
        return
    real_pos = real[0].position.to_tuple()
    occupied = _occupied_cells(task)
    path = _path_cells(task)
    later: list[Coord] = []
    if real_pos in path:
        later = [cell for cell in path[path.index(real_pos) + 1 :] if cell not in occupied]
    if not later:
        later = [cell for cell in _find_free_open_cells(task)]
    if not later:
        return

    rng = random.Random(task.seed + 151)
    cell = rng.choice(later)
    task.mechanisms.switches.append(
        SwitchSpec(id="inactive_sb", position=_pos(cell), controls=[], color="blue")
    )
    if task.distractors is None:
        task.distractors = []
    task.distractors.append(
        Distractor(
            type="inactive_switch",
            element_id="inactive_sb",
            description="inactive blue switch that does not control any gate",
        )
    )
    task.metadata["inactive_switch"] = {
        "id": "inactive_sb",
        "position": list(cell),
        "position_rule": "after_real_switch",
    }


def add_wrong_key_distractors(task: TaskSpecification, colors: List[str]) -> None:
    """Place keys whose colors match no door, on open cells in the maze.

    The benchmark puts these on the corridor itself. Picking one up is safe
    because it opens no required door.
    """
    rng = random.Random(task.seed + 117)
    candidates = _find_free_open_cells(task)
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
    color: str = "yellow",
) -> None:
    """Put a key and door on an existing dead-end leaf. The walls stay put.

    Matches the benchmark D3 maps: the door sits on the leaf, and the key sits
    on the one open cell that leads into it. The pair is not on the required chain.
    """
    if chain_type != "kd":
        raise ValueError("V1 distractor_chain only supports chain_type='kd'")

    real_colors = {door.requires_key for door in task.mechanisms.doors}
    if color in real_colors:
        return

    open_cells = _open_cells(task)
    occupied = _occupied_cells(task)
    leaves: list[tuple[Coord, Coord]] = []
    for cell in open_cells:
        if cell in occupied:
            continue
        neighbors = [nb for nb in neighbors4(cell) if nb in open_cells]
        if len(neighbors) != 1:
            continue
        approach = neighbors[0]
        if approach in occupied:
            continue
        leaves.append((cell, approach))
    if not leaves:
        return

    rng = random.Random(task.seed + 129)
    leaf, approach = rng.choice(leaves)
    key_id = f"k{color[0].upper()}"
    door_id = f"D{color[0].upper()}"
    task.mechanisms.keys.append(KeySpec(id=key_id, position=_pos(approach), color=color))
    task.mechanisms.doors.append(
        DoorSpec(id=door_id, position=_pos(leaf), requires_key=color)
    )
    if task.distractors is None:
        task.distractors = []
    task.distractors.append(
        Distractor(
            type="distractor_chain",
            element_id="distractor_kd",
            description=f"unused {color} key-door in a dead end",
        )
    )
    task.metadata["distractor_chain"] = {
        "type": "kd",
        "color": color,
        "deadend_branch_leaf": list(leaf),
        "key": list(approach),
    }
