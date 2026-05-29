"""Hardcoded action plans for smoke tests (no external solver)."""

from __future__ import annotations

from interface.coords import FACING_ORDER, FACING_TO_DELTA


def v01_empty_room_trajectory() -> list[str]:
    """Reach goal (6,6) from start (1,1) facing EAST."""
    return [
        *["MOVE_FORWARD"] * 5,
        "TURN_RIGHT",
        *["MOVE_FORWARD"] * 5,
        "DONE",
    ]


def v04_single_key_trajectory() -> list[str]:
    """From pr1 ``smoke_bfs`` on V04; omits leading ``TURN_RIGHT`` (pr1 starts NORTH, MiniGrid EAST)."""
    return [
        "MOVE_FORWARD",
        "MOVE_FORWARD",
        "MOVE_FORWARD",
        "MOVE_FORWARD",
        "MOVE_FORWARD",
        "TURN_RIGHT",
        "MOVE_FORWARD",
        "MOVE_FORWARD",
        "MOVE_FORWARD",
        "MOVE_FORWARD",
        "TURN_RIGHT",
        "MOVE_FORWARD",
        "TURN_LEFT",
        "MOVE_FORWARD",
        "MOVE_FORWARD",
        "PICKUP",
        "TURN_LEFT",
        "MOVE_FORWARD",
        "TURN_LEFT",
        "MOVE_FORWARD",
        "MOVE_FORWARD",
        "MOVE_FORWARD",
        "MOVE_FORWARD",
        "MOVE_FORWARD",
        "TURN_RIGHT",
        "MOVE_FORWARD",
        "MOVE_FORWARD",
        "TOGGLE",
        "MOVE_FORWARD",
        "MOVE_FORWARD",
        "MOVE_FORWARD",
        "MOVE_FORWARD",
        "MOVE_FORWARD",
        "DONE",
    ]


def _turn_to_face(cur: str, target: str) -> list[str]:
    ci = FACING_ORDER.index(cur)
    ti = FACING_ORDER.index(target)
    diff = (ti - ci) % 4
    if diff == 0:
        return []
    if diff == 1:
        return ["TURN_RIGHT"]
    if diff == 2:
        return ["TURN_RIGHT", "TURN_RIGHT"]
    return ["TURN_LEFT"]


def plan_to_goal_from_prompt(
    row: int, col: int, facing: str, grow: int, gcol: int, budget: int = 6
) -> list[str]:
    actions: list[str] = []
    if col != gcol:
        target = "EAST" if gcol > col else "WEST"
        actions.extend(_turn_to_face(facing, target))
        actions.extend(["MOVE_FORWARD"] * min(abs(gcol - col), max(1, budget - len(actions))))
    elif row != grow:
        target = "SOUTH" if grow > row else "NORTH"
        actions.extend(_turn_to_face(facing, target))
        actions.extend(["MOVE_FORWARD"] * min(abs(grow - row), max(1, budget - len(actions))))
    else:
        actions.append("DONE")
    return actions[:budget] if actions else ["DONE"]
