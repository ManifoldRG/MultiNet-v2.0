"""Map NLU parser action tokens to MiniGrid integer actions."""

from __future__ import annotations

from gridworld.actions import MiniGridActions

from interface.parser import VALID_ACTIONS

NLU_TO_MINIGRID: dict[str, int] = {
    "TURN_LEFT": MiniGridActions.TURN_LEFT,
    "TURN_RIGHT": MiniGridActions.TURN_RIGHT,
    "MOVE_FORWARD": MiniGridActions.MOVE_FORWARD,
    "PICKUP": MiniGridActions.PICKUP,
    "DROP": MiniGridActions.DROP,
    "TOGGLE": MiniGridActions.TOGGLE,
    "DONE": MiniGridActions.DONE,
}


def nlu_action_to_int(action: str) -> int:
    key = action.strip().upper()
    if key not in VALID_ACTIONS:
        raise ValueError(f"Unknown NLU action: {action!r}")
    return int(NLU_TO_MINIGRID[key])
