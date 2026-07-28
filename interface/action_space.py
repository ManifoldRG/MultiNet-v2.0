"""Action-space vocabularies and cardinal→egocentric translation.

The runtime backend only understands the egocentric MiniGrid primitives
(``TURN_LEFT/TURN_RIGHT/MOVE_FORWARD/PICKUP/DROP/TOGGLE/DONE``). The *cardinal*
action space is a model-facing interface only: the model emits absolute moves
(``MOVE_NORTH`` …) plus ``PICKUP``/``DROP``/``INTERACT``/``DONE``, and the runner
expands each cardinal move into the minimal sequence of egocentric primitives
for the agent's current facing. Each emitted primitive costs one environment
step, so the two action spaces are directly comparable on step economy.
"""

from __future__ import annotations

from typing import Literal

from interface import parser as _parser

ActionSpace = Literal["egocentric", "cardinal"]

EGOCENTRIC_ACTIONS: tuple[str, ...] = _parser.ACTION_ORDER
CARDINAL_ACTIONS: tuple[str, ...] = (
    "MOVE_NORTH",
    "MOVE_SOUTH",
    "MOVE_EAST",
    "MOVE_WEST",
    "PICKUP",
    "DROP",
    "INTERACT",
    "DONE",
)

# Clockwise, matching interface.coords.FACING_ORDER and the env's turn-right.
_FACING_ORDER: tuple[str, ...] = ("NORTH", "EAST", "SOUTH", "WEST")
_MOVE_TO_FACING: dict[str, str] = {
    "MOVE_NORTH": "NORTH",
    "MOVE_SOUTH": "SOUTH",
    "MOVE_EAST": "EAST",
    "MOVE_WEST": "WEST",
}

_CARDINAL_SYNONYMS: dict[str, str] = {
    "move north": "MOVE_NORTH",
    "go north": "MOVE_NORTH",
    "north": "MOVE_NORTH",
    "move south": "MOVE_SOUTH",
    "go south": "MOVE_SOUTH",
    "south": "MOVE_SOUTH",
    "move east": "MOVE_EAST",
    "go east": "MOVE_EAST",
    "east": "MOVE_EAST",
    "move west": "MOVE_WEST",
    "go west": "MOVE_WEST",
    "west": "MOVE_WEST",
    "pick up": "PICKUP",
    "pickup": "PICKUP",
    "drop": "DROP",
    "drop key": "DROP",
    "put down": "DROP",
    "release": "DROP",
    "interact": "INTERACT",
    "done": "DONE",
    "finished": "DONE",
}


def valid_actions(space: ActionSpace) -> set[str]:
    """The token vocabulary the model may emit in this action space."""
    return set(CARDINAL_ACTIONS if space == "cardinal" else EGOCENTRIC_ACTIONS)


def actions_hint(space: ActionSpace) -> str:
    """Comma-joined action list for the ``Valid actions:`` prompt line."""
    tokens = CARDINAL_ACTIONS if space == "cardinal" else EGOCENTRIC_ACTIONS
    return ", ".join(tokens)


def synonyms(space: ActionSpace) -> dict[str, str]:
    """Phrase→token map for the parser's free-text regex fallback."""
    return dict(_CARDINAL_SYNONYMS) if space == "cardinal" else dict(_parser._SYNONYMS)


def _turns_between(current: str, target: str) -> list[str]:
    """Minimal turn primitives to rotate from ``current`` to ``target`` facing."""
    delta = (_FACING_ORDER.index(target) - _FACING_ORDER.index(current)) % 4
    if delta == 0:
        return []
    if delta == 1:
        return ["TURN_RIGHT"]
    if delta == 3:
        return ["TURN_LEFT"]
    return ["TURN_RIGHT", "TURN_RIGHT"]


def cardinal_to_primitives(token: str, facing: str) -> list[str]:
    """Expand one cardinal token into egocentric primitives for ``facing``.

    Egocentric tokens (including already-expanded primitives) pass through
    unchanged, so the runner can re-feed injected primitives idempotently.
    """
    if token in _MOVE_TO_FACING:
        return _turns_between(facing, _MOVE_TO_FACING[token]) + ["MOVE_FORWARD"]
    if token == "INTERACT":
        return ["TOGGLE"]
    return [token]
