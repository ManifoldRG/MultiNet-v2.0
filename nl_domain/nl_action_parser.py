"""Parse simple natural-language commands into MiniGrid action IDs."""

from __future__ import annotations

from gridworld.actions import MiniGridActions


class NLActionParser:
    """Small rule-based parser for smoke tests and manual demos."""

    _PHRASE_ACTIONS = (
        (("turn left", "rotate left"), int(MiniGridActions.TURN_LEFT)),
        (("turn right", "rotate right"), int(MiniGridActions.TURN_RIGHT)),
        (("go forward", "move forward", "walk ahead", "advance", "forward"), int(MiniGridActions.MOVE_FORWARD)),
        (("pick up", "pickup", "grab"), int(MiniGridActions.PICKUP)),
        (("drop", "release"), int(MiniGridActions.DROP)),
        (("toggle", "open", "press"), int(MiniGridActions.TOGGLE)),
        (("wait", "stay", "do nothing", "done"), int(MiniGridActions.DONE)),
    )

    _COMPASS_DIRECTIONS = {
        "east": 0,
        "south": 1,
        "west": 2,
        "north": 3,
    }

    def parse(self, command: str, agent_facing: int = 0) -> list[int]:
        """Return one or more MiniGrid action IDs for a short command."""
        text = command.strip().lower()
        if not text:
            return [int(MiniGridActions.DONE)]

        for direction, target_dir in self._COMPASS_DIRECTIONS.items():
            if f"move {direction}" in text or f"go {direction}" in text:
                return self._turns_to_direction(agent_facing, target_dir) + [
                    int(MiniGridActions.MOVE_FORWARD)
                ]

        parts = [part.strip() for part in text.replace(",", " then ").split(" then ") if part.strip()]
        actions = []
        for part in parts or [text]:
            actions.extend(self._parse_single(part))
        return actions or [int(MiniGridActions.DONE)]

    def _parse_single(self, text: str) -> list[int]:
        """Parse one command fragment, such as ``turn left``."""
        for phrases, action in self._PHRASE_ACTIONS:
            if any(phrase in text for phrase in phrases):
                return [action]
        return [int(MiniGridActions.DONE)]

    def _turns_to_direction(self, current_dir: int, target_dir: int) -> list[int]:
        """Generate the shortest turn sequence before moving forward."""
        diff = (target_dir - current_dir) % 4
        if diff == 0:
            return []
        if diff == 1:
            return [int(MiniGridActions.TURN_RIGHT)]
        if diff == 2:
            return [int(MiniGridActions.TURN_RIGHT), int(MiniGridActions.TURN_RIGHT)]
        return [int(MiniGridActions.TURN_LEFT)]
