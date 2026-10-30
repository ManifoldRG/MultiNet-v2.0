"""Shared fixtures for the R1-rulebook tests on square and hex tilings."""

from gridworld.task_spec import TaskSpecification


def every_mechanic(tiling: str, **overrides) -> TaskSpecification:
    """A 9x9 serpentine whose only route (on either tiling: whole wall rows
    split it) takes the red key, opens the red door, crosses the ice, toggles
    the switch, passes the gate, rides the turntable and takes the portal over
    a death cell. A second death cell sits in a pocket next to the start.

        row 1  S . k . . . .       S start, k key
        row 2  # x # # # # D       x death pocket, D red door
        row 3  s . . * . . .       s switch, * ice
        row 4  G # # # # # #       G gate
        row 5  . . @ . . . .       @ turntable
        row 6  # # # # # # .
        row 7  g p . x . P .       g goal, p/P portal ends
    """
    d = {
        "task_id": f"every_mechanic_{tiling}",
        "seed": 3,
        "difficulty_tier": 1,
        "maze": {
            "dimensions": [9, 9],
            "walls": [[1, 2], [3, 2], [4, 2], [5, 2], [6, 2]]
            + [[x, 4] for x in range(2, 8)]
            + [[x, 6] for x in range(1, 7)],
            "start": [1, 1],
            "goal": [1, 7],
            "tiling": tiling,
        },
        "mechanisms": {
            "keys": [{"id": "kR", "position": [3, 1], "color": "red"}],
            "doors": [{"id": "DR", "position": [7, 2], "requires_key": "red", "initial_state": "locked"}],
            "switches": [{"id": "s1", "position": [1, 3], "controls": ["g1"], "color": "yellow"}],
            "gates": [{"id": "g1", "position": [1, 4], "color": "black"}],
            "teleporters": [{
                "id": "portal_cyan", "position_a": [6, 7], "position_b": [2, 7],
                "color": "cyan", "bidirectional": True,
            }],
            "death_portals": [[2, 2], [4, 7]],
            "frozen_tiles": [[4, 3]],
            "freeze_steps": 3,
            "rotating_tiles": [[3, 5]],
            "rotating_initial_directions": [2],
        },
        "goal": {"type": "reach_position", "target": [1, 7]},
        "max_steps": 400,
    }
    d.update(overrides)
    return TaskSpecification.from_dict(d)


class PlannerReplayAgent:
    """Replays the planner's optimal actions through the real runner, one maze
    after another, at zero API cost.

    ``plans`` are action-name lists in the order the pipeline runs the mazes.
    With context_window="full" every query after an episode's first resends
    the earlier steps under the full-history header, so a request without it
    opens the next episode: the agent moves to the next plan and refuses to if
    the previous one was not used up exactly.
    """

    def __init__(self, plans: list[list[str]]):
        self._plans = [list(plan) for plan in plans]
        self._current: list[str] | None = None
        self.episodes = 0
        self.last_usage = None

    def __call__(self, messages: list[dict]) -> str:
        from prompting_experiments.prompt_templates.user import FULL_HISTORY_USER_PROMPT

        self.last_usage = {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}
        content = messages[-1]["content"]
        texts = [content] if isinstance(content, str) else [b.get("text", "") for b in content]
        if not any(FULL_HISTORY_USER_PROMPT["header"] in t for t in texts):
            if self._current:
                raise AssertionError(f"episode ended with {len(self._current)} planned actions unused")
            self._current = self._plans.pop(0)
            self.episodes += 1
        if not self._current:
            raise AssertionError("the plan is used up but the episode goes on")
        return f"FINAL_OUTPUT: {self._current.pop(0)}"
