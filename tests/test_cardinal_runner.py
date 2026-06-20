"""End-to-end runner behavior for the cardinal action space.

A scripted agent emits cardinal tokens; the runner must expand each into the
egocentric primitives the backend understands, count every primitive as a step,
and record the originating cardinal token on each expanded step.
"""

from __future__ import annotations

from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.task_spec import TaskSpecification
from interface.config import ExperimentConfig
from interface.runner import build_runner


class ScriptedAgent:
    """Returns one scripted cardinal action per query (step_by_step)."""

    def __init__(self, actions):
        self._actions = list(actions)
        self._i = 0
        self.last_usage = {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}

    def __call__(self, messages):
        action = self._actions[self._i] if self._i < len(self._actions) else "DONE"
        self._i += 1
        return f"FINAL_OUTPUT: {action}"


def _steps(result):
    return [e for e in result["transcript"] if e.get("kind") == "step"]


def _run(spec_dict, actions, **config_kwargs):
    spec = TaskSpecification.from_dict(spec_dict)
    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(spec)
    runner = build_runner(
        ExperimentConfig(action_space="cardinal", **config_kwargs), backend, spec
    )
    return runner.run(ScriptedAgent(actions), verbose=False)


def test_cardinal_move_expands_with_turn_and_counts_each_primitive():
    # Agent starts at (1,1) facing EAST (dir 0); goal is two cells south at (1,3).
    spec = {
        "task_id": "cardinal_move",
        "seed": 0,
        "difficulty_tier": 1,
        "maze": {"dimensions": [5, 5], "walls": [], "start": [1, 1], "goal": [1, 3]},
        "mechanisms": {},
        "goal": {"type": "reach_position", "target": [1, 3]},
        "max_steps": 20,
    }

    result = _run(spec, ["MOVE_SOUTH", "MOVE_SOUTH", "DONE"])

    steps = _steps(result)
    executed = [s["action"] for s in steps]
    # MOVE_SOUTH while facing EAST -> TURN_RIGHT, MOVE_FORWARD; the second
    # MOVE_SOUTH (now facing SOUTH) -> MOVE_FORWARD, which lands on the goal cell
    # and auto-terminates (reach_position goal), so the scripted DONE never runs.
    assert executed == ["TURN_RIGHT", "MOVE_FORWARD", "MOVE_FORWARD"]
    assert result["success"] is True
    # Each primitive consumed one environment step.
    assert result["steps_used"] == 3
    # Provenance: every expanded primitive carries its originating cardinal token.
    assert [s.get("cardinal_action") for s in steps] == [
        "MOVE_SOUTH",
        "MOVE_SOUTH",
        "MOVE_SOUTH",
    ]


def test_cardinal_pickup_and_interact_open_a_locked_door():
    # Corridor: key at (2,1), locked red door at (3,1), goal at (5,1). Agent faces
    # EAST the whole way, so MOVE_EAST never needs a turn.
    spec = {
        "task_id": "cardinal_mechanics",
        "seed": 0,
        "difficulty_tier": 2,
        "maze": {"dimensions": [7, 3], "walls": [], "start": [1, 1], "goal": [5, 1]},
        "mechanisms": {
            "keys": [{"id": "k1", "position": [2, 1], "color": "red"}],
            "doors": [
                {
                    "id": "d1",
                    "position": [3, 1],
                    "requires_key": "red",
                    "initial_state": "locked",
                }
            ],
        },
        "goal": {"type": "reach_position", "target": [5, 1]},
        "max_steps": 30,
    }

    result = _run(
        spec,
        ["MOVE_EAST", "PICKUP", "INTERACT", "MOVE_EAST", "MOVE_EAST", "MOVE_EAST", "DONE"],
    )

    steps = _steps(result)
    executed = [s["action"] for s in steps]
    assert result["success"] is True
    # INTERACT translated to a TOGGLE that opened the door, tagged with its source.
    toggle_steps = [s for s in steps if s["action"] == "TOGGLE"]
    assert len(toggle_steps) == 1
    assert toggle_steps[0]["cardinal_action"] == "INTERACT"
    # Same-cell pickup happened (key collected) and the agent reached the goal.
    assert "PICKUP" in executed
    assert "k1" in result["final_state"]["collected_keys"]
