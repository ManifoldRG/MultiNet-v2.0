from gridworld.task_spec import TaskSpecification
from gridworld.backends.minigrid_backend import MiniGridBackend
from interface.config import ExperimentConfig
from interface.runner import build_runner


class ScriptedAgent:
    def __init__(self, actions):
        self._a = list(actions)
        self._i = 0
        self.last_usage = {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}

    def __call__(self, messages):
        a = self._a[self._i] if self._i < len(self._a) else "DONE"
        self._i += 1
        return f"FINAL_OUTPUT: {a}"


def _reach_spec():
    return TaskSpecification.from_dict({
        "task_id": "reach_2s", "seed": 0, "difficulty_tier": 1,
        "maze": {"dimensions": [5, 5], "walls": [], "start": [1, 1], "goal": [1, 3]},
        "mechanisms": {}, "goal": {"type": "reach_position", "target": [1, 3]},
        "max_steps": 40,
    })


def _run(spec, actions, **cfg):
    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(spec)
    runner = build_runner(ExperimentConfig(**cfg), backend, spec)
    return runner.run(ScriptedAgent(actions), verbose=False)


def test_reach_position_done_still_succeeds():
    # Regression: the existing DONE->success path is unchanged for reach_position.
    # Agent starts at (1,1) facing EAST; goal (1,3) is two cells south, so
    # TURN_RIGHT (EAST->SOUTH) then two MOVE_FORWARDs lands on the goal cell
    # (see tests/test_cardinal_runner.py::test_cardinal_move_expands_with_turn_and_counts_each_primitive
    # for the identical spec's documented facing/path). The scripted trailing
    # DONE is never reached because MOVE_FORWARD onto the goal cell already
    # terminates the episode with event_type "DONE" (interface/feedback.py).
    res = _run(_reach_spec(), ["TURN_RIGHT", "MOVE_FORWARD", "MOVE_FORWARD", "DONE"])
    assert res["success"] is True
    assert res["end_reason"] == "success"
