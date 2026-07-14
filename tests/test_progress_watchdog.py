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


def _activate_switch_spec():
    # Same spec as
    # tests/test_backend_integration.py::test_minigrid_activate_switch_goal_terminates_from_toggle_branch,
    # which shows the switch goal terminates (reward>0, goal_reached=True) on
    # a TOGGLE step whose event_type is "TOGGLED", never "DONE".
    return TaskSpecification.from_dict({
        "task_id": "activate_switch_goal",
        "seed": 13,
        "difficulty_tier": 2,
        "maze": {"dimensions": [5, 5], "walls": [], "start": [1, 1], "goal": [3, 3]},
        "mechanisms": {"switches": [{"id": "s1", "position": [2, 1], "controls": []}]},
        "goal": {"type": "activate_switch", "target_ids": ["s1"]},
        "max_steps": 20,
    })


def _hazard_spec():
    # A lava hazard placed one step south of the start. Verified directly
    # against MiniGridBackend: MOVE_FORWARD onto the hazard cell yields
    # terminated=True, reward=0, goal_reached=False, event_type="MOVED"
    # (never DONE/BLOCKED/WRONG_DONE/INVALID) — a genuine backend
    # termination without reaching the goal.
    return TaskSpecification.from_dict({
        "task_id": "hazard_fail",
        "seed": 0,
        "difficulty_tier": 1,
        "maze": {"dimensions": [5, 5], "walls": [], "start": [1, 1], "goal": [3, 3]},
        "mechanisms": {"hazards": [{"id": "h1", "position": [1, 2], "hazard_type": "lava"}]},
        "goal": {"type": "reach_position", "target": [3, 3]},
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


def test_goal_reached_from_non_done_event_still_succeeds():
    # OR-success branch (interface/runner.py): a backend goal can terminate
    # the episode from an action other than DONE. Here TOGGLE activates the
    # switch goal and terminates with reward>0/goal_reached=True while
    # event_type is "TOGGLED" — success must come from
    # `terminated and state.goal_reached`, not from event_type == "DONE".
    res = _run(_activate_switch_spec(), ["MOVE_FORWARD", "TOGGLE"])
    assert res["success"] is True
    assert res["end_reason"] == "success"


def test_backend_termination_without_goal_is_terminated_failure():
    # terminated_failure branch (interface/runner.py): the backend can end
    # the episode (terminated=True) without the goal being reached, e.g. a
    # lava hazard. event_type is "MOVED" here, not DONE/BLOCKED/WRONG_DONE/
    # INVALID, so this must be caught by the `if terminated:` fallback that
    # sets end_reason = "terminated_failure", not the success OR-branch.
    res = _run(_hazard_spec(), ["TURN_RIGHT", "MOVE_FORWARD"])
    assert res["success"] is False
    assert res["end_reason"] == "terminated_failure"
