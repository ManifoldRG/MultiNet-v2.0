from gridworld.task_spec import TaskSpecification
from gridworld.backends.base import GridState
from gridworld.backends.minigrid_backend import MiniGridBackend
from interface.config import ExperimentConfig
from interface.runner import _progress_signature, build_runner


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


def _state(**kw):
    # GridState's only required fields are agent_position and
    # agent_direction (0=right/1=down/2=left/3=up); everything else
    # (agent_carrying, collected_keys, open_doors, active_switches,
    # open_gates, block_positions, observability_mode, explored_cells, ...)
    # defaults per gridworld/backends/base.py. agent_direction stands in
    # for "facing" here — _progress_signature must ignore it.
    base = dict(agent_position=(1, 1), agent_direction=0)
    base.update(kw)
    return GridState(**base)


def test_signature_ignores_facing():
    assert _progress_signature(_state(agent_direction=0)) == _progress_signature(_state(agent_direction=2))


def test_signature_changes_on_each_mechanism_axis():
    base = _progress_signature(_state())
    assert _progress_signature(_state(agent_carrying="kR")) != base
    assert _progress_signature(_state(collected_keys={"kR"})) != base
    assert _progress_signature(_state(open_doors={"DR"})) != base
    assert _progress_signature(_state(active_switches={"s1"})) != base
    assert _progress_signature(_state(open_gates={"g1"})) != base
    assert _progress_signature(_state(block_positions={"b1": (2, 2)})) != base


def _oscillate_spec():
    # 1-wide, 3-cell-interior corridor: start (1,1), oscillation cell (2,1),
    # goal (3,1) sits one further cell down the corridor so the scripted
    # bounce between (1,1) and (2,1) below never reaches it (a goal at
    # (2,1) would make the very first MOVE_FORWARD succeed immediately,
    # defeating the point of this fixture).
    return TaskSpecification.from_dict({
        "task_id": "osc", "seed": 0, "difficulty_tier": 1,
        "maze": {"dimensions": [5, 3], "walls": [], "start": [1, 1], "goal": [3, 1]},
        "mechanisms": {}, "goal": {"type": "reach_position", "target": [3, 1]},
        "max_steps": 200,
    })


def test_oscillator_stalls_at_exactly_k():
    # Move to a new cell once (novel), then bounce forever between two visited cells.
    actions = ["MOVE_FORWARD"] + ["TURN_LEFT", "TURN_LEFT", "MOVE_FORWARD",
                                  "TURN_LEFT", "TURN_LEFT", "MOVE_FORWARD"] * 50
    res = _run(_oscillate_spec(), actions, progress_stall_k=20)
    assert res["end_reason"] == "stalled"
    assert res["success"] is False
    # Observed: 1 novel move + 20 repeat-signature steps to trip K=20
    # (stall_count reaches K on the 20th repeat, i.e. step K+1 overall).
    assert res["steps_used"] == 21


def test_k_none_does_not_stall():
    actions = ["TURN_LEFT"] * 300  # spins forever
    res = _run(_oscillate_spec(), actions, progress_stall_k=None)
    assert res["end_reason"] != "stalled"


def test_turn_only_stalls_when_enabled():
    res = _run(_oscillate_spec(), ["TURN_LEFT"] * 300, progress_stall_k=20)
    assert res["end_reason"] == "stalled"


def test_survive_steps_rejects_watchdog():
    spec = TaskSpecification.from_dict({
        "task_id": "surv", "seed": 0, "difficulty_tier": 1,
        "maze": {"dimensions": [4, 4], "walls": [], "start": [1, 1], "goal": [2, 2]},
        "mechanisms": {}, "goal": {"type": "survive_steps"}, "max_steps": 50,
    })
    import pytest
    with pytest.raises(ValueError):
        _run(spec, ["TURN_LEFT"] * 5, progress_stall_k=20)


def test_explored_cells_only_counts_under_partial_observation():
    full = _state(observability_mode="full", explored_cells={(9, 9)})
    full2 = _state(observability_mode="full", explored_cells={(8, 8)})
    assert _progress_signature(full) == _progress_signature(full2)  # ignored when full
    fog = _state(observability_mode="fog_of_war", explored_cells={(9, 9)})
    fog2 = _state(observability_mode="fog_of_war", explored_cells={(9, 9), (8, 8)})
    assert _progress_signature(fog) != _progress_signature(fog2)   # counts under fog
