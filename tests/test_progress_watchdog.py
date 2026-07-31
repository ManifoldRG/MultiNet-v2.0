import numpy as np

from gridworld.task_spec import TaskSpecification
from gridworld.backends.base import GridState
from gridworld.backends.minigrid_backend import MiniGridBackend
from interface.config import ExperimentConfig
from interface.progress_watchdog import _progress_signature
from interface.runner import build_runner


class ScriptedAgent:
    def __init__(self, actions):
        self._a = list(actions)
        self._i = 0
        self.calls = 0
        self.last_usage = {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}

    def __call__(self, messages):
        self.calls += 1
        a = self._a[self._i] if self._i < len(self._a) else "DONE"
        self._i += 1
        return f"FINAL_OUTPUT: {a}"


class OneReplyAgent:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0
        self.last_usage = {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}

    def __call__(self, messages):
        self.calls += 1
        return self.reply


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


def _run_agent(spec, agent, **cfg):
    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(spec)
    runner = build_runner(ExperimentConfig(**cfg), backend, spec)
    return runner.run(agent, verbose=False)


def _run(spec, actions, **cfg):
    return _run_agent(spec, ScriptedAgent(actions), **cfg)


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
    res = _run(
        _activate_switch_spec(),
        ["MOVE_FORWARD", "TOGGLE"],
        progress_stall_k=1,
    )
    assert res["success"] is True
    assert res["end_reason"] == "success"


def test_backend_termination_without_goal_is_terminated_failure():
    # terminated_failure branch (interface/runner.py): the backend can end
    # the episode (terminated=True) without the goal being reached, e.g. a
    # lava hazard. event_type is "MOVED" here, not DONE/BLOCKED/WRONG_DONE/
    # INVALID, so this must be caught by the `if terminated:` fallback that
    # sets end_reason = "terminated_failure", not the success OR-branch.
    res = _run(
        _hazard_spec(),
        ["TURN_RIGHT", "MOVE_FORWARD"],
        progress_stall_k=2,
    )
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


def test_signature_distinguishes_same_color_keys_by_collected_id():
    first = _state(agent_carrying="red", collected_keys={"key_1"})
    second = _state(agent_carrying="red", collected_keys={"key_2"})

    assert _progress_signature(first) != _progress_signature(second)


def test_signature_normalizes_live_backend_numpy_positions():
    numpy_state = _state(
        agent_position=np.array([1, 1]),
        block_positions={"b1": np.array([2, 2])},
        observability_mode="view_cone",
        explored_cells={(1, 1), (1, 2)},
    )
    tuple_state = _state(
        agent_position=(1, 1),
        block_positions={"b1": (2, 2)},
        observability_mode="view_cone",
        explored_cells={(1, 1), (1, 2)},
    )

    signature = _progress_signature(numpy_state)

    assert signature == _progress_signature(tuple_state)
    _ = hash(signature)


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
    assert res["end_reason"] == "truncated"
    assert res["steps_used"] == 200


def test_turn_only_stalls_when_enabled():
    res = _run(_oscillate_spec(), ["TURN_LEFT"] * 300, progress_stall_k=20)
    assert res["end_reason"] == "stalled"
    assert res["steps_used"] == 20


def test_parse_failures_do_not_advance_stall_streak():
    agent = ScriptedAgent(["NOT_AN_ACTION", "STILL_NOT_AN_ACTION", "TURN_LEFT"])

    res = _run_agent(
        _oscillate_spec(),
        agent,
        progress_stall_k=1,
        max_parse_retries=5,
    )

    assert res["end_reason"] == "stalled"
    assert res["steps_used"] == 1
    assert res["query_count"] == 3
    assert agent.calls == 3


def test_survive_steps_rejects_watchdog():
    spec = TaskSpecification.from_dict({
        "task_id": "surv", "seed": 0, "difficulty_tier": 1,
        "maze": {"dimensions": [4, 4], "walls": [], "start": [1, 1], "goal": [2, 2]},
        "mechanisms": {}, "goal": {"type": "survive_steps"}, "max_steps": 50,
    })
    import pytest

    agent = ScriptedAgent(["TURN_LEFT"] * 5)
    with pytest.raises(ValueError):
        _run_agent(spec, agent, progress_stall_k=20)
    assert agent.calls == 0


def test_explored_cells_only_counts_under_partial_observation():
    full = _state(observability_mode="full", explored_cells={(9, 9)})
    full2 = _state(observability_mode="full", explored_cells={(8, 8)})
    assert _progress_signature(full) == _progress_signature(full2)  # ignored when full
    fog = _state(observability_mode="fog_of_war", explored_cells={(9, 9)})
    fog2 = _state(observability_mode="fog_of_war", explored_cells={(9, 9), (8, 8)})
    assert _progress_signature(fog) != _progress_signature(fog2)   # counts under fog


def test_key_pickup_gives_more_than_k_backtrack_budget():
    # Walk K+1 cells east to a key, pick it up, then retrace those same cells.
    # The collected-key/carrying regime makes every backtrack position novel,
    # so the watchdog must not kill this eventually successful policy.
    k = 3
    spec = TaskSpecification.from_dict({
        "task_id": "key_backtrack",
        "seed": 0,
        "difficulty_tier": 2,
        "maze": {
            "dimensions": [k + 4, 3],
            "walls": [],
            "start": [1, 1],
            "goal": [1, 1],
        },
        "mechanisms": {
            "keys": [{"id": "k1", "position": [k + 2, 1], "color": "red"}],
        },
        "goal": {"type": "reach_position", "target": [1, 1]},
        "max_steps": 40,
    })
    actions = (
        ["MOVE_FORWARD"] * (k + 1)
        + ["PICKUP", "TURN_LEFT", "TURN_LEFT"]
        + ["MOVE_FORWARD"] * (k + 1)
    )

    res = _run(spec, actions, progress_stall_k=k)

    assert res["success"] is True
    assert res["end_reason"] == "success"
    assert res["steps_used"] > k
    assert "k1" in res["final_state"]["collected_keys"]


def test_signature_distinguishes_moved_key_positions():
    # After a DROP the key lies somewhere the spec never put it; revisiting a
    # cell with the key layout changed is genuine progress, not a stall.
    pre = _state(agent_position=(2, 1), key_positions={"kR": (2, 1)})
    post = _state(agent_position=(2, 1), key_positions={"kR": (3, 1)})
    assert _progress_signature(pre) != _progress_signature(post)


def test_signature_normalizes_numpy_key_positions():
    numpy_state = _state(key_positions={"kR": np.array([2, 1])})
    tuple_state = _state(key_positions={"kR": (2, 1)})
    signature = _progress_signature(numpy_state)
    assert signature == _progress_signature(tuple_state)
    _ = hash(signature)


def test_drop_retrace_is_not_a_stall():
    # pickup -> walk -> DROP -> retrace over previously visited cells. After
    # the drop the agent's (position, carrying, collected) regime matches its
    # pre-pickup visits exactly; only the key's live position differs. The
    # watchdog must treat the retrace as novel, not as a stall.
    spec = TaskSpecification.from_dict({
        "task_id": "drop_retrace",
        "seed": 0,
        "difficulty_tier": 2,
        "maze": {"dimensions": [8, 4], "walls": [], "start": [1, 1], "goal": [1, 2]},
        "mechanisms": {
            "keys": [{"id": "kR", "position": [2, 1], "color": "red"}],
        },
        "goal": {"type": "reach_position", "target": [1, 2]},
        "max_steps": 40,
    })
    # East to the key, pick it up, one more cell east, drop it at (3,1), turn
    # around, retrace west over (2,1) and (1,1) — both visited pre-pickup with
    # empty hands — then turn south and finish on the goal. Without
    # key_positions in the signature the two retrace moves repeat the
    # pre-pickup signatures and stall_count hits K=3 right after the two
    # in-place turns, killing the episode mid-recovery.
    actions = (
        ["MOVE_FORWARD", "PICKUP", "MOVE_FORWARD", "DROP",
         "TURN_LEFT", "TURN_LEFT", "MOVE_FORWARD", "MOVE_FORWARD",
         "TURN_LEFT", "MOVE_FORWARD"]
    )

    res = _run(spec, actions, progress_stall_k=3)

    assert res["end_reason"] == "success"
    assert res["success"] is True
    assert res["steps_used"] == len(actions)


def test_pickup_drop_oscillation_still_stalls():
    # The inverse guarantee of test_drop_retrace_is_not_a_stall: PICKUP/DROP
    # cycled at one cell yields exactly two distinct signatures (held vs on
    # the ground), both seen after the first cycle. Signature novelty is
    # set-membership, so the loop cannot farm resets and the watchdog fires.
    spec = TaskSpecification.from_dict({
        "task_id": "drop_oscillation",
        "seed": 0,
        "difficulty_tier": 2,
        "maze": {"dimensions": [8, 4], "walls": [], "start": [1, 1], "goal": [6, 1]},
        "mechanisms": {
            "keys": [{"id": "kR", "position": [2, 1], "color": "red"}],
        },
        "goal": {"type": "reach_position", "target": [6, 1]},
        "max_steps": 60,
    })
    k = 4
    actions = ["MOVE_FORWARD"] + ["PICKUP", "DROP"] * (3 * k)

    res = _run(spec, actions, progress_stall_k=k)

    assert res["end_reason"] == "stalled"
    assert res["success"] is False
    # dies K steps after the last novel signature (the first full cycle),
    # far before the scripted oscillation runs out
    assert res["steps_used"] < len(actions)


def test_push_block_progress_and_terminal_success_precede_watchdog():
    spec = TaskSpecification.from_dict({
        "task_id": "block_progress",
        "seed": 0,
        "difficulty_tier": 2,
        "maze": {
            "dimensions": [6, 3],
            "walls": [],
            "start": [1, 1],
            "goal": [4, 1],
        },
        "mechanisms": {
            "blocks": [{"id": "b1", "position": [2, 1], "color": "grey"}],
        },
        "goal": {
            "type": "push_block_to",
            "target_ids": ["b1"],
            "target_positions": [[4, 1]],
        },
        "max_steps": 20,
    })

    res = _run(spec, ["MOVE_FORWARD", "MOVE_FORWARD"], progress_stall_k=1)

    assert res["success"] is True
    assert res["end_reason"] == "success"
    assert res["final_state"]["block_positions"]["b1"] == [4, 1]


def test_collect_all_terminal_success_precedes_watchdog():
    spec = TaskSpecification.from_dict({
        "task_id": "collect_all_terminal",
        "seed": 0,
        "difficulty_tier": 1,
        "maze": {
            "dimensions": [4, 4],
            "walls": [],
            "start": [1, 1],
            "goal": [2, 2],
        },
        "mechanisms": {
            "keys": [{"id": "k1", "position": [1, 1], "color": "red"}],
        },
        "goal": {"type": "collect_all", "target_ids": ["k1"]},
        "max_steps": 10,
    })

    res = _run(spec, ["PICKUP"], progress_stall_k=1)

    assert res["success"] is True
    assert res["end_reason"] == "success"
    assert res["steps_used"] == 1


def test_backend_truncation_precedes_simultaneous_k_threshold():
    spec = TaskSpecification.from_dict({
        "task_id": "truncate_at_k",
        "seed": 0,
        "difficulty_tier": 1,
        "maze": {
            "dimensions": [5, 5],
            "walls": [],
            "start": [1, 1],
            "goal": [3, 3],
        },
        "mechanisms": {},
        "goal": {"type": "reach_position", "target": [3, 3]},
        "max_steps": 1,
    })

    res = _run(spec, ["TURN_LEFT"], progress_stall_k=1)

    assert res["success"] is False
    assert res["end_reason"] == "truncated"
    assert res["steps_used"] == 1


def test_partial_observation_exploration_is_progress_until_reveals_stop():
    spec = TaskSpecification.from_dict({
        "task_id": "partial_turn_exploration",
        "seed": 0,
        "difficulty_tier": 2,
        "maze": {
            "dimensions": [9, 9],
            "walls": [],
            "start": [4, 4],
            "goal": [7, 7],
        },
        "mechanisms": {},
        "rules": {"observability": "view_cone", "view_size": 5},
        "goal": {"type": "reach_position", "target": [7, 7]},
        "max_steps": 20,
    })

    res = _run(spec, ["TURN_LEFT"] * 10, progress_stall_k=1)

    assert res["end_reason"] == "stalled"
    # More than one turn survived K=1 because each new view expanded explored_cells.
    assert res["steps_used"] > 1
    explored_sizes = [
        len(rec["state_after"]["explored_cells"])
        for rec in res["transcript"]
        if rec.get("kind") == "step"
    ]
    assert max(explored_sizes) > explored_sizes[0]
    assert explored_sizes[-1] == explored_sizes[-2]


def test_stall_discards_remaining_subgoal_queue_without_requerying():
    agent = OneReplyAgent(
        "SUB_GOAL: spin then move\n"
        "FINAL_OUTPUT: TURN_LEFT, TURN_LEFT, MOVE_FORWARD, MOVE_FORWARD"
    )

    res = _run_agent(
        _oscillate_spec(),
        agent,
        progress_stall_k=2,
        querying="subgoal",
    )

    steps = [rec for rec in res["transcript"] if rec.get("kind") == "step"]
    assert res["end_reason"] == "stalled"
    assert [rec["action"] for rec in steps] == ["TURN_LEFT", "TURN_LEFT"]
    assert agent.calls == 1
    assert res["query_count"] == 1


def test_stall_discards_remaining_cardinal_primitives_without_requerying():
    # Facing EAST, MOVE_WEST expands to TURN_RIGHT, TURN_RIGHT, MOVE_FORWARD.
    # K=1 fires on the first turn, so neither later primitive may execute.
    agent = OneReplyAgent("FINAL_OUTPUT: MOVE_WEST")

    res = _run_agent(
        _oscillate_spec(),
        agent,
        progress_stall_k=1,
        action_space="cardinal",
    )

    steps = [rec for rec in res["transcript"] if rec.get("kind") == "step"]
    assert res["end_reason"] == "stalled"
    assert [rec["action"] for rec in steps] == ["TURN_RIGHT"]
    assert steps[0]["cardinal_action"] == "MOVE_WEST"
    assert res["final_state"]["agent_position"] == [1, 1]
    assert agent.calls == 1
    assert res["query_count"] == 1
