"""Teleport, death and ice are reported to the model as what they are.

The world model is the single source of truth: ``classify_mechanic`` labels a
transition (teleported / died / froze / frozen) next to the rules, the env
exposes it as ``info["mechanic"]``, and feedback + the activity summary only
render it.
"""

from __future__ import annotations

import json
from dataclasses import replace

from gridworld.actions import MiniGridActions as A
from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.task_spec import TaskSpecification
from gridworld.world_model import TaskPlanningContext, classify_mechanic, successors
from interface.feedback import format_step_feedback, is_failed_step
from interface.config import ExperimentConfig
from interface.runner import build_runner
from interface.episode_log import _json_safe
from interface.observation import text_summary_history

PORTAL = {
    "id": "portal_purple",
    "position_a": [4, 1],
    "position_b": [6, 3],
    "color": "purple",
    "bidirectional": True,
}


def _spec(mechanisms: dict, goal=(7, 1)) -> TaskSpecification:
    return TaskSpecification.from_dict(
        {
            "task_id": "mechanic_feedback",
            "seed": 0,
            "difficulty_tier": 1,
            "maze": {"dimensions": [9, 5], "walls": [], "start": [1, 1], "goal": list(goal)},
            "mechanisms": mechanisms,
            "goal": {"type": "reach_position", "target": list(goal)},
            "max_steps": 60,
        }
    )


def teleport_spec() -> TaskSpecification:
    return _spec({"teleporters": [PORTAL]})


def ice_spec() -> TaskSpecification:
    return _spec({"frozen_tiles": [[2, 1]], "freeze_steps": 3})


def death_spec() -> TaskSpecification:
    return _spec(
        {"keys": [{"id": "kR", "position": [2, 1], "color": "red"}], "death_portals": [[4, 1]]}
    )


def _ride_spec(**extra) -> TaskSpecification:
    # Turntable at (2, 1) starts pointing EAST and spins clockwise after every
    # step, so after stepping onto it the next MOVE_FORWARD rides SOUTH to (2, 2).
    return _spec({"rotating_tiles": [[2, 1]], "rotating_initial_directions": [0], **extra})


def _play(spec: TaskSpecification, actions):
    """Apply ``actions`` through the world model; return (state, mechanics)."""
    ctx = TaskPlanningContext(spec)
    state = ctx.initial_state()
    mechanics = []
    for action in actions:
        transition = next(t for t in successors(ctx, state) if t.action == int(action))
        mechanics.append(
            classify_mechanic(ctx, state, int(action), transition.label, transition.next_state)
        )
        state = transition.next_state
    return state, mechanics


# --- classify_mechanic -------------------------------------------------------


def test_plain_moves_turns_and_pickups_are_not_mechanics():
    _, mechanics = _play(
        death_spec(), [A.MOVE_FORWARD, A.PICKUP, A.TURN_LEFT, A.TURN_RIGHT, A.MOVE_FORWARD]
    )
    assert mechanics == [None] * 5


def test_entering_a_portal_pad_is_a_teleport_to_its_partner():
    state, mechanics = _play(teleport_spec(), [A.MOVE_FORWARD] * 3)
    assert mechanics[:2] == [None, None]
    assert mechanics[2] == {
        "kind": "teleported",
        "from_xy": (4, 1),
        "to_xy": (6, 3),
        "color": "purple",
    }
    assert state.agent_pos == (6, 3)


def test_bidirectional_portal_teleports_back():
    ctx = TaskPlanningContext(teleport_spec())
    before = replace(ctx.initial_state(), agent_pos=(5, 3), agent_dir=0)
    transition = next(t for t in successors(ctx, before) if t.action == int(A.MOVE_FORWARD))
    mechanic = classify_mechanic(ctx, before, int(A.MOVE_FORWARD), transition.label, transition.next_state)
    assert mechanic == {"kind": "teleported", "from_xy": (6, 3), "to_xy": (4, 1), "color": "purple"}


def test_stepping_onto_ice_freezes_then_each_swallowed_action_counts_down():
    state, mechanics = _play(
        ice_spec(), [A.MOVE_FORWARD, A.MOVE_FORWARD, A.TURN_LEFT, A.DONE, A.TURN_LEFT]
    )
    assert mechanics == [
        {"kind": "froze", "cell_xy": (2, 1), "steps": 3},
        {"kind": "frozen", "remaining": 2},
        {"kind": "frozen", "remaining": 1},
        {"kind": "frozen", "remaining": 0},
        None,  # thawed: the turn takes effect again
    ]
    assert state.agent_pos == (2, 1)
    assert state.agent_dir == 3


def test_ice_with_zero_freeze_steps_is_not_a_mechanic():
    spec = _spec({"frozen_tiles": [[2, 1]], "freeze_steps": 0})
    _, mechanics = _play(spec, [A.MOVE_FORWARD, A.MOVE_FORWARD])
    assert mechanics == [None, None]


def test_death_while_carrying_a_key_resets_the_whole_maze():
    state, mechanics = _play(
        death_spec(), [A.MOVE_FORWARD, A.PICKUP, A.MOVE_FORWARD, A.MOVE_FORWARD]
    )
    assert mechanics[:3] == [None, None, None]
    assert mechanics[3] == {"kind": "died", "cell_xy": (4, 1), "start_xy": (1, 1), "start_dir": 0}
    ctx = TaskPlanningContext(death_spec())
    assert state == ctx.initial_state()


def test_death_reached_through_a_portal_names_the_death_tile_entered():
    # Not a valid spec (portal endpoint overlaps the death tile), but the
    # world model supports it, so the label must still say "died" there.
    spec = _spec(
        {
            "teleporters": [{**PORTAL, "position_a": [3, 1]}],
            "death_portals": [[6, 3]],
        }
    )
    state, mechanics = _play(spec, [A.MOVE_FORWARD, A.MOVE_FORWARD])
    assert mechanics == [
        None,
        {"kind": "died", "cell_xy": (6, 3), "start_xy": (1, 1), "start_dir": 0},
    ]
    assert state.agent_pos == (1, 1)


def test_turntable_ride_onto_a_portal_pad_teleports():
    spec = _ride_spec(teleporters=[{**PORTAL, "position_a": [2, 2]}])
    state, mechanics = _play(spec, [A.MOVE_FORWARD, A.MOVE_FORWARD])
    assert mechanics == [
        None,
        {"kind": "teleported", "from_xy": (2, 2), "to_xy": (6, 3), "color": "purple"},
    ]
    assert state.agent_pos == (6, 3)


def test_turntable_ride_onto_a_death_tile_dies():
    _, mechanics = _play(_ride_spec(death_portals=[[2, 2]]), [A.MOVE_FORWARD, A.MOVE_FORWARD])
    assert mechanics == [
        None,
        {"kind": "died", "cell_xy": (2, 2), "start_xy": (1, 1), "start_dir": 0},
    ]


def test_turntable_ride_onto_ice_freezes():
    _, mechanics = _play(
        _ride_spec(frozen_tiles=[[2, 2]], freeze_steps=2), [A.MOVE_FORWARD, A.MOVE_FORWARD]
    )
    assert mechanics == [None, {"kind": "froze", "cell_xy": (2, 2), "steps": 2}]


def test_plain_turntable_ride_and_waiting_on_it_are_not_mechanics():
    state, mechanics = _play(_ride_spec(), [A.MOVE_FORWARD, A.TURN_LEFT, A.MOVE_FORWARD])
    # Each step spins the arrow clockwise: SOUTH after boarding, WEST after the
    # turn, so the ride goes back WEST to the start.
    assert mechanics == [None, None, None]
    assert state.agent_pos == (1, 1)


# --- env exposes the label ---------------------------------------------------


def _env_infos(spec: TaskSpecification, actions):
    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(spec)
    backend.reset(seed=spec.seed)
    return [backend.step(int(a))[5] for a in actions]


def test_env_step_reports_the_mechanic_in_info_only_when_one_fired():
    infos = _env_infos(teleport_spec(), [A.TURN_LEFT, A.TURN_RIGHT, A.MOVE_FORWARD, A.MOVE_FORWARD, A.MOVE_FORWARD])
    assert all("mechanic" not in info for info in infos[:4])
    assert infos[4]["mechanic"] == {
        "kind": "teleported",
        "from_xy": (4, 1),
        "to_xy": (6, 3),
        "color": "purple",
    }


def test_env_step_reports_freeze_and_death():
    ice = _env_infos(ice_spec(), [A.MOVE_FORWARD, A.MOVE_FORWARD, A.MOVE_FORWARD, A.MOVE_FORWARD, A.MOVE_FORWARD])
    assert [info.get("mechanic") for info in ice] == [
        {"kind": "froze", "cell_xy": (2, 1), "steps": 3},
        {"kind": "frozen", "remaining": 2},
        {"kind": "frozen", "remaining": 1},
        {"kind": "frozen", "remaining": 0},
        None,
    ]
    death = _env_infos(death_spec(), [A.MOVE_FORWARD, A.PICKUP, A.MOVE_FORWARD, A.MOVE_FORWARD])
    assert [info.get("mechanic") for info in death] == [
        None,
        None,
        None,
        {"kind": "died", "cell_xy": (4, 1), "start_xy": (1, 1), "start_dir": 0},
    ]


# --- step feedback -------------------------------------------------------------


def _feedback(spec: TaskSpecification, actions, level="standard"):
    """(feedback, event_type, failed) per step, as the episode loop computes them."""
    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(spec)
    _, state, _ = backend.reset(seed=spec.seed)
    out = []
    for action in actions:
        name = A(int(action)).name
        _, reward, terminated, _, curr, info = backend.step(int(action))
        text, event_type = format_step_feedback(
            name, state, curr, reward, terminated, spec,
            level=level, mechanic=info.get("mechanic"),
        )
        failed = is_failed_step(name, state, curr, reward, terminated, spec)
        out.append((text, event_type, failed))
        state = curr
    return out


def test_teleport_feedback_names_portal_colour_and_both_ends():
    *_, last = _feedback(teleport_spec(), [A.MOVE_FORWARD] * 3)
    assert last == (
        "TELEPORTED — MOVE_FORWARD: Took the purple portal from (1, 4) to (3, 6).",
        "TELEPORTED",
        False,
    )


def test_death_feedback_says_the_whole_maze_reset():
    *_, last = _feedback(death_spec(), [A.MOVE_FORWARD, A.PICKUP, A.MOVE_FORWARD, A.MOVE_FORWARD])
    assert last == (
        "DIED — MOVE_FORWARD: Stepped on a death tile at (1, 4). The maze reset: "
        "you are back at the start tile (1, 1) facing EAST; keys, doors and "
        "switches are back where they started.",
        "DIED",
        False,
    )


def test_ice_feedback_counts_down_to_the_thaw():
    steps = _feedback(
        ice_spec(), [A.MOVE_FORWARD, A.MOVE_FORWARD, A.TURN_LEFT, A.DONE, A.TURN_LEFT]
    )
    assert [s[:2] for s in steps] == [
        ("MOVED — MOVE_FORWARD: Moved. The ice froze you for 3 steps.", "MOVED"),
        ("FROZEN — MOVE_FORWARD: You are frozen; the action had no effect. "
         "2 more steps until you thaw.", "FROZEN"),
        ("FROZEN — TURN_LEFT: You are frozen; the action had no effect. "
         "1 more step until you thaw.", "FROZEN"),
        ("FROZEN — DONE: You are frozen; the action had no effect. "
         "You have thawed; your next action will take effect.", "FROZEN"),
        ("TURNED — TURN_LEFT: Now facing NORTH.", "TURNED"),
    ]


def test_one_step_freeze_uses_the_singular():
    spec = _spec({"frozen_tiles": [[2, 1]], "freeze_steps": 1})
    steps = _feedback(spec, [A.MOVE_FORWARD, A.TURN_RIGHT])
    assert steps[0][0] == "MOVED — MOVE_FORWARD: Moved. The ice froze you for 1 step."
    assert steps[1][0] == (
        "FROZEN — TURN_RIGHT: You are frozen; the action had no effect. "
        "You have thawed; your next action will take effect."
    )


def test_minimal_feedback_is_the_new_event_type_alone():
    assert _feedback(teleport_spec(), [A.MOVE_FORWARD] * 3, level="minimal")[-1][:2] == (
        "TELEPORTED",
        "TELEPORTED",
    )
    assert [s[:2] for s in _feedback(ice_spec(), [A.MOVE_FORWARD] * 2, level="minimal")] == [
        ("MOVED", "MOVED"),
        ("FROZEN", "FROZEN"),
    ]
    death = _feedback(death_spec(), [A.MOVE_FORWARD] * 3, level="minimal")
    assert death[-1][:2] == ("DIED", "DIED")


def test_causal_feedback_matches_standard_for_mechanics():
    for spec, actions in (
        (teleport_spec(), [A.MOVE_FORWARD] * 3),
        (ice_spec(), [A.MOVE_FORWARD] * 3),
        (death_spec(), [A.MOVE_FORWARD] * 3),
    ):
        assert _feedback(spec, actions, level="causal") == _feedback(spec, actions)


def test_teleporting_onto_the_goal_is_still_success():
    *_, last = _feedback(_spec({"teleporters": [PORTAL]}, goal=(6, 3)), [A.MOVE_FORWARD] * 3)
    assert last == ("SUCCESS — MOVE_FORWARD: Reached the goal.", "DONE", False)


def test_failure_counting_is_unchanged_by_mechanic_labels():
    # Frozen: a swallowed MOVE_FORWARD used to be BLOCKED and a swallowed DONE
    # WRONG_DONE (failures); a swallowed turn / PICKUP / TOGGLE used to be
    # NOTHING (not a failure). The FROZEN label keeps exactly that.
    spec = _spec({"frozen_tiles": [[2, 1]], "freeze_steps": 6})
    steps = _feedback(
        spec,
        [A.MOVE_FORWARD, A.MOVE_FORWARD, A.TURN_LEFT, A.PICKUP, A.TOGGLE, A.DONE, A.TURN_RIGHT],
    )
    assert [(s[1], s[2]) for s in steps] == [
        ("MOVED", False),
        ("FROZEN", True),
        ("FROZEN", False),
        ("FROZEN", False),
        ("FROZEN", False),
        ("FROZEN", True),
        ("FROZEN", False),
    ]
    # Teleport and death were plain moves: not failures.
    assert _feedback(teleport_spec(), [A.MOVE_FORWARD] * 3)[-1][2] is False
    assert _feedback(death_spec(), [A.MOVE_FORWARD] * 3)[-1][2] is False
    # ...except a death that resets you onto the cell you just left, which the
    # mechanic-blind outcome saw as a wall bump (BLOCKED): still a failure.
    from_start = _feedback(_spec({"death_portals": [[2, 1]]}), [A.MOVE_FORWARD])
    assert from_start[0][1:] == ("DIED", True)


# --- episode loop ----------------------------------------------------------------


class _ScriptedAgent:
    def __init__(self, actions):
        self._actions = [A(int(a)).name for a in actions]
        self._i = 0
        self.last_usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}

    def __call__(self, messages):
        self.last_messages = messages
        action = self._actions[self._i] if self._i < len(self._actions) else "DONE"
        self._i += 1
        return f"FINAL_OUTPUT: {action}"


R1_CELL = dict(
    prompting="minimal",
    observation="image_only",
    context_window="text_summary_and_last_n",
    chat_history="stateless",
    in_context_learning="zero_shot",
    action_space="egocentric",
    querying="step_by_step",
    progress_stall_k=30,
)


def _run(spec: TaskSpecification, actions, **overrides):
    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(spec)
    runner = build_runner(ExperimentConfig(**{**R1_CELL, **overrides}), backend, spec)
    agent = _ScriptedAgent(actions)
    result = runner.run(agent, verbose=False)
    steps = [rec for rec in result["transcript"] if rec.get("kind") == "step"]
    return result, steps, agent


def test_runner_records_mechanic_events_and_keeps_failure_counting():
    _, steps, _ = _run(
        ice_spec(),
        [A.MOVE_FORWARD, A.MOVE_FORWARD, A.TURN_LEFT, A.DONE, A.TURN_LEFT],
        feedback="standard",
    )
    assert [s["event_type"] for s in steps[:5]] == ["MOVED", "FROZEN", "FROZEN", "FROZEN", "TURNED"]
    assert steps[1]["feedback"] == (
        "FROZEN — MOVE_FORWARD: You are frozen; the action had no effect. "
        "2 more steps until you thaw."
    )
    assert [s["consecutive_failures_after"] for s in steps[:5]] == [0, 1, 0, 1, 0]
    assert steps[0]["mechanic"] == {"kind": "froze", "cell_xy": (2, 1), "steps": 3}
    assert steps[3]["mechanic"] == {"kind": "frozen", "remaining": 0}
    assert "mechanic" not in steps[4]


def test_runner_reports_teleport_and_death_as_non_failures():
    _, steps, _ = _run(teleport_spec(), [A.MOVE_FORWARD] * 3, feedback="standard")
    assert steps[2]["event_type"] == "TELEPORTED"
    assert steps[2]["consecutive_failures_after"] == 0
    assert steps[2]["position_after_row_col"] == [3, 6]
    assert "mechanic" not in steps[0]

    _, steps, _ = _run(
        death_spec(), [A.MOVE_FORWARD, A.PICKUP, A.MOVE_FORWARD, A.MOVE_FORWARD]
    )
    assert [s["event_type"] for s in steps[:4]] == ["MOVED", "PICKUP", "MOVED", "DIED"]
    assert steps[3]["feedback"] == "DIED"  # R1 cell: minimal feedback
    assert steps[3]["consecutive_failures_after"] == 0
    assert steps[3]["mechanic"]["kind"] == "died"


# --- activity summary ------------------------------------------------------------


def _summary(spec: TaskSpecification, actions) -> str:
    result, _, _ = _run(spec, list(actions) + [A.TURN_LEFT])
    # The scripted TURN_LEFT pads the episode so the summary reflects ``actions``
    # only through the trail: turns never enter it.
    return text_summary_history(result["transcript"], spec)


def test_summary_reports_the_portal_and_restarts_the_trail():
    assert _summary(teleport_spec(), [A.MOVE_FORWARD] * 4) == (
        "You started at (1, 1) facing EAST.\n"
        "Activity summary:\n"
        "first you took the purple portal from (1, 4) to (3, 6), finally you passed (3, 7)"
    )


def test_summary_reports_one_freeze_event_not_each_swallowed_action():
    assert _summary(ice_spec(), [A.MOVE_FORWARD] * 5) == (
        "You started at (1, 1) facing EAST.\n"
        "Activity summary:\n"
        "first you were frozen at (1, 2) for 3 steps, finally you passed (1, 3)"
    )
    one_step = _spec({"frozen_tiles": [[2, 1]], "freeze_steps": 1})
    assert _summary(one_step, [A.MOVE_FORWARD]).endswith(
        "first you were frozen at (1, 2) for 1 step"
    )


def test_summary_keeps_events_before_a_death():
    assert _summary(
        death_spec(),
        [A.MOVE_FORWARD, A.PICKUP, A.MOVE_FORWARD, A.MOVE_FORWARD, A.MOVE_FORWARD],
    ) == (
        "You started at (1, 1) facing EAST.\n"
        "Activity summary:\n"
        "first you picked up the red key, then you stepped on a death tile at (1, 4) "
        "and the maze reset to the start tile (1, 1), finally you passed (1, 2)"
    )


def test_summary_survives_a_json_round_trip():
    result, _, _ = _run(teleport_spec(), [A.MOVE_FORWARD] * 4)
    reloaded = json.loads(json.dumps(_json_safe(result["transcript"]), default=str))
    assert text_summary_history(reloaded, teleport_spec()) == text_summary_history(
        result["transcript"], teleport_spec()
    )
