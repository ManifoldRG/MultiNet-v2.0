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
        {"kind": "rotating", "cell_xy": (2, 1), "pointing": 1, "stayed": False},
        {"kind": "teleported", "from_xy": (2, 2), "to_xy": (6, 3), "color": "purple"},
    ]
    assert state.agent_pos == (6, 3)


def test_turntable_ride_onto_a_death_tile_dies():
    _, mechanics = _play(_ride_spec(death_portals=[[2, 2]]), [A.MOVE_FORWARD, A.MOVE_FORWARD])
    assert mechanics == [
        {"kind": "rotating", "cell_xy": (2, 1), "pointing": 1, "stayed": False},
        {"kind": "died", "cell_xy": (2, 2), "start_xy": (1, 1), "start_dir": 0},
    ]


def test_turntable_ride_onto_ice_freezes():
    _, mechanics = _play(
        _ride_spec(frozen_tiles=[[2, 2]], freeze_steps=2), [A.MOVE_FORWARD, A.MOVE_FORWARD]
    )
    assert mechanics == [{"kind": "rotating", "cell_xy": (2, 1), "pointing": 1, "stayed": False}, {"kind": "froze", "cell_xy": (2, 2), "steps": 2}]


def test_stepping_onto_a_turntable_then_riding_it():
    # Turntable at (2, 1) starts EAST and spins clockwise every step: stepping
    # onto it leaves it pointing SOUTH, so the next MOVE_FORWARD rides SOUTH.
    _, mechanics = _play(_ride_spec(), [A.MOVE_FORWARD, A.MOVE_FORWARD])
    assert mechanics == [
        {"kind": "rotating", "cell_xy": (2, 1), "pointing": 1, "stayed": False},
        {"kind": "carried", "from_xy": (2, 1), "to_xy": (2, 2), "direction": 1},
    ]


def test_waiting_on_a_turntable_is_not_a_mechanic_and_a_blocked_ride_stays():
    # On the tile (SOUTH), two turns spin it to NORTH, into the border wall.
    _, mechanics = _play(
        _ride_spec(), [A.MOVE_FORWARD, A.TURN_LEFT, A.TURN_LEFT, A.MOVE_FORWARD]
    )
    assert mechanics[1:] == [
        None,
        None,
        {"kind": "rotating", "cell_xy": (2, 1), "pointing": 0, "stayed": True},
    ]


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
        failed = is_failed_step(
            name, state, curr, reward, terminated, spec, mechanic=info.get("mechanic")
        )
        out.append((text, event_type, failed))
        state = curr
    return out


def test_teleport_feedback():
    *_, last = _feedback(teleport_spec(), [A.MOVE_FORWARD] * 3)
    assert last[:2] == ("TELEPORTED — MOVE_FORWARD: You teleported to (3,6).", "TELEPORTED")


def test_death_feedback():
    *_, last = _feedback(death_spec(), [A.MOVE_FORWARD, A.PICKUP, A.MOVE_FORWARD, A.MOVE_FORWARD])
    assert last[:2] == (
        "DEATH RESET — MOVE_FORWARD: Stepped on a death portal and lost your progress.",
        "DEATH RESET",
    )


def test_death_taken_from_the_start_cell_is_reported_as_a_death():
    # The reset lands on the cell the agent left, so positions alone look like a bump.
    (only,) = _feedback(_spec({"death_portals": [[2, 1]]}), [A.MOVE_FORWARD])
    assert only[:2] == (
        "DEATH RESET — MOVE_FORWARD: Stepped on a death portal and lost your progress.",
        "DEATH RESET",
    )


def test_freeze_feedback_standard_and_causal():
    actions = [A.MOVE_FORWARD] * 4  # land on ice (3 steps), then 3 swallowed moves
    assert [s[0] for s in _feedback(ice_spec(), actions)] == [
        "FROZEN — MOVE_FORWARD: You stepped on a frozen tile.",
        "FROZEN — MOVE_FORWARD: This action did nothing, wait until the freeze ends.",
        "FROZEN — MOVE_FORWARD: This action did nothing, wait until the freeze ends.",
        "FROZEN — MOVE_FORWARD: This action did nothing. The freeze has ended.",
    ]
    assert [s[0] for s in _feedback(ice_spec(), actions, level="causal")] == [
        "FROZEN — MOVE_FORWARD: You stepped on a frozen tile. You are frozen for 3 steps.",
        "FROZEN — MOVE_FORWARD: This action did nothing. Frozen for 2 more steps.",
        "FROZEN — MOVE_FORWARD: This action did nothing. Frozen for 1 more steps.",
        "FROZEN — MOVE_FORWARD: This action did nothing. The freeze has ended.",
    ]


def test_rotating_tile_feedback():
    steps = _feedback(_ride_spec(), [A.MOVE_FORWARD, A.MOVE_FORWARD])
    assert [s[:2] for s in steps] == [
        ("ROTATING — MOVE_FORWARD: You stepped on a rotating tile. It points SOUTH.", "ROTATING"),
        ("CARRIED — MOVE_FORWARD: The rotating tile carried you SOUTH.", "CARRIED"),
    ]
    blocked = _feedback(_ride_spec(), [A.MOVE_FORWARD, A.TURN_LEFT, A.TURN_LEFT, A.MOVE_FORWARD])
    assert blocked[-1][:2] == (
        "ROTATING — MOVE_FORWARD: This action did not carry you off the rotating tile. "
        "It now points EAST.",
        "ROTATING",
    )


def test_minimal_feedback_carries_the_direction_or_destination():
    assert [s[0] for s in _feedback(_ride_spec(), [A.MOVE_FORWARD] * 2, level="minimal")] == [
        "ROTATING SOUTH",
        "CARRIED SOUTH",
    ]
    assert _feedback(teleport_spec(), [A.MOVE_FORWARD] * 3, level="minimal")[-1][0] == (
        "TELEPORTED (3,6)"
    )
    assert _feedback(ice_spec(), [A.MOVE_FORWARD] * 2, level="minimal")[-1][0] == "FROZEN"
    assert _feedback(
        _spec({"death_portals": [[2, 1]]}), [A.MOVE_FORWARD], level="minimal"
    )[0][0] == "DEATH RESET"


def test_a_ride_onto_a_teleporter_reports_the_teleport():
    # Turntable at (2, 1) points SOUTH after the first step; a portal pad sits at (2, 2).
    spec = _ride_spec(teleporters=[{**PORTAL, "position_a": [2, 2]}])
    *_, last = _feedback(spec, [A.MOVE_FORWARD, A.MOVE_FORWARD])
    assert last[1] == "TELEPORTED"


def test_teleporting_onto_the_goal_is_still_success():
    *_, last = _feedback(_spec({"teleporters": [PORTAL]}, goal=(6, 3)), [A.MOVE_FORWARD] * 3)
    assert last == ("SUCCESS — MOVE_FORWARD: Reached the goal.", "DONE", False)


def test_frozen_actions_are_not_failures_except_done():
    spec = _spec({"frozen_tiles": [[2, 1]], "freeze_steps": 6})
    steps = _feedback(
        spec,
        [A.MOVE_FORWARD, A.MOVE_FORWARD, A.TURN_LEFT, A.PICKUP, A.TOGGLE, A.DONE, A.TURN_RIGHT],
    )
    assert [(s[1], s[2]) for s in steps] == [
        ("FROZEN", False),  # landed on the frozen tile
        ("FROZEN", False),  # swallowed MOVE_FORWARD: the ice, not the model
        ("FROZEN", False),
        ("FROZEN", False),
        ("FROZEN", False),
        ("WRONG_DONE", True),  # DONE while frozen is still a wrong DONE
        ("FROZEN", False),
    ]
    assert steps[5][0] == "WRONG DONE — DONE: DONE called but not at the goal."


def test_done_on_the_last_frozen_step_is_a_wrong_done():
    steps = _feedback(ice_spec(), [A.MOVE_FORWARD, A.TURN_LEFT, A.TURN_LEFT, A.DONE])
    assert steps[-1][1:] == ("WRONG_DONE", True)


def test_deaths_are_failures_teleports_are_not():
    death = _feedback(death_spec(), [A.MOVE_FORWARD, A.PICKUP, A.MOVE_FORWARD, A.MOVE_FORWARD])
    assert death[-1][2] is True
    assert _feedback(_spec({"death_portals": [[2, 1]]}), [A.MOVE_FORWARD])[0][2] is True
    assert _feedback(teleport_spec(), [A.MOVE_FORWARD] * 3)[-1][2] is False


def test_a_blocked_ride_is_still_a_failure_like_main():
    blocked = _feedback(_ride_spec(), [A.MOVE_FORWARD, A.TURN_LEFT, A.TURN_LEFT, A.MOVE_FORWARD])
    assert blocked[-1][1:] == ("ROTATING", True)


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


def test_runner_records_mechanic_events_and_the_failure_rule():
    _, steps, _ = _run(
        ice_spec(),
        [A.MOVE_FORWARD, A.MOVE_FORWARD, A.TURN_LEFT, A.DONE, A.TURN_LEFT],
        feedback="standard",
    )
    assert [s["event_type"] for s in steps[:5]] == [
        "FROZEN", "FROZEN", "FROZEN", "WRONG_DONE", "TURNED"
    ]
    assert steps[1]["feedback"] == (
        "FROZEN — MOVE_FORWARD: This action did nothing, wait until the freeze ends."
    )
    assert [s["consecutive_failures_after"] for s in steps[:5]] == [0, 0, 0, 1, 0]
    assert steps[0]["mechanic"] == {"kind": "froze", "cell_xy": (2, 1), "steps": 3}
    assert steps[3]["mechanic"] == {"kind": "frozen", "remaining": 0}
    assert "mechanic" not in steps[4]


def test_runner_teleport_is_not_a_failure_death_is():
    _, steps, _ = _run(teleport_spec(), [A.MOVE_FORWARD] * 3, feedback="standard")
    assert steps[2]["event_type"] == "TELEPORTED"
    assert steps[2]["consecutive_failures_after"] == 0
    assert steps[2]["position_after_row_col"] == [3, 6]
    assert "mechanic" not in steps[0]

    _, steps, _ = _run(
        death_spec(), [A.MOVE_FORWARD, A.PICKUP, A.MOVE_FORWARD, A.MOVE_FORWARD]
    )
    assert [s["event_type"] for s in steps[:4]] == ["MOVED", "PICKUP", "MOVED", "DEATH RESET"]
    assert steps[3]["feedback"] == "DEATH RESET"  # R1 cell: minimal feedback
    assert steps[3]["consecutive_failures_after"] == 1
    assert steps[3]["mechanic"]["kind"] == "died"


class _OneReplyAgent:
    """Answers the first query with ``reply`` and every later one with DONE."""

    def __init__(self, reply):
        self._reply, self.calls = reply, 0
        self.last_usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}

    def __call__(self, messages):
        self.calls += 1
        return self._reply if self.calls == 1 else "FINAL_OUTPUT: DONE"


def test_a_death_clears_the_queued_actions_and_asks_again():
    agent = _OneReplyAgent(
        "SUB_GOAL: east\n"
        "FINAL_OUTPUT: MOVE_FORWARD, PICKUP, MOVE_FORWARD, MOVE_FORWARD, TURN_LEFT"
    )
    backend = MiniGridBackend(render_mode="rgb_array")
    spec = death_spec()
    backend.configure(spec)
    config = ExperimentConfig(**{**R1_CELL, "querying": "subgoal"})
    result = build_runner(config, backend, spec).run(agent, verbose=False)
    steps = [rec for rec in result["transcript"] if rec.get("kind") == "step"]
    assert [s["event_type"] for s in steps[:4]] == ["MOVED", "PICKUP", "MOVED", "DEATH RESET"]
    # TURN_LEFT was dropped and the model asked again.
    assert (steps[4]["query_index"], steps[4]["action"]) == (2, "DONE")


def test_a_freeze_swallows_queued_moves_without_asking_again():
    agent = _OneReplyAgent(
        "SUB_GOAL: east\nFINAL_OUTPUT: MOVE_FORWARD, MOVE_FORWARD, MOVE_FORWARD"
    )
    backend = MiniGridBackend(render_mode="rgb_array")
    spec = ice_spec()
    backend.configure(spec)
    config = ExperimentConfig(**{**R1_CELL, "querying": "subgoal"})
    result = build_runner(config, backend, spec).run(agent, verbose=False)
    steps = [rec for rec in result["transcript"] if rec.get("kind") == "step"]
    assert [s["event_type"] for s in steps[:3]] == ["FROZEN", "FROZEN", "FROZEN"]
    assert [s["query_index"] for s in steps[:3]] == [1, 1, 1]


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


# --- other feedback callers: demo + prompt preview ------------------------------


def test_demo_session_labels_mechanics_like_the_runner(tmp_path):
    from demo.r1_config import R1_CONFIG
    from demo.session import MiniGridPlaySession

    path = tmp_path / "teleport.json"
    path.write_text(json.dumps(teleport_spec().to_dict()))
    session = MiniGridPlaySession(
        task_path=str(path), config=replace(R1_CONFIG, feedback="standard")
    )
    for _ in range(3):
        session._step_token("MOVE_FORWARD")
    last = session.transcript[-1]
    assert last["event_type"] == "TELEPORTED"
    assert last["feedback"] == (
        "TELEPORTED — MOVE_FORWARD: You teleported to (3,6)."
    )
    assert last["mechanic"]["kind"] == "teleported"
    assert "took the purple portal from (1, 4) to (3, 6)" in text_summary_history(
        session._model_transcript(), session.task_spec
    )


def test_demo_sounds_for_mechanic_events():
    from types import SimpleNamespace

    from demo.sounds import sfx_for_dispatch

    # The event_type fallback only: no prev_state and no mechanic tag, so the
    # warp/kill clips (tests/test_demo_fx_mechanics.py) do not fire.
    def sfx(event_type):
        session = SimpleNamespace(
            episode_done=False,
            episode_success=False,
            event_log=[],
            transcript=[{"kind": "step", "event_type": event_type}],
        )
        return sfx_for_dispatch(session, 0)

    assert sfx("TELEPORTED") == sfx("MOVED") == "step"
    assert sfx("DIED") == sfx("FROZEN") == sfx("NOTHING") == "invalid"


def test_prompt_preview_labels_mechanics():
    from prompting_experiments.preview_prompts import _solution_preview_steps

    spec = teleport_spec()
    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(spec)
    runner = build_runner(ExperimentConfig(**R1_CELL), backend, spec)
    runner.last_rgb, state, _ = backend.reset(seed=spec.seed)
    _, _, transcript, _ = _solution_preview_steps(runner, state, ["MOVE_FORWARD"] * 3)
    assert transcript[-1]["event_type"] == "TELEPORTED"
    assert transcript[-1]["mechanic"]["kind"] == "teleported"
    assert "mechanic" not in transcript[0]


# --- end to end: what the model is told -------------------------------------------


def _prompt_texts(result) -> list[str]:
    """Joined text blocks of each query's user message, in query order."""
    texts = []
    for rec in result["transcript"]:
        if rec.get("kind") != "query":
            continue
        content = rec["agent_messages"][-1]["content"]
        blocks = [content] if isinstance(content, str) else [
            b["text"] for b in content if b.get("type") == "text"
        ]
        texts.append("\n".join(blocks))
    return texts


def test_r1_cell_prompt_tells_the_model_about_the_death_reset():
    actions = [A.MOVE_FORWARD, A.PICKUP, A.MOVE_FORWARD, A.MOVE_FORWARD, A.MOVE_FORWARD]
    result, _, _ = _run(death_spec(), actions)
    after_death_and_one_move = _prompt_texts(result)[5]
    assert (
        "Activity summary:\nfirst you picked up the red key, then you stepped on a "
        "death tile at (1, 4) and the maze reset to the start tile (1, 1), finally "
        "you passed (1, 2)"
    ) in after_death_and_one_move


def test_image_text_history_shows_mechanic_feedback():
    result, _, _ = _run(
        ice_spec(), [A.MOVE_FORWARD, A.MOVE_FORWARD], observation="image_text", feedback="standard"
    )
    prompt = _prompt_texts(result)[2]
    assert (
        "Position before: (1, 1), facing EAST\nFINAL_OUTPUT: MOVE_FORWARD\n"
        "Feedback: MOVED — MOVE_FORWARD: Moved. The ice froze you for 3 steps.\n"
        "Position before: (1, 2), facing EAST\nFINAL_OUTPUT: MOVE_FORWARD\n"
        "Feedback: FROZEN — MOVE_FORWARD: You are frozen; the action had no effect. "
        "2 more steps until you thaw."
    ) in prompt
    assert "first you were frozen at (1, 2) for 3 steps" in prompt
