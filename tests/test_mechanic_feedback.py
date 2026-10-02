"""Teleport, death and ice are reported to the model as what they are.

The world model is the single source of truth: ``classify_mechanic`` labels a
transition (teleported / died / froze / frozen) next to the rules, the env
exposes it as ``info["mechanic"]``, and feedback + the activity summary only
render it.
"""

from __future__ import annotations

from dataclasses import replace

from gridworld.actions import MiniGridActions as A
from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.task_spec import TaskSpecification
from gridworld.world_model import TaskPlanningContext, classify_mechanic, successors

PORTAL = {
    "id": "portal_purple",
    "position_a": [4, 1],
    "position_b": [6, 3],
    "color": "purple",
    "bidirectional": True,
}


def _spec(mechanisms: dict, goal=(7, 3)) -> TaskSpecification:
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
