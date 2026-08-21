"""Tests for the DROP action's state bookkeeping.

DROP places the held key in the agent's OWN cell (not the front cell — the
same-cell convention mirrors same-cell PICKUP), removes it from
`collected_keys`, and the observation layer renders it from the live
`state.key_positions` rather than the task spec's static position. Before this
was implemented, DROP fell through to MiniGridEnv.step's front-cell placement
with no bookkeeping, leaving a dropped key on the grid that was invisible to
the agent and still reported as held; these tests pin the fixed behavior.
"""

import sys
from pathlib import Path

import pytest

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from gridworld.actions import MiniGridActions
from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.custom_env import Key
from gridworld.task_spec import TaskSpecification
from interface import coords
from interface.renderer import render_user_observation_text


def _spec(**overrides):
    d = {
        "task_id": "drop_test",
        "seed": 1,
        "difficulty_tier": 3,
        "maze": {"dimensions": [8, 8], "walls": [], "start": [1, 1], "goal": [6, 6]},
        "mechanisms": {
            "keys": [{"id": "kR", "position": [2, 1], "color": "red"}],
            "doors": [
                {
                    "id": "DR",
                    "position": [4, 4],
                    "requires_key": "red",
                    "initial_state": "locked",
                }
            ],
        },
        "goal": {"type": "reach_position", "target": [6, 6]},
        "max_steps": 200,
    }
    d.update(overrides)
    return TaskSpecification.from_dict(d)


@pytest.fixture
def backend():
    b = MiniGridBackend(render_mode="rgb_array")
    b.configure(_spec())
    b.reset(seed=1)
    return b


def _keys_on_grid(env):
    found = {}
    for x in range(env.width):
        for y in range(env.height):
            cell = env.grid.get(x, y)
            if isinstance(cell, Key):
                found[getattr(cell, "key_id", "?")] = (x, y)
    return found


def _pick_up_the_key(backend):
    """Agent starts at (1,1) facing east; the key sits at (2,1)."""
    backend.env.step(MiniGridActions.MOVE_FORWARD)
    backend.env.step(MiniGridActions.PICKUP)
    assert backend.env.carrying is not None, "precondition: key must be held"


class TestDropPlacesKeyInAgentCell:
    def test_dropped_key_lands_under_the_agent(self, backend):
        _pick_up_the_key(backend)                    # agent at (2,1) holding kR
        backend.env.step(MiniGridActions.DROP)

        assert backend.env.carrying is None
        assert _keys_on_grid(backend.env) == {"kR": (2, 1)}
        assert "kR" not in backend.env.collected_keys, (
            "a dropped key is back on the grid and is no longer collected"
        )

    def test_drop_then_pickup_round_trips_in_two_actions(self, backend):
        _pick_up_the_key(backend)
        backend.env.step(MiniGridActions.DROP)
        backend.env.step(MiniGridActions.PICKUP)

        assert getattr(backend.env.carrying, "key_id", None) == "kR"
        assert "kR" in backend.env.collected_keys

    def test_drop_succeeds_regardless_of_facing(self, backend):
        """Same-cell placement must not depend on what is in front."""
        _pick_up_the_key(backend)
        for _ in range(2):
            backend.env.step(MiniGridActions.TURN_LEFT)   # face west
        backend.env.step(MiniGridActions.MOVE_FORWARD)    # to (1,1), wall ahead
        backend.env.step(MiniGridActions.DROP)

        assert backend.env.carrying is None
        assert _keys_on_grid(backend.env) == {"kR": (1, 1)}

    def test_drop_with_empty_hands_is_a_rejected_no_op(self, backend):
        before = backend.env.step_count
        _, _, _, _, info = backend.env.step(MiniGridActions.DROP)

        assert info.get("invalid_action") is True
        assert backend.env.carrying is None
        assert backend.env.step_count == before + 1

    def test_blocked_drop_message_says_how_to_recover(self):
        """The blocking case that matters is standing on the key you want while
        holding the wrong one. The agent must be told to step off first."""
        from prompting_experiments.prompt_templates import feedback

        assert "MOVE_FORWARD" in feedback.DROP_BLOCKED
        assert "empty cell" in feedback.DROP_BLOCKED

    def test_drop_onto_an_occupied_cell_is_rejected(self):
        """Standing on a switch, the cell is taken; the key stays in hand."""
        b = MiniGridBackend(render_mode="rgb_array")
        b.configure(_spec(mechanisms={
            "keys": [{"id": "kR", "position": [2, 1], "color": "red"}],
            "switches": [{"id": "s1", "position": [3, 1], "controls": ["g1"],
                          "color": "yellow", "switch_type": "toggle",
                          "initial_state": "off"}],
            "gates": [{"id": "g1", "position": [5, 5], "initial_state": "closed",
                       "color": "black"}],
        }))
        b.reset(seed=1)
        b.env.step(MiniGridActions.MOVE_FORWARD)   # (2,1), onto the key
        b.env.step(MiniGridActions.PICKUP)
        b.env.step(MiniGridActions.MOVE_FORWARD)   # (3,1), onto the switch
        held = b.env.carrying

        _, _, _, _, info = b.env.step(MiniGridActions.DROP)

        assert info.get("invalid_action") is True
        assert b.env.carrying is held
        assert "kR" in b.env.collected_keys


class TestDropIsInTheModelFacingVocabulary:
    """DROP is offered wherever PICKUP is, and nowhere else.

    Inverted deliberately from the pre-fix assertions: exposing DROP is what
    makes a wrong-key pickup recoverable, and the exclusion was previously
    pinned here precisely so this flip could not happen by accident.
    """

    def test_drop_is_offered_in_both_action_spaces(self):
        from interface.action_space import actions_hint, valid_actions

        assert "DROP" in valid_actions("egocentric")
        assert "DROP" in valid_actions("cardinal")
        assert "DROP" in actions_hint("egocentric")
        assert "DROP" in actions_hint("cardinal")

    def test_parser_accepts_a_drop_reply(self):
        from interface.parser import parse_final_output

        assert parse_final_output("FINAL_OUTPUT: DROP") == ["DROP"]
        assert parse_final_output("FINAL_OUTPUT: PICKUP") == ["PICKUP"]

    def test_drop_synonyms_resolve_on_the_regex_fallback_path(self):
        """Synonyms apply only when there is no FINAL_OUTPUT line — as for PICKUP."""
        from interface.parser import parse_final_output

        assert parse_final_output("I will put down the key") == ["DROP"]
        assert parse_final_output("drop key") == ["DROP"]
        # both vocabularies accept the same DROP phrasings ("release" included)
        from interface.action_space import synonyms, valid_actions

        assert parse_final_output(
            "release the key",
            valid_actions=valid_actions("cardinal"),
            synonyms=synonyms("cardinal"),
        ) == ["DROP"]
        # the FINAL_OUTPUT path takes exact tokens only, for DROP as for PICKUP
        assert parse_final_output("FINAL_OUTPUT: put down") is None
        assert parse_final_output("FINAL_OUTPUT: pick up") is None

    def test_actions_map_resolves_drop(self):
        from gridworld.actions import MiniGridActions
        from interface.actions_map import nlu_action_to_int

        assert nlu_action_to_int("DROP") == int(MiniGridActions.DROP)


class TestDroppedKeyIsObservable:
    def test_state_reports_live_key_positions(self, backend):
        _pick_up_the_key(backend)
        backend.env.step(MiniGridActions.DROP)

        state = backend.get_state()
        assert state.key_positions == {"kR": (2, 1)}, (
            "the agent must be able to see where a dropped key actually is"
        )

    def test_held_key_has_no_grid_position(self, backend):
        _pick_up_the_key(backend)
        state = backend.get_state()
        assert state.key_positions == {}

    def test_key_at_cell_follows_the_dropped_key(self, backend):
        _pick_up_the_key(backend)
        backend.env.step(MiniGridActions.DROP)
        state = backend.get_state()
        spec = _spec()

        # dropped in the agent's own cell (x,y)=(2,1) -> (row,col)=(1,2),
        # which here coincides with the key's spec cell, so move first and
        # re-drop somewhere the spec position cannot explain.
        assert coords.key_at_cell(spec, state, 1, 2) == "red"

    def test_dropped_key_is_observed_at_a_non_spec_cell(self, backend):
        """The observation must follow the live key, not the spec position.

        The two tests above drop at the key's spec cell, so they would pass
        even if the observation layer ignored state.key_positions. Here the
        drop lands somewhere the spec cannot explain.
        """
        _pick_up_the_key(backend)                     # agent at (2,1) holding kR
        backend.env.step(MiniGridActions.MOVE_FORWARD)  # to (3,1)
        backend.env.step(MiniGridActions.DROP)          # key now at (3,1)
        state = backend.get_state()
        spec = _spec()

        # (x,y)=(3,1) -> (row,col)=(1,3); the spec cell (x,y)=(2,1) -> (1,2)
        assert coords.key_at_cell(spec, state, 1, 3) == "red"
        assert coords.key_at_cell(spec, state, 1, 2) is None, (
            "the key's spec cell is empty after the drop moved it"
        )

        text = render_user_observation_text(spec, state)
        compact = text.replace(" ", "")
        assert "red key" in text.lower()
        assert "(1,3)" in compact, "listed at its current cell"

    def test_text_observation_lists_the_dropped_key(self, backend):
        _pick_up_the_key(backend)
        backend.env.step(MiniGridActions.DROP)
        state = backend.get_state()

        text = render_user_observation_text(_spec(), state)

        assert "red key" in text.lower(), "a dropped key must reappear in the text observation"
        assert "(1,2)" in text.replace(" ", ""), "listed at its current cell"


class TestKeyPositionsSerialization:
    """episode.json state snapshots must record where a dropped key sits."""

    def test_to_dict_round_trips_a_moved_key(self):
        from gridworld.backends.base import GridState

        state = GridState(
            agent_position=(3, 1),
            agent_direction=0,
            key_positions={"kR": (3, 1)},
        )

        d = state.to_dict()
        assert d["key_positions"] == {"kR": [3, 1]}, (
            "post-hoc analysis needs drop locations in the snapshot"
        )

        restored = GridState.from_dict(d)
        assert restored.key_positions == {"kR": (3, 1)}

    def test_from_dict_defaults_key_positions_on_legacy_snapshots(self):
        from gridworld.backends.base import GridState

        legacy = {
            "agent_position": [1, 1],
            "agent_direction": 0,
            "collected_keys": ["kR"],
        }

        restored = GridState.from_dict(legacy)
        assert restored.key_positions == {}


class TestDropAppearsWherePickupDoes:
    def test_mechanism_rules_explain_drop(self):
        from prompting_experiments.prompt_templates import system

        assert "PICKUP:" in system.MECHANISM_RULES       # guard the premise
        assert "DROP:" in system.MECHANISM_RULES
        assert "one key at a time" in system.MECHANISM_RULES

    def test_minimal_prompt_lists_drop_without_explaining_it(self):
        """minimal emits no MECHANISM_RULES — DROP appears exactly as PICKUP does."""
        from interface.action_space import actions_hint
        from interface.prompt_strategies import MinimalPromptStrategy

        sys_prompt = MinimalPromptStrategy(actions_hint("egocentric")).build_system_prompt()

        assert "DROP" in sys_prompt
        assert "PICKUP" in sys_prompt
        assert "DROP:" not in sys_prompt

    def test_verbose_prompt_explains_drop(self):
        from interface.action_space import actions_hint
        from interface.prompt_strategies import VerbosePromptStrategy

        sys_prompt = VerbosePromptStrategy(actions_hint("egocentric")).build_system_prompt()

        assert "DROP:" in sys_prompt


class TestDropFeedback:
    def test_successful_drop_reports_a_drop_event(self, backend):
        from interface.feedback import infer_step_outcome

        _pick_up_the_key(backend)
        prev = backend.get_state()
        backend.env.step(MiniGridActions.DROP)
        curr = backend.get_state()

        event, message = infer_step_outcome("DROP", prev, curr, 0.0, False, _spec())

        assert event == "DROP"
        assert "red" in message.lower()
        # dropped in the agent's own cell (x,y)=(2,1) -> (row,col)=(1,2)
        assert "(1,2)" in message.replace(" ", "")

    def test_drop_with_empty_hands_reports_nothing(self, backend):
        from interface.feedback import infer_step_outcome

        prev = backend.get_state()
        backend.env.step(MiniGridActions.DROP)
        curr = backend.get_state()

        event, message = infer_step_outcome("DROP", prev, curr, 0.0, False, _spec())

        assert event == "NOTHING"
        assert "not carrying" in message.lower()


class TestDropTextSummary:
    def test_drop_appears_in_the_text_summary_history(self):
        from interface.observation import _extract_mechanism_events

        steps = [{
            "kind": "step",
            "event_type": "DROP",
            "state_before": {"agent_carrying": "red", "collected_keys": ["kR"]},
            "state_after": {"agent_carrying": None, "collected_keys": []},
        }]

        events = _extract_mechanism_events(steps, _spec())

        assert [text for _, text in events] == ["dropped the red key"]

    def test_drop_summary_resolves_color_from_the_collected_diff(self):
        """The dropped key is the one that leaves collected_keys — the color
        must not depend on agent_carrying surviving into the record."""
        from interface.observation import _extract_mechanism_events

        steps = [{
            "kind": "step",
            "event_type": "DROP",
            "state_before": {"agent_carrying": None, "collected_keys": ["kR"]},
            "state_after": {"agent_carrying": None, "collected_keys": []},
        }]

        events = _extract_mechanism_events(steps, _spec())

        assert [text for _, text in events] == ["dropped the red key"]

    def test_drop_summary_without_key_details_stays_sensible(self):
        """Never render 'dropped the a key' when the color is unknowable."""
        from interface.observation import _extract_mechanism_events

        steps = [{
            "kind": "step",
            "event_type": "DROP",
            "state_before": {},
            "state_after": {},
        }]

        events = _extract_mechanism_events(steps, _spec())

        assert [text for _, text in events] == ["dropped the key"]

    def test_pickup_then_drop_reads_in_order(self):
        from interface.observation import _extract_mechanism_events

        steps = [
            {
                "kind": "step",
                "event_type": "PICKUP",
                "state_before": {"agent_carrying": None, "collected_keys": []},
                "state_after": {"agent_carrying": "red", "collected_keys": ["kR"]},
            },
            {
                "kind": "step",
                "event_type": "DROP",
                "state_before": {"agent_carrying": "red", "collected_keys": ["kR"]},
                "state_after": {"agent_carrying": None, "collected_keys": []},
            },
        ]

        events = _extract_mechanism_events(steps, _spec())

        assert [text for _, text in events] == [
            "picked up the red key",
            "dropped the red key",
        ]


class TestPlannerIgnoresDrop:
    """DROP is deliberately absent from the BFS and greedy planners.

    Carrying a key does exactly two things: it blocks further PICKUP, and it
    opens a matching door. Dropping spends an action to restore the former while
    losing the latter, and the former is only worth restoring while holding a key
    you did not need -- which neither an optimal nor a greedy solver ever picks
    up. So DROP is never on either planner's path.

    Modelling it would force key positions into PlannerState, multiplying the
    state space by cells**keys (R1's largest maze: 4,696 states -> ~1e9) and
    making BFS difficulty scoring infeasible. The planners stay complete for the
    optimum, which is what optimal_steps and beatability are defined over.

    These tests exist to fail loudly if someone adds a DROP edge without first
    solving that state-space blowup (or accepting the difficulty-scoring cost).
    """

    def test_planner_emits_no_drop_actions(self):
        from gridworld.baselines import plan_bfs_path

        path = plan_bfs_path(_spec())

        assert path.success
        assert int(MiniGridActions.DROP) not in path.actions

    def test_no_fixture_maze_needs_a_drop_to_solve(self):
        """Exposing DROP must not shift any difficulty number."""
        import json

        from gridworld.baselines import plan_bfs_path

        mazes_root = Path(__file__).resolve().parent.parent / "mazes"
        ogbench_root = (
            Path(__file__).resolve().parent.parent
            / "ogbench" / "ogbench" / "procgen" / "maze_jsons"
        )
        maze_paths = list(mazes_root.rglob("*.json"))
        if ogbench_root.exists():
            maze_paths.extend(ogbench_root.rglob("*.json"))
        checked = 0
        for path in sorted(maze_paths):
            with open(path) as fh:
                try:
                    raw = json.load(fh)
                except json.JSONDecodeError:
                    continue
            if not isinstance(raw, dict) or "maze" not in raw:
                continue
            try:
                spec = TaskSpecification.from_dict(raw)
                ok, _ = spec.validate()
            except Exception:
                continue
            if not ok:
                continue
            plan = plan_bfs_path(spec)
            if plan.success:
                assert int(MiniGridActions.DROP) not in plan.actions, path
                checked += 1

        assert checked > 20, f"only {checked} fixture mazes exercised"
