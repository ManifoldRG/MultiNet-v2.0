"""Seam tests for ``interface.episode_step.EpisodeStepper``.

The stepper is the shared per-episode state machine that both the serial
``ExperimentRunner.run`` driver and the future batch-API lockstep runner drive.
These tests pin the seam contract: (1) driving the stepper by hand produces the
exact same transcript as ``runner.run`` on the same scripted episode, (2) a
cardinal move drains ALL its expanded primitives before another query is issued,
and (3) an unparseable reply re-queries without stepping the environment, and
after ``max_parse_retries`` the episode ends with ``parse_failed``.
"""

from __future__ import annotations

from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.task_spec import TaskSpecification
from interface.agents.reply import Reply
from interface.config import ExperimentConfig
from interface.episode_step import EpisodeStepper
from interface.episode_log import _json_safe
from interface.runner import _reset_agent_usage, build_runner


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


_MECHANICS_SPEC = {
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

_MECHANICS_ACTIONS = [
    "MOVE_EAST",
    "PICKUP",
    "INTERACT",
    "MOVE_EAST",
    "MOVE_EAST",
    "MOVE_EAST",
    "DONE",
]


def _build(spec_dict, **config_kwargs):
    spec = TaskSpecification.from_dict(spec_dict)
    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(spec)
    runner = build_runner(
        ExperimentConfig(action_space="cardinal", **config_kwargs), backend, spec
    )
    return runner


def _drive_by_hand(runner, agent, *, verbose=False):
    """The exact serial driver, hand-written against the stepper seam."""
    stepper = EpisodeStepper(runner, verbose=verbose)
    stepper.start()
    while (messages := stepper.next_query()) is not None:
        _reset_agent_usage(agent)
        if hasattr(agent, "generate"):
            reply = agent.generate(messages)
        else:
            text = agent(messages)
            reply = Reply(
                text=text,
                usage=getattr(agent, "last_usage", None),
                thinking=getattr(agent, "last_thinking", None),
            )
        stepper.apply_reply(reply)
    return stepper.result()


def _scrub(transcript):
    """Drop non-deterministic latency fields and rgb frames for comparison."""
    scrubbed = _json_safe(transcript)  # drops "_"-prefixed rgb keys + numpy
    for rec in scrubbed:
        rec.pop("llm_latency_s", None)
    return scrubbed


def test_stepper_serial_equivalence():
    # Same scripted cardinal episode via runner.run vs a hand-driven stepper.
    runner_a = _build(_MECHANICS_SPEC)
    result_a = runner_a.run(ScriptedAgent(_MECHANICS_ACTIONS), verbose=False)

    runner_b = _build(_MECHANICS_SPEC)
    result_b = _drive_by_hand(runner_b, ScriptedAgent(_MECHANICS_ACTIONS))

    assert result_a["success"] is True
    assert result_b["success"] is True
    assert result_a["end_reason"] == result_b["end_reason"]
    assert result_a["query_count"] == result_b["query_count"]
    assert result_a["steps_used"] == result_b["steps_used"]
    assert _scrub(result_a["transcript"]) == _scrub(result_b["transcript"])


def test_stepper_query_boundary():
    # Facing EAST, MOVE_SOUTH expands to TURN_RIGHT, MOVE_FORWARD (two
    # primitives). Both must execute inside a single next_query() drain with no
    # query issued between them: the transcript must read reset, query,
    # step(TURN_RIGHT), step(MOVE_FORWARD), query -- never query in the middle.
    spec = {
        "task_id": "boundary",
        "seed": 0,
        "difficulty_tier": 1,
        "maze": {"dimensions": [7, 7], "walls": [], "start": [1, 1], "goal": [5, 5]},
        "mechanisms": {},
        "goal": {"type": "reach_position", "target": [5, 5]},
        "max_steps": 30,
    }
    runner = _build(spec)
    stepper = EpisodeStepper(runner, verbose=False)
    stepper.start()

    first = stepper.next_query()
    assert first is not None
    assert stepper.query_count == 1
    stepper.apply_reply(Reply(text="FINAL_OUTPUT: MOVE_SOUTH"))

    # One drain call must execute BOTH primitives and only then re-query.
    second = stepper.next_query()
    assert second is not None  # goal not reached; a new query is needed
    assert stepper.query_count == 2

    # The query#2 RECORD is written by apply_reply, not next_query; what matters
    # here is that both primitives stepped with no query record between them.
    kinds = [rec.get("kind") for rec in stepper.transcript]
    assert kinds == ["reset", "query", "step", "step"]
    steps = [rec for rec in stepper.transcript if rec.get("kind") == "step"]
    assert [s["action"] for s in steps] == ["TURN_RIGHT", "MOVE_FORWARD"]
    assert [s["cardinal_action"] for s in steps] == ["MOVE_SOUTH", "MOVE_SOUTH"]


def test_parse_failure_requeries_without_step():
    spec = {
        "task_id": "parse_fail",
        "seed": 0,
        "difficulty_tier": 1,
        "maze": {"dimensions": [5, 5], "walls": [], "start": [1, 1], "goal": [3, 3]},
        "mechanisms": {},
        "goal": {"type": "reach_position", "target": [3, 3]},
        "max_steps": 30,
    }
    runner = _build(spec, max_parse_retries=3)
    stepper = EpisodeStepper(runner, verbose=False)
    stepper.start()

    messages = stepper.next_query()
    assert messages is not None
    step_count_before = stepper.state.step_count

    # First garbage reply: re-query, env not stepped.
    stepper.apply_reply(Reply(text="i will not comply"))
    assert stepper.parse_failures == 1
    assert not stepper.finished
    messages = stepper.next_query()
    assert messages is not None  # requeried
    assert stepper.state.step_count == step_count_before  # no env step
    assert not any(rec.get("kind") == "step" for rec in stepper.transcript)

    # Second garbage reply: still under the cap, re-query again.
    stepper.apply_reply(Reply(text="still garbage"))
    assert stepper.parse_failures == 2
    messages = stepper.next_query()
    assert messages is not None

    # Third garbage reply hits max_parse_retries -> episode ends parse_failed.
    stepper.apply_reply(Reply(text="nope"))
    assert stepper.parse_failures == 3
    assert stepper.finished
    assert stepper.next_query() is None
    assert stepper.result()["end_reason"] == "parse_failed"
    assert stepper.state.step_count == step_count_before


def test_stepper_records_reply_stop_reason_and_truncation_additively():
    # New additive keys appear ONLY when present on the Reply.
    runner = _build(_MECHANICS_SPEC)
    stepper = EpisodeStepper(runner, verbose=False)
    stepper.start()
    stepper.next_query()
    stepper.apply_reply(
        Reply(text="FINAL_OUTPUT: MOVE_EAST", stop_reason="max_tokens", token_truncated=True)
    )
    query_rec = next(r for r in stepper.transcript if r.get("kind") == "query")
    assert query_rec["stop_reason"] == "max_tokens"
    assert query_rec["token_truncated"] is True

    # Legacy-shim reply (defaults) adds neither key.
    runner2 = _build(_MECHANICS_SPEC)
    stepper2 = EpisodeStepper(runner2, verbose=False)
    stepper2.start()
    stepper2.next_query()
    stepper2.apply_reply(Reply(text="FINAL_OUTPUT: MOVE_EAST"))
    q2 = next(r for r in stepper2.transcript if r.get("kind") == "query")
    assert "stop_reason" not in q2
    assert "token_truncated" not in q2
