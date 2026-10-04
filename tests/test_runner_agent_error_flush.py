"""A transport failure must not discard an in-flight episode.

``run_pipeline`` has no per-episode exception isolation and episode artifacts are
only written after ``runner.run`` returns, so an agent that raises used to kill
the whole process and take every paid query of that episode with it — no
episode.json, no transcript, nothing to score. That is not hypothetical: the R1
Kimi rerun lost arm 3 (~54 queries, ~14h) to five consecutive 2400s Moonshot
timeouts exhausting the retry budget.

The runner now ends the episode with an ``agent_error:`` end_reason instead,
so the partial transcript survives and downstream scoring sees an honest,
filterable outcome.
"""

from __future__ import annotations

import pytest

from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.task_spec import TaskSpecification
from interface.config import ExperimentConfig
from interface.runner import build_runner

SPEC = {
    "task_id": "agent_error_flush",
    "seed": 0,
    "difficulty_tier": 1,
    "maze": {"dimensions": [5, 5], "walls": [], "start": [1, 1], "goal": [1, 3]},
    "mechanisms": {},
    "goal": {"type": "reach_position", "target": [1, 3]},
    "max_steps": 20,
}


class RaisingAgent:
    """Answers `ok_replies` queries, then raises like a dead transport."""

    def __init__(self, ok_replies: int, exc: Exception):
        self.ok_replies = ok_replies
        self.exc = exc
        self.calls = 0
        self.last_usage = {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}

    def __call__(self, messages):
        self.calls += 1
        if self.calls > self.ok_replies:
            raise self.exc
        return "FINAL_OUTPUT: TURN_LEFT"


def _run(agent):
    spec = TaskSpecification.from_dict(SPEC)
    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(spec)
    runner = build_runner(ExperimentConfig(), backend, spec)
    return runner.run(agent, verbose=False)


def _steps(result):
    return [e for e in result["transcript"] if e.get("kind") == "step"]


def test_agent_error_ends_the_episode_instead_of_propagating():
    agent = RaisingAgent(2, RuntimeError("Moonshot API timed out after 2400s"))

    result = _run(agent)

    assert result["end_reason"].startswith("agent_error:RuntimeError")
    assert "timed out after 2400s" in result["end_reason"]
    assert result["success"] is False


def test_agent_error_preserves_the_work_already_paid_for():
    agent = RaisingAgent(2, RuntimeError("boom"))

    result = _run(agent)

    # Both successful replies are still in the transcript, and the query counter
    # matches the applied queries — the failed round's reserved index is rolled
    # back rather than left dangling.
    assert len(_steps(result)) == 2
    assert result["query_count"] == 2


def test_a_clean_episode_is_unaffected():
    """The guard must not change any non-raising run."""
    agent = RaisingAgent(99, RuntimeError("never raised"))

    result = _run(agent)

    assert not result["end_reason"].startswith("agent_error")
    assert result["query_count"] > 2


def test_keyboardinterrupt_still_propagates():
    """Ctrl-C must stay a real interrupt, not become an episode outcome."""
    agent = RaisingAgent(1, KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        _run(agent)
