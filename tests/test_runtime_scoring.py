"""Regression coverage for scorer/runtime.py::compute_runtime_score.

Locks in that new early-termination failure reasons (``stalled`` and
``terminated_failure``) can never accidentally earn success credit: the
scorer keys success credit off the ``success`` boolean alone, and these
end_reason values must always co-occur with ``success=False``.
"""

import pytest

from gridworld.task_spec import TaskSpecification
from scorer.scoring import (
    ScorerConfig,
    compute_canonical_paths,
    compute_runtime_score,
    compute_static_score_artifact,
)


def make_spec(**overrides):
    data = {
        "task_id": "runtime_failure_case",
        "seed": 7,
        "difficulty_tier": 1,
        "maze": {
            "dimensions": [5, 5],
            "walls": [],
            "start": [1, 1],
            "goal": [3, 1],
        },
        "mechanisms": {},
        "rules": {"observability": "full", "view_size": 7},
        "goal": {"type": "reach_position", "target": [3, 1]},
        "max_steps": 20,
    }
    data.update(overrides)
    return TaskSpecification.from_dict(data)


@pytest.mark.parametrize("end_reason", ["stalled", "terminated_failure"])
def test_stalled_and_terminated_failure_score_zero_success_credit(end_reason):
    spec = make_spec()
    canonical = compute_canonical_paths(spec)
    static_score = compute_static_score_artifact(spec)
    run = {
        "task_id": spec.task_id,
        "backend": "minigrid",
        "adapter": "unit",
        "model_id": "unit-model",
        "seed": 7,
        "success": False,
        "end_reason": end_reason,
        "steps_taken": 5,
        "terminated": True,
        "truncated": False,
        "total_tokens": 500,
        "trajectory": [
            {"state": {"agent_position": [1, 1]}},
            {"state": {"agent_position": [2, 1]}},
        ],
        # Agent never reaches the goal tile (3, 1) — a genuine failure, not
        # merely a success flag flipped off while the trajectory succeeded.
        "final_state": {"agent_position": [2, 1], "step_count": 5},
    }

    config = ScorerConfig.from_dict({"runtime_weights": {"greedy_penalty": 0.0}})
    score = compute_runtime_score(
        run,
        static_score=static_score,
        canonical_paths=canonical,
        config=config,
        difficulty_max_static_score=static_score.static_score,
    )

    # The success signal itself must be falsy...
    assert score.signals["success"] in (False, 0, 0.0)
    # ...and the end_reason must be threaded through into the runtime
    # signals rather than silently dropped or overwritten by "unknown".
    assert score.signals["terminated_reason"] == end_reason
    # ...and because the scorer multiplies by success_factor, a failed run
    # must receive an exactly-zero composite score no matter what the
    # (irrelevant, since success=False) step/token/overlap signals look like.
    assert score.composite == 0.0
