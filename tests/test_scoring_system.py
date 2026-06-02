import json

import pytest

from gridworld.actions import MiniGridActions
from gridworld.baselines import plan_bfs_path, trace_planned_actions
from gridworld.task_spec import TaskSpecification
from gridworld.task_validator import TaskValidator
from scorer.scoring import (
    ScorerConfig,
    compute_12d_score,
    compute_canonical_paths,
    compute_runtime_score,
    compute_static_score_artifact,
    score_task_file,
)


def make_spec(**overrides):
    data = {
        "task_id": "scorer_case",
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


def test_canonical_paths_include_bfs_actions_and_positions():
    spec = make_spec()

    report = compute_canonical_paths(spec)

    assert report.success is True
    assert report.actions == ["move_forward", "move_forward"]
    assert report.positions == [(1, 1), (2, 1), (3, 1)]
    assert report.optimal_steps == 2
    assert report.states_explored > 0
    assert report.greedy is not None
    assert report.greedy["success"] is True


def test_static_score_uses_configurable_weights():
    spec = make_spec()
    default_score = compute_12d_score(spec)
    config = ScorerConfig.from_dict(
        {
            "version": "unit",
            "static_dimension_weights": {
                "optimal_path_length": 2.0,
                "grid_size": 0.0,
            },
        }
    )

    weighted = compute_12d_score(spec, config=config)

    assert weighted.weights[0] == 2.0
    assert weighted.weights[8] == 0.0
    assert weighted.composite != default_score.composite


def test_static_score_rejects_partial_explicit_weight_vectors():
    spec = make_spec()

    with pytest.raises(ValueError, match="Expected 12 static weights"):
        compute_12d_score(spec, weights=[1.0, 2.0])
    with pytest.raises(ValueError, match="Expected 12 static weights"):
        compute_12d_score(spec, weights=[])


def test_score_task_file_writes_stage_two_artifacts(tmp_path):
    spec = make_spec()
    task_path = tmp_path / "task.json"
    spec.to_json(str(task_path))

    canonical, static_score = score_task_file(task_path, output_dir=tmp_path / "artifacts")

    assert canonical.success is True
    assert static_score.is_beatable is True
    assert (tmp_path / "artifacts" / "canonical_paths.json").exists()
    scored_path = tmp_path / "artifacts" / "scored_static.json"
    assert scored_path.exists()
    with open(scored_path) as f:
        payload = json.load(f)
    assert payload["task_id"] == spec.task_id
    assert "dimensions_12" in payload
    assert "dimensions" not in payload
    assert "composite" not in payload
    assert payload["validation"]["schema_valid"] is True
    assert payload["canonical_agent_features"]["greedy_solvability"] == 1.0


def test_score_task_file_reuses_primary_validator_result(tmp_path, monkeypatch):
    spec = make_spec()
    task_path = tmp_path / "task.json"
    spec.to_json(str(task_path))
    calls = 0
    original_validate = TaskValidator.validate

    def count_validate(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original_validate(self, *args, **kwargs)

    monkeypatch.setattr(TaskValidator, "validate", count_validate)

    score_task_file(task_path)

    assert calls == 1


def test_runtime_score_from_episode_json_payload():
    spec = make_spec()
    canonical = compute_canonical_paths(spec)
    static_score = compute_static_score_artifact(spec)
    run = {
        "task_id": spec.task_id,
        "backend": "minigrid",
        "adapter": "unit",
        "model_id": "unit-model",
        "seed": 7,
        "success": True,
        "steps_taken": 2,
        "terminated": True,
        "truncated": False,
        "total_tokens": 500,
        "trajectory": [
            {"state": {"agent_position": [1, 1]}},
            {"state": {"agent_position": [2, 1]}},
        ],
        "final_state": {"agent_position": [3, 1], "step_count": 2},
    }

    config = ScorerConfig.from_dict({"runtime_weights": {"greedy_penalty": 0.0}})
    score = compute_runtime_score(
        run,
        static_score=static_score,
        canonical_paths=canonical,
        config=config,
        difficulty_max_static_score=static_score.static_score,
    )

    assert score.task_id == spec.task_id
    assert score.composite == 1.0
    assert score.signals["step_ratio"] == 1.0
    assert score.signals["cell_overlap_bfs"] == 1.0
    assert score.signals["cell_overlap_greedy"] == 1.0
    assert score.signals["token_efficiency"] == 1.0
    assert "path_choice" not in score.signals
    assert "distractor_interactions" not in score.signals


def test_runtime_score_prefers_interface_state_after_over_row_col_position_after():
    spec = make_spec()
    canonical = compute_canonical_paths(spec)
    static_score = compute_static_score_artifact(spec)
    run = {
        "success": True,
        "steps_used": 2,
        "end_reason": "success",
        "task_spec": spec.to_dict(),
        "initial_state": {"agent_position": [1, 1]},
        "final_state": {"agent_position": [3, 1], "step_count": 2},
        "transcript": [
            {
                "kind": "reset",
                "state": {"agent_position": [1, 1]},
            },
            {
                "kind": "step",
                "position_after": [1, 2],
                "state_after": {"agent_position": [2, 1]},
            },
            {
                "kind": "step",
                "position_after": [1, 3],
                "state_after": {"agent_position": [3, 1]},
            },
        ],
    }

    config = ScorerConfig.from_dict({"runtime_weights": {"greedy_penalty": 0.0}})
    score = compute_runtime_score(
        run,
        static_score=static_score,
        canonical_paths=canonical,
        config=config,
        difficulty_max_static_score=static_score.static_score,
    )

    assert score.signals["cell_overlap_bfs"] == 1.0


def test_runtime_score_requires_suite_difficulty_normalizer():
    spec = make_spec()
    canonical = compute_canonical_paths(spec)
    static_score = compute_static_score_artifact(spec)

    with pytest.raises(ValueError, match="difficulty_max_static_score"):
        compute_runtime_score(
            {"success": True, "steps": 2},
            static_score=static_score,
            canonical_paths=canonical,
        )


def test_runtime_score_rejects_suite_max_smaller_than_task_score():
    spec = make_spec()
    canonical = compute_canonical_paths(spec)
    static_score = compute_static_score_artifact(spec)

    with pytest.raises(ValueError, match="at least the task static score"):
        compute_runtime_score(
            {"success": True, "steps": 2},
            static_score=static_score,
            canonical_paths=canonical,
            difficulty_max_static_score=static_score.static_score - 1,
        )


def test_runtime_score_rejects_unevaluated_greedy_solvability():
    spec = make_spec()
    canonical = compute_canonical_paths(spec)
    static_score = compute_static_score_artifact(spec).to_dict()
    static_score["canonical_agent_features"]["greedy_solvability"] = None

    with pytest.raises(ValueError, match="greedy_solvability"):
        compute_runtime_score(
            {"success": True, "steps": 2},
            static_score=static_score,
            canonical_paths=canonical,
            difficulty_max_static_score=static_score["static_score"],
        )


def test_runtime_token_count_does_not_double_count_nested_step_tokens():
    spec = make_spec()
    canonical = compute_canonical_paths(spec)
    static_score = compute_static_score_artifact(spec)
    score = compute_runtime_score(
        {
            "success": True,
            "steps": 2,
            "trajectory": [{"tokens": 100, "info": {"tokens": 100}}],
        },
        static_score=static_score,
        canonical_paths=canonical,
        difficulty_max_static_score=static_score.static_score,
    )

    assert score.signals["token_count"] == 100


def test_runtime_token_count_reads_query_transcript_usage():
    spec = make_spec()
    canonical = compute_canonical_paths(spec)
    static_score = compute_static_score_artifact(spec)
    score = compute_runtime_score(
        {
            "success": True,
            "steps": 2,
            "transcript": [
                {
                    "kind": "query",
                    "usage": {"input_tokens": 80, "output_tokens": 20},
                }
            ],
        },
        static_score=static_score,
        canonical_paths=canonical,
        difficulty_max_static_score=static_score.static_score,
    )

    assert score.signals["token_count"] == 100


def test_zero_step_plans_do_not_inflate_optimal_steps_with_done():
    spec = make_spec(
        maze={
            "dimensions": [5, 5],
            "walls": [],
            "start": [1, 1],
            "goal": [1, 1],
        },
        goal={"type": "reach_position", "target": [1, 1]},
    )

    path = plan_bfs_path(spec)
    traced_done = trace_planned_actions(spec, [int(MiniGridActions.DONE)])

    assert path.success is True
    assert path.action_labels == []
    assert traced_done.success is True
    assert traced_done.action_labels == []
