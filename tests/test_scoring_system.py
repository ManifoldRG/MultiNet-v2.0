import json

from scorer.scoring import (
    ScorerConfig,
    compute_12d_score,
    compute_canonical_paths,
    compute_runtime_score,
    compute_static_score_artifact,
    score_task_file,
)
from gridworld.task_spec import TaskSpecification


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
    assert payload["validation"]["schema_valid"] is True
    assert payload["canonical_agent_features"]["greedy_solvability"] == 1.0


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
    )

    assert score.task_id == spec.task_id
    assert score.composite == 1.0
    assert score.signals["step_ratio"] == 1.0
    assert score.signals["cell_overlap_bfs"] == 1.0
    assert score.signals["token_efficiency"] == 1.0


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
    )

    assert score.signals["cell_overlap_bfs"] == 1.0
