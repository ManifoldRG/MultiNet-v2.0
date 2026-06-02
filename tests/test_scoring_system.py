import argparse
import json
import pathlib

import pytest

from gridworld.actions import MiniGridActions
from gridworld.baselines import plan_bfs_path, trace_planned_actions
from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.runner.grid_runner import GridRunner
from gridworld.task_spec import TaskSpecification
from gridworld.task_validator import TaskValidator, compute_difficulty
from scorer.artifacts import CanonicalPathReport, ScoredDifficulty
from scorer.config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DISTRACTOR_TYPE_WEIGHTS,
    DEFAULT_RUNTIME_WEIGHTS,
    DIMENSION_NAMES,
    load_scorer_config,
)
from scorer.scoring import (
    ScorerConfig,
    build_experiment_reports,
    compute_12d_score,
    compute_canonical_paths,
    compute_runtime_score,
    compute_static_score_artifact,
    score_task_file,
)
from scripts.score_json import _default_runtime_output, _runtime, _static_target_dirs
from scripts.visualize_scores import main as visualize_scores_main


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


def test_shipped_config_matches_code_defaults():
    config = load_scorer_config(DEFAULT_CONFIG_PATH)

    assert list(config.static_dimension_weights) == DIMENSION_NAMES
    assert config.distractor_type_weights == DEFAULT_DISTRACTOR_TYPE_WEIGHTS
    assert config.runtime_weights == DEFAULT_RUNTIME_WEIGHTS


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


def test_score_task_file_rejects_invalid_schema_before_planning(tmp_path, monkeypatch):
    spec = make_spec(
        maze={
            "dimensions": [5, 5],
            "walls": [],
            "start": [1, 1],
            "goal": [9, 9],
        },
        goal={"type": "reach_position", "target": [9, 9]},
    )
    task_path = tmp_path / "task.json"
    spec.to_json(str(task_path))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("planner must not execute for schema-invalid tasks")

    monkeypatch.setattr("scorer.static.plan_bfs_path", fail_if_called)
    monkeypatch.setattr("scorer.static.plan_greedy_path", fail_if_called)

    with pytest.raises(ValueError, match="failed schema validation"):
        score_task_file(task_path)


def test_static_score_uses_validator_metrics():
    spec = make_spec()
    difficulty = compute_difficulty(spec)
    score = compute_12d_score(spec)

    # Path/search dimensions are sourced from the validator, which is the
    # authoritative static complexity solver.
    assert score.dimensions[0] == float(difficulty.optimal_steps)
    assert score.dimensions[1] == float(difficulty.states_explored)
    assert score.dimensions[2] == float(difficulty.backtrack_count)


PUSH_TASK_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "gridworld"
    / "tasks"
    / "tier4"
    / "blocked_path_002.json"
)


@pytest.mark.skipif(
    not PUSH_TASK_PATH.exists(), reason="push-block corpus task not available"
)
def test_push_required_task_scores_with_replayable_canonical_path(tmp_path):
    """Regression: runtime canonical paths must include required block pushes."""
    from scorer.io import load_json, task_spec_from_payload

    spec = task_spec_from_payload(load_json(PUSH_TASK_PATH))
    difficulty = compute_difficulty(spec)
    assert difficulty.is_beatable is True
    bfs_path = plan_bfs_path(spec)
    assert bfs_path.success is True
    assert any(label.startswith("push:") for label in bfs_path.action_labels)

    canonical, static = score_task_file(PUSH_TASK_PATH, output_dir=tmp_path / "art")

    assert canonical.success is True
    assert canonical.optimal_steps == len(bfs_path.action_labels)
    assert canonical.optimal_steps > 0
    assert static.is_beatable is True
    assert static.dimensions["optimal_path_length"] == float(difficulty.optimal_steps)
    assert static.dimensions["search_space_size"] == float(difficulty.states_explored)
    assert static.dimensions["optimal_path_length"] > 0


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
        "task_spec": spec.to_dict(),
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
    assert score.signals["optimality_ratio"] == 1.0
    assert score.signals["cell_overlap_bfs"] == 1.0
    assert score.signals["cell_overlap_greedy"] == 1.0
    assert score.signals["token_efficiency"] == 1.0
    assert score.signals["terminated_reason"] == "goal_reached"
    assert score.signals["distractor_interactions"] == 0
    assert score.signals["irreversible_failures"] == 0


@pytest.mark.skipif(
    not PUSH_TASK_PATH.exists(), reason="push-block corpus task not available"
)
def test_push_required_canonical_path_replays_and_scores_as_optimal():
    from scorer.io import load_json, task_spec_from_payload

    spec = task_spec_from_payload(load_json(PUSH_TASK_PATH))
    canonical, static = score_task_file(PUSH_TASK_PATH)
    actions = iter(plan_bfs_path(spec).actions)
    result = GridRunner(MiniGridBackend()).run_episode(
        spec,
        policy_fn=lambda *_: next(actions),
    )
    run = result.to_dict()
    run["task_spec"] = spec.to_dict()
    run["total_tokens"] = 100

    score = compute_runtime_score(
        run,
        static_score=static,
        canonical_paths=canonical,
        difficulty_max_static_score=static.static_score,
    )

    assert result.success is True
    assert score.signals["step_ratio"] == 1.0
    assert score.signals["cell_overlap_bfs"] == 1.0
    assert score.signals["mechanism_interaction_order"] == [
        "push:block_a:4,5",
        "push:block_a:4,6",
    ]


def test_runtime_score_prefers_interface_state_after_over_row_col_position_after():
    spec = make_spec()
    canonical = compute_canonical_paths(spec)
    static_score = compute_static_score_artifact(spec)
    run = {
        "success": True,
        "steps_used": 2,
        "total_tokens": 100,
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
            {"success": True, "steps": 2, "total_tokens": 100},
            static_score=static_score,
            canonical_paths=canonical,
        )


def test_runtime_score_rejects_suite_max_smaller_than_task_score():
    spec = make_spec()
    canonical = compute_canonical_paths(spec)
    static_score = compute_static_score_artifact(spec)

    with pytest.raises(ValueError, match="at least the task static score"):
        compute_runtime_score(
            {"success": True, "steps": 2, "total_tokens": 100},
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
            {"success": True, "steps": 2, "total_tokens": 100},
            static_score=static_score,
            canonical_paths=canonical,
            difficulty_max_static_score=static_score["static_score"],
        )


def test_runtime_score_rejects_schema_invalid_static_artifact_clearly():
    spec = make_spec()
    canonical = compute_canonical_paths(spec)
    static_score = compute_static_score_artifact(spec).to_dict()
    static_score["validation"]["schema_valid"] = False

    with pytest.raises(ValueError, match="schema-valid"):
        compute_runtime_score(
            {"success": True, "steps": 2, "total_tokens": 100},
            static_score=static_score,
            canonical_paths=canonical,
            difficulty_max_static_score=static_score["static_score"],
        )


def test_runtime_score_rejects_unbeatable_static_artifact_clearly():
    spec = make_spec()
    canonical = compute_canonical_paths(spec)
    static_score = compute_static_score_artifact(spec).to_dict()
    static_score["is_beatable"] = False

    with pytest.raises(ValueError, match="beatable"):
        compute_runtime_score(
            {"success": True, "steps": 2, "total_tokens": 100},
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
            "task_spec": spec.to_dict(),
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
            "task_spec": spec.to_dict(),
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


def test_runtime_hash_ignores_non_scoring_transcript_context():
    spec = make_spec()
    canonical = compute_canonical_paths(spec)
    static_score = compute_static_score_artifact(spec)
    base_run = {
        "success": True,
        "steps": 2,
        "total_tokens": 100,
        "task_spec": spec.to_dict(),
        "transcript": [
            {
                "kind": "query",
                "agent_messages": [{"role": "user", "content": "first"}],
            }
        ],
    }
    changed_context = {
        **base_run,
        "transcript": [
            {
                "kind": "query",
                "agent_messages": [{"role": "user", "content": "second"}],
            }
        ],
    }

    first = compute_runtime_score(
        base_run,
        static_score=static_score,
        canonical_paths=canonical,
        difficulty_max_static_score=static_score.static_score,
    )
    second = compute_runtime_score(
        changed_context,
        static_score=static_score,
        canonical_paths=canonical,
        difficulty_max_static_score=static_score.static_score,
    )

    assert first.inputs_hash == second.inputs_hash


@pytest.mark.parametrize("token_count", [None, 0])
def test_runtime_score_rejects_missing_or_zero_token_telemetry(token_count):
    spec = make_spec()
    canonical = compute_canonical_paths(spec)
    static_score = compute_static_score_artifact(spec)
    run = {"success": True, "steps": 2}
    if token_count is not None:
        run["total_tokens"] = token_count

    with pytest.raises(ValueError, match="token"):
        compute_runtime_score(
            run,
            static_score=static_score,
            canonical_paths=canonical,
            difficulty_max_static_score=static_score.static_score,
        )


def test_runtime_score_rejects_missing_step_telemetry():
    spec = make_spec()
    canonical = compute_canonical_paths(spec)
    static_score = compute_static_score_artifact(spec)

    with pytest.raises(ValueError, match="step telemetry"):
        compute_runtime_score(
            {"success": True, "total_tokens": 100},
            static_score=static_score,
            canonical_paths=canonical,
            difficulty_max_static_score=static_score.static_score,
        )


def test_runtime_score_rejects_missing_runtime_diagnostics():
    spec = make_spec()
    canonical = compute_canonical_paths(spec)
    static_score = compute_static_score_artifact(spec)

    with pytest.raises(ValueError, match="task_spec or precomputed"):
        compute_runtime_score(
            {"success": True, "steps": 2, "total_tokens": 100},
            static_score=static_score,
            canonical_paths=canonical,
            difficulty_max_static_score=static_score.static_score,
        )


def test_runtime_score_rejects_off_contract_terminated_reason():
    spec = make_spec()
    canonical = compute_canonical_paths(spec)
    static_score = compute_static_score_artifact(spec)

    with pytest.raises(ValueError, match="Unsupported terminated_reason"):
        compute_runtime_score(
            {
                "success": False,
                "terminated_reason": "unknown",
                "steps": 2,
                "total_tokens": 100,
            },
            static_score=static_score,
            canonical_paths=canonical,
            difficulty_max_static_score=static_score.static_score,
        )


def test_runtime_score_requires_path_choice_for_test_two():
    spec = make_spec()
    canonical = compute_canonical_paths(spec)
    static_score = compute_static_score_artifact(spec)

    with pytest.raises(ValueError, match="path_choice"):
        compute_runtime_score(
            {
                "experiment": "complexity_distance",
                "success": True,
                "steps": 2,
                "total_tokens": 100,
                "task_spec": spec.to_dict(),
            },
            static_score=static_score,
            canonical_paths=canonical,
            difficulty_max_static_score=static_score.static_score,
        )


def test_runtime_score_reconstructs_distractor_and_irreversible_push_signals():
    spec = make_spec(
        maze={
            "dimensions": [6, 6],
            "walls": [],
            "start": [2, 3],
            "goal": [4, 4],
        },
        mechanisms={
            "blocks": [
                {"id": "block", "position": [2, 2], "pushable": True},
            ],
        },
        goal={
            "type": "push_block_to",
            "target_ids": ["block"],
            "target_positions": [[3, 2]],
        },
        distractors=[
            {
                "type": "decoy_block",
                "element_id": "block",
                "description": "Pushing upward traps the target block.",
            },
        ],
    )
    canonical = compute_canonical_paths(spec)
    static_score = compute_static_score_artifact(spec)
    before = {
        "agent_position": [2, 3],
        "agent_direction": 3,
        "block_positions": {"block": [2, 2]},
    }
    after = {
        "agent_position": [2, 2],
        "agent_direction": 3,
        "block_positions": {"block": [2, 1]},
    }
    score = compute_runtime_score(
        {
            "experiment": "mechanism_ordering",
            "success": False,
            "terminated_reason": "deadlock",
            "steps": 1,
            "total_tokens": 100,
            "task_spec": spec.to_dict(),
            "distractor_interactions": 0,
            "irreversible_failures": 0,
            "mechanism_interaction_order": [],
            "failure_point": None,
            "transcript": [
                {
                    "kind": "step",
                    "step_index": 1,
                    "action": "MOVE_FORWARD",
                    "state_before": before,
                    "state_after": after,
                },
            ],
        },
        static_score=static_score,
        canonical_paths=canonical,
        difficulty_max_static_score=static_score.static_score,
    )

    assert score.signals["distractor_interactions"] == 1
    assert score.signals["irreversible_failures"] == 1
    assert score.signals["mechanism_interaction_order"] == ["push:block:2,1"]
    assert score.signals["failure_point"]["step"] == 1


def test_runtime_score_requires_transition_snapshots_for_test_three():
    spec = make_spec()
    canonical = compute_canonical_paths(spec)
    static_score = compute_static_score_artifact(spec)

    with pytest.raises(ValueError, match="transition telemetry"):
        compute_runtime_score(
            {
                "experiment": "mechanism_ordering",
                "success": True,
                "steps": 2,
                "total_tokens": 100,
                "task_spec": spec.to_dict(),
            },
            static_score=static_score,
            canonical_paths=canonical,
            difficulty_max_static_score=static_score.static_score,
        )


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


def test_runtime_zero_step_success_gets_full_step_credit():
    spec = make_spec(
        maze={
            "dimensions": [5, 5],
            "walls": [],
            "start": [1, 1],
            "goal": [1, 1],
        },
        goal={"type": "reach_position", "target": [1, 1]},
    )
    canonical = compute_canonical_paths(spec)
    static_score = compute_static_score_artifact(spec)
    score = compute_runtime_score(
        {
            "success": True,
            "steps": 0,
            "total_tokens": 100,
            "task_spec": spec.to_dict(),
            "initial_state": {"agent_position": [1, 1]},
            "final_state": {"agent_position": [1, 1], "step_count": 0},
        },
        static_score=static_score,
        canonical_paths=canonical,
        config=ScorerConfig.from_dict({"runtime_weights": {"greedy_penalty": 0.0}}),
        difficulty_max_static_score=static_score.static_score,
    )

    assert score.signals["step_ratio"] == 1.0
    assert score.composite == 1.0


def test_static_cli_target_dirs_reject_same_stem_collisions(tmp_path):
    files = [tmp_path / "a" / "task.json", tmp_path / "b" / "task.json"]

    with pytest.raises(ValueError, match="collide"):
        _static_target_dirs(files, tmp_path / "scores")


def test_runtime_cli_default_output_uses_source_stem(tmp_path):
    assert _default_runtime_output(tmp_path / "run.json") == tmp_path / "run_score.json"
    assert _default_runtime_output(tmp_path / "episode.json") == tmp_path / "episode_score.json"


def test_runtime_cli_rejects_half_specified_artifacts(tmp_path):
    args = argparse.Namespace(
        config=None,
        run=str(tmp_path / "episode.json"),
        output=None,
        static_score=str(tmp_path / "scored_static.json"),
        canonical_paths=None,
        task=str(tmp_path / "task.json"),
        artifact_dir=None,
        difficulty_max_static_score=100.0,
    )

    with pytest.raises(ValueError, match="provided together"):
        _runtime(args)


def test_runtime_cli_explains_missing_suite_maximum(tmp_path):
    args = argparse.Namespace(
        config=None,
        run=str(tmp_path / "episode.json"),
        output=None,
        static_score=str(tmp_path / "scored_static.json"),
        canonical_paths=str(tmp_path / "canonical_paths.json"),
        task=None,
        artifact_dir=None,
        difficulty_max_static_score=None,
    )

    with pytest.raises(ValueError, match="--difficulty-max-static-score"):
        _runtime(args)


def test_visualize_dimensions_validates_before_writing_csv(tmp_path):
    payload = compute_static_score_artifact(make_spec()).to_dict()
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(payload))
    second.write_text(json.dumps(payload))
    csv_path = tmp_path / "scores.csv"

    with pytest.raises(ValueError, match="exactly one"):
        visualize_scores_main(
            [
                str(first),
                str(second),
                "--kind",
                "dimensions",
                "--csv",
                str(csv_path),
            ]
        )

    assert not csv_path.exists()


def test_artifact_serialization_returns_detached_data():
    scored = ScoredDifficulty(dimensions=[1.0], dimension_names=["only"], weights=[2.0])
    scored_payload = scored.to_dict()
    scored_payload["dimensions"][0] = 9.0
    scored_payload["weights"][0] = 9.0

    canonical = CanonicalPathReport(
        task_id="task",
        success=True,
        actions=["move_forward"],
        positions=[(1, 1), (2, 1)],
        optimal_steps=1,
        states_explored=2,
        message="ok",
        greedy={"actions": ["move_forward"]},
    )
    canonical_payload = canonical.to_dict()
    canonical_payload["bfs"]["actions"][0] = "mutated"
    canonical_payload["greedy"]["actions"][0] = "mutated"

    assert scored.dimensions == [1.0]
    assert scored.weights == [2.0]
    assert canonical.actions == ["move_forward"]
    assert canonical.greedy == {"actions": ["move_forward"]}


def test_stage_five_reports_cover_calibration_and_experiment_outputs():
    first = compute_static_score_artifact(make_spec()).to_dict()
    second = compute_static_score_artifact(
        make_spec(task_id="scorer_case_two", difficulty_tier=2)
    ).to_dict()
    reports = build_experiment_reports(
        [first, second],
        [
            {
                "task_id": "scorer_case",
                "experiment": "complexity_distance",
                "condition": "short",
                "model_id": "unit",
                "seed": 1,
                "signals": {
                    "success": True,
                    "optimality_ratio": 1.0,
                    "path_choice": "short",
                },
                "composite": 1.0,
            },
            {
                "task_id": "scorer_case",
                "experiment": "mechanism_ordering",
                "condition": "ordered",
                "pair_id": "pair-a",
                "model_id": "unit",
                "seed": 1,
                "signals": {
                    "success": True,
                    "optimality_ratio": 1.0,
                    "mechanism_interaction_order": ["toggle:switch"],
                    "failure_point": None,
                },
                "composite": 1.0,
            },
            {
                "task_id": "scorer_case_two",
                "experiment": "mechanism_ordering",
                "condition": "reversed",
                "pair_id": "pair-a",
                "model_id": "unit",
                "seed": 1,
                "signals": {
                    "success": False,
                    "optimality_ratio": 0.0,
                    "mechanism_interaction_order": [],
                    "failure_point": {"step": 1},
                },
                "composite": 0.0,
            },
        ],
    )

    calibration = reports["scoring_calibration_summary.json"]
    complexity = reports["complexity_distance_summary.json"]
    ordering = reports["mechanism_ordering_pairs.json"]
    assert "grid_size" in calibration["dimension_correlation_matrix"]
    assert calibration["tier_boundary_candidates"]
    assert complexity["path_choice_counts"][0]["path_choice"] == "short"
    assert ordering["paired_deltas"][0]["paired_success_deltas"] == {
        "reversed_minus_ordered": -1.0,
    }


def test_stage_five_rejects_test_two_rows_without_path_choice():
    static_score = compute_static_score_artifact(make_spec()).to_dict()

    with pytest.raises(ValueError, match="path_choice"):
        build_experiment_reports(
            [static_score],
            [
                {
                    "task_id": "scorer_case",
                    "experiment": "complexity_distance",
                    "condition": "short",
                    "model_id": "unit",
                    "signals": {"success": True, "optimality_ratio": 1.0},
                    "composite": 1.0,
                },
            ],
        )
