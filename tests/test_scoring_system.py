import argparse
import json

import pytest

from gridworld.actions import MiniGridActions
from gridworld.baselines import plan_bfs_path, trace_planned_actions
from gridworld.task_spec import TaskSpecification
from gridworld.task_validator import TaskValidator
from scorer.artifacts import CanonicalPathReport, ScoredDifficulty
from scorer.config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DISTRACTOR_TYPE_WEIGHTS,
    DEFAULT_RUNTIME_WEIGHTS,
    DIMENSION_NAMES,
    load_scorer_config,
)
from scorer.io import dump_json, load_json
from scorer.scoring import (
    ScorerConfig,
    compute_12d_score,
    compute_canonical_paths,
    compute_runtime_score,
    compute_static_score_artifact,
    score_task_file,
)
from scripts.score_json import _default_runtime_output, _runtime, _static, _static_target_dirs


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


def test_planner_toggle_trace_matches_current_cell_switch_precedence():
    spec = make_spec(
        maze={
            "dimensions": [7, 5],
            "walls": [[1, 2], [2, 2], [3, 2], [4, 2], [5, 2]],
            "start": [1, 1],
            "goal": [5, 1],
        },
        mechanisms={
            "keys": [{"id": "k1", "position": [2, 1], "color": "red"}],
            "doors": [
                {
                    "id": "d1",
                    "position": [4, 1],
                    "requires_key": "red",
                    "initial_state": "locked",
                }
            ],
            "switches": [
                {
                    "id": "s1",
                    "position": [3, 1],
                    "controls": [],
                    "switch_type": "toggle",
                    "initial_state": "off",
                }
            ],
        },
        goal={"type": "reach_position", "target": [5, 1]},
        max_steps=30,
    )

    traced = trace_planned_actions(
        spec,
        [
            int(MiniGridActions.PICKUP),
            int(MiniGridActions.MOVE_FORWARD),
            int(MiniGridActions.MOVE_FORWARD),
            int(MiniGridActions.TOGGLE),
        ],
    )
    bfs_path = plan_bfs_path(spec)

    assert traced.action_labels[-1] == "toggle:s1"
    assert "open_door:d1" not in traced.action_labels
    assert bfs_path.success is False


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


def test_explicit_missing_config_path_fails(tmp_path):
    with pytest.raises(FileNotFoundError, match="Scorer config not found"):
        load_scorer_config(tmp_path / "missing_config.json")


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
    with open(scored_path, encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["task_id"] == spec.task_id
    assert "dimensions_12" in payload
    assert "dimensions" not in payload
    assert "composite" not in payload
    assert payload["validation"]["schema_valid"] is True
    assert payload["canonical_agent_features"]["greedy_solvability"] == 1.0


def test_scorer_json_io_uses_utf8_encoding(tmp_path, monkeypatch):
    real_open = open
    observed: list[tuple[str, str, str | None]] = []

    def tracking_open(path, mode="r", *args, **kwargs):
        observed.append((str(path), mode, kwargs.get("encoding")))
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", tracking_open)

    payload = {"message": "reach \u2192 caf\u00e9", "label": "caf\u00e9"}
    path = tmp_path / "unicode.json"
    dump_json(path, payload)

    assert load_json(path) == payload
    assert (str(path), "w", "utf-8") in observed
    assert (str(path), "r", "utf-8") in observed


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


def test_static_score_uses_canonical_bfs_metrics():
    spec = make_spec()
    bfs_path = plan_bfs_path(spec)
    score = compute_12d_score(spec, bfs_path=bfs_path)

    assert score.dimensions[0] == len(bfs_path.action_labels)
    assert score.dimensions[1] == bfs_path.states_explored


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


def test_runtime_hash_ignores_non_scoring_transcript_context():
    spec = make_spec()
    canonical = compute_canonical_paths(spec)
    static_score = compute_static_score_artifact(spec)
    base_run = {
        "success": True,
        "steps": 2,
        "total_tokens": 100,
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


def test_static_cli_continues_after_file_failure_and_summarizes(tmp_path, capsys):
    task_a = tmp_path / "task_a.json"
    task_b = tmp_path / "task_b.json"
    bad_task = tmp_path / "bad.json"
    dump_json(task_a, make_spec(task_id="ok_a").to_dict())
    dump_json(task_b, make_spec(task_id="ok_b").to_dict())
    bad_task.write_text("{", encoding="utf-8")

    exit_code = _static(
        argparse.Namespace(
            config=None,
            inputs=[str(task_a), str(bad_task), str(task_b)],
            output_dir=str(tmp_path / "scores"),
        )
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "static: ok input=" in captured.out
    assert "task_id=ok_a" in captured.out
    assert "task_id=ok_b" in captured.out
    assert "static: error input=" in captured.err
    assert "bad.json" in captured.err
    assert "JSONDecodeError" in captured.err
    assert "Traceback" not in captured.err
    assert "static: summary scored=2 failed=1 total=3" in captured.err
    assert (tmp_path / "scores" / "task_a" / "scored_static.json").exists()
    assert (tmp_path / "scores" / "task_b" / "scored_static.json").exists()
    assert not (tmp_path / "scores" / "bad" / "scored_static.json").exists()


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
