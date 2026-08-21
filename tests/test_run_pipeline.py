"""End-to-end test for the bare-bones run pipeline using a replay stub agent.

Runs are live-model-only in production, but the runner accepts any callable
``messages -> str`` agent, so a deterministic replay stub exercises the full
Stage 1->5 chain (real MiniGrid backend, episode log, and scorer) with no API.
"""

from __future__ import annotations

import itertools
import io
import json
import shutil
import tarfile
from collections import Counter
from pathlib import Path

import pytest

from interface.loader import default_maze_path
from interface.smoke_tests.plans import v01_empty_room_trajectory
from scorer import load_scorer_config, score_task_file
from scorer.io import load_json, task_spec_from_payload

from scripts.run_pipeline import (
    _condition_configs,
    _expected_run_hash,
    _expected_static_hash,
    _runtime_capped_spec,
    _resolve_source,
    check_run_config_expectations,
    condition_variant_names,
    load_manifest,
    load_run_config,
    resolve_task_rows,
    run_from_config,
    run_pipeline,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIXTURES = _REPO_ROOT / "gridworld" / "fixtures"
_MANIFEST = _FIXTURES / "manifest.json"
_OGBENCH_50_MANIFEST = _FIXTURES / "manifest.ogbench_50_smbd.json"
_CONDITIONAL_EVAL_MANIFEST = _FIXTURES / "manifest.conditional_eval.json"
_SMOKE_EVAL_MANIFEST = _FIXTURES / "manifest.smoke_eval.json"
_COORDINATOR_SMOKE_MANIFEST = _FIXTURES / "manifest.coordinator_smoke_validation10_ogbench.json"
_COORDINATOR_SMOKE_RUN_CONFIG = _FIXTURES / "run_config.coordinator_smoke_qwen_kimi.json"
_PENDING_VALIDATION10_CONFIGS = {
    "Prompt": _FIXTURES / "run_config.validation10_prompt_claude_kimi_qwen.json",
    "Observation format": _FIXTURES / "run_config.validation10_observation_format_claude_kimi_qwen.json",
    "Context window": _FIXTURES / "run_config.validation10_context_window_claude_kimi_qwen.json",
    "Querying strategy": _FIXTURES / "run_config.validation10_querying_strategy_claude_kimi_qwen.json",
}
_CONDITIONAL_CONFIGS = {
    "Prompt": _FIXTURES / "run_config.conditional_prompt_claude_kimi_qwen.json",
    "Observation format": _FIXTURES / "run_config.conditional_observation_format_claude_kimi_qwen.json",
    "Context window": _FIXTURES / "run_config.conditional_context_window_claude_kimi_qwen.json",
    "Action space": _FIXTURES / "run_config.conditional_action_space_claude_kimi_qwen.json",
    "Querying strategy": _FIXTURES / "run_config.conditional_querying_strategy_claude_kimi_qwen.json",
    "In-context learning": _FIXTURES / "run_config.conditional_in_context_learning_claude_kimi_qwen.json",
    "History mechanism": _FIXTURES / "run_config.conditional_history_mechanism_claude_kimi_qwen.json",
}
_SMOKE_EVAL_RUN_CONFIG = _FIXTURES / "run_config.smoke_eval_qwen_kimi.json"
_STABLE_DIFFICULTY_MAX = 1000.0


class ReplayAgent:
    """Replays a fixed action plan and reports token usage (scorer needs >0)."""

    def __init__(self, actions):
        self._actions = iter(actions)
        self.last_usage = None

    def __call__(self, messages):
        self.last_usage = {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}
        try:
            action = next(self._actions)
        except StopIteration:
            action = "DONE"
        return f"FINAL_OUTPUT: {action}"


def _write_manifest(tmp_path: Path) -> Path:
    manifest = {
        "tasks": [
            {
                "task_id": "validation_10_v01_empty_room",
                "experiment": "test1",
                "condition": "default",
                "variant": "empty_room",
                "source": str(default_maze_path("V01_empty_room.json")),
                "expected_mechanisms": [],
                "notes": "E2E smoke task.",
            }
        ]
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_pipeline_writes_full_artifact_tree(tmp_path):
    manifest_path = _write_manifest(tmp_path)
    artifacts = tmp_path / "artifacts"

    payloads = run_pipeline(
        manifest_path=manifest_path,
        experiment="test1",
        agent=ReplayAgent(v01_empty_room_trajectory()),
        agent_name="replay-stub",
        seeds=[0],
        conditions=None,
        artifacts_root=artifacts,
        run_set_id="smoke",
        difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
    )

    task_id = "validation_10_v01_empty_room"
    task_dir = artifacts / "tasks" / task_id
    assert (task_dir / "canonical_paths.json").exists()
    assert (task_dir / "scored_static.json").exists()
    assert (artifacts / "tasks" / "_suite.json").exists()

    run_dir = artifacts / "runs" / task_id / "minigrid" / "replay-stub" / "seed_0" / "default"
    assert (run_dir / "episode.json").exists()
    run_score = json.loads((run_dir / "run_score.json").read_text())
    assert "composite" in run_score
    assert run_score["signals"]["success"] is True

    jsonl = (artifacts / "episode_runs.jsonl").read_text().strip().splitlines()
    assert len(jsonl) == 1
    row = json.loads(jsonl[0])
    for field in (
        "task_id", "experiment", "condition", "prompt_variant", "backend",
        "agent_or_model", "seed", "success", "terminated", "truncated", "reward",
        "steps", "optimal_steps", "optimality_ratio", "path_choice",
        "mechanism_interaction_order", "failure_point", "tokens", "raw_output_ref",
    ):
        assert field in row, f"missing episode_runs field: {field}"
    assert row["prompt_variant"] == "default"
    assert row["tokens"] and row["tokens"] > 0

    report_dir = artifacts / "reports" / "smoke"
    for name in (
        "scoring_calibration_summary",
        "complexity_distance_summary",
        "mechanism_ordering_pairs",
    ):
        assert (report_dir / f"{name}.json").exists()
    assert payloads["scoring_calibration_summary"]["run_count"] == 1


def test_pipeline_requires_stable_difficulty_max(tmp_path):
    manifest_path = _write_manifest(tmp_path)

    with pytest.raises(ValueError, match="difficulty_max_static_score"):
        run_pipeline(
            manifest_path=manifest_path,
            experiment="test1",
            agent=ReplayAgent(v01_empty_room_trajectory()),
            agent_name="replay-stub",
            seeds=[0],
            artifacts_root=tmp_path / "artifacts",
            run_set_id="smoke",
        )


def test_pipeline_uses_configured_difficulty_max(tmp_path):
    manifest_path = _write_manifest(tmp_path)
    artifacts = tmp_path / "artifacts"
    config = load_scorer_config()
    config.difficulty_max_static_score = _STABLE_DIFFICULTY_MAX

    run_pipeline(
        manifest_path=manifest_path,
        experiment="test1",
        agent=ReplayAgent(v01_empty_room_trajectory()),
        agent_name="replay-stub",
        seeds=[0],
        artifacts_root=artifacts,
        run_set_id="smoke",
        scorer_config=config,
    )

    task_id = "validation_10_v01_empty_room"
    suite = load_json(artifacts / "tasks" / "_suite.json")
    static_score = load_json(artifacts / "tasks" / task_id / "scored_static.json")
    run_score = load_json(
        artifacts / "runs" / task_id / "minigrid" / "replay-stub" / "seed_0" / "default" / "run_score.json"
    )
    assert suite["difficulty_max_static_score"] == _STABLE_DIFFICULTY_MAX
    assert run_score["signals"]["difficulty_weight"] == pytest.approx(
        static_score["static_score"] / _STABLE_DIFFICULTY_MAX
    )


# --------------------------------------------------------------------------- #
# Task resolution (run-config entries -> catalog rows with metadata)
# --------------------------------------------------------------------------- #
def _catalog():
    return json.loads(_MANIFEST.read_text())["tasks"]


def test_resolve_experiment_keyword_expands_from_catalog():
    rows = resolve_task_rows(["test3"], _catalog(), _MANIFEST)
    assert rows and all(r["experiment"] == "test3" for r in rows)
    assert {"T3_corr_key_first", "T3_corr_switch_first"} <= {r["task_id"] for r in rows}


def test_resolve_task_file_attaches_catalog_metadata():
    rows = resolve_task_rows(
        ["gridworld/fixtures/test3/T3_corr_key_first.json"], _catalog(), _MANIFEST
    )
    assert len(rows) == 1
    assert rows[0]["task_id"] == "T3_corr_key_first"
    assert rows[0]["expected_mechanisms"] == ["kB", "s1"]
    assert rows[0]["pair_id"] == "corridor"


def test_resolve_unknown_file_synthesizes_test1_row(tmp_path):
    task_file = str(default_maze_path("V01_empty_room.json"))
    rows = resolve_task_rows([task_file], _catalog(), _MANIFEST)
    # V01 is in the catalog by path -> keeps its catalog task_id.
    assert rows[0]["task_id"] == "validation_10_v01_empty_room"


def test_validate_fixtures_reports_missing_source_without_traceback(tmp_path, capsys):
    from scripts.validate_fixtures import main as validate_fixtures_main

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": "missing_task",
                        "experiment": "test1",
                        "source": "does_not_exist.json",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    exit_code = validate_fixtures_main(["--manifest", str(manifest_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Fixture validation FAILED:" in captured.out
    assert "missing_task: Task source not found: does_not_exist.json" in captured.out
    assert "Traceback" not in captured.out
    assert "Traceback" not in captured.err


def test_pending_validation10_condition_run_configs_load_and_resolve_all_tasks():
    catalog = _catalog()
    expected_variant_counts = {
        "Prompt": 3,
        "Observation format": 3,
        "Context window": 4,
        "Querying strategy": 3,
    }

    for condition_name, config_path in _PENDING_VALIDATION10_CONFIGS.items():
        cfg = load_run_config(config_path)
        assert set(cfg["models"]) == {"qwen35_27b_hf", "kimi_k26", "claude_sonnet"}
        assert {m["provider"] for m in cfg["models"].values()} == {"qwen", "kimi", "claude"}
        assert len(condition_variant_names(condition_name)) == expected_variant_counts[condition_name]

        for model_cfg in cfg["models"].values():
            assert model_cfg["tasks"] == ["all"]
            rows = resolve_task_rows(model_cfg["tasks"], catalog, _MANIFEST)
            assert [r["task_id"] for r in rows] == [r["task_id"] for r in catalog]


def test_observation_format_uses_observation_names_for_run_dirs():
    assert condition_variant_names("Observation format") == [
        "image_only",
        "text_only",
        "image_text",
    ]


def test_coordinator_smoke_run_config_loads_and_resolves_smoke_manifest():
    cfg = load_run_config(_COORDINATOR_SMOKE_RUN_CONFIG)
    catalog = json.loads(_COORDINATOR_SMOKE_MANIFEST.read_text(encoding="utf-8"))["tasks"]

    assert set(cfg["models"]) == {"qwen35_27b_hf", "kimi_k26"}
    assert {m["provider"] for m in cfg["models"].values()} == {"qwen", "kimi"}

    for model_cfg in cfg["models"].values():
        assert model_cfg["tasks"] == ["all"]
        rows = resolve_task_rows(model_cfg["tasks"], catalog, _COORDINATOR_SMOKE_MANIFEST)
        assert len(rows) == 14

    assert len(catalog) * len(cfg["models"]) * len(condition_variant_names(None)) == 28


def test_coordinator_smoke_manifest_selection_metadata_and_holdouts():
    smoke = json.loads(_COORDINATOR_SMOKE_MANIFEST.read_text(encoding="utf-8"))
    smoke_rows = smoke["tasks"]
    base_rows = _catalog()
    ogbench50_sources = {
        r["source"]
        for r in json.loads(_OGBENCH_50_MANIFEST.read_text(encoding="utf-8"))["tasks"]
    }

    validation_rows = [r for r in smoke_rows if r["task_id"].startswith("validation_10_")]
    expected_validation_rows = [r for r in base_rows if r["experiment"] == "test1"]
    assert [r["task_id"] for r in validation_rows] == [r["task_id"] for r in expected_validation_rows]

    holdout_rows = [r for r in smoke_rows if r.get("experiment") == "coordinator_smoke_ogbench"]
    assert [r["source"] for r in holdout_rows] == [
        "ogbench/ogbench/procgen/maze_jsons/S5/14x14_corridor_1.json",
        "ogbench/ogbench/procgen/maze_jsons/M1/10x10_corridor_kr_0.json",
        "ogbench/ogbench/procgen/maze_jsons/D1/10x10_corridor_wrong_ky_kr_0.json",
        "ogbench/ogbench/procgen/maze_jsons/D1/10x10_corridor_wrong_ky_kr_1.json",
    ]
    assert not ({r["source"] for r in holdout_rows} & ogbench50_sources)

    family_counts = Counter(r["maze_family"] for r in holdout_rows)
    assert dict(family_counts) == {"S": 1, "M": 1, "D": 2}
    assert smoke["selection"]["counts"] == {"validation_10": 10, "S": 1, "M": 1, "B": 0, "D": 2}
    assert "B-family" in smoke["selection"]["b_family_omission"]


def test_conditional_eval_manifest_is_15_mazes_held_out_from_the_50_run():
    manifest = json.loads(_CONDITIONAL_EVAL_MANIFEST.read_text(encoding="utf-8"))
    rows = manifest["tasks"]
    assert len(rows) == 15

    # 10 validation_10 rows (same as the base catalog's test1 rows) ...
    val_rows = [r for r in rows if r["task_id"].startswith("validation_10_")]
    expected_val = [r for r in _catalog() if r["experiment"] == "test1"]
    assert [r["task_id"] for r in val_rows] == [r["task_id"] for r in expected_val]

    # ... plus 5 conditional S/M/B/D/D rows.
    conditional = [r for r in rows if r.get("experiment") == "conditional"]
    assert dict(Counter(r["maze_family"] for r in conditional)) == {"S": 1, "M": 1, "B": 1, "D": 2}

    # Every added maze must be held out from the 50-maze run so the verification
    # experiments cannot affect it.
    ogbench50_sources = {
        r["source"] for r in json.loads(_OGBENCH_50_MANIFEST.read_text(encoding="utf-8"))["tasks"]
    }
    assert not ({r["source"] for r in conditional} & ogbench50_sources)

    # The B row is the held-out blind probe maze.
    b_rows = [r for r in conditional if r["maze_family"] == "B"]
    assert len(b_rows) == 1
    assert b_rows[0]["source"] == "mazes/conditional/blind_probe_B_holdout.json"
    assert (_REPO_ROOT / b_rows[0]["source"]).exists()
    assert manifest["selection"]["counts"] == {"validation_10": 10, "S": 1, "M": 1, "B": 1, "D": 2}


def test_smoke_eval_manifest_is_three_resolvable_mazes():
    manifest = json.loads(_SMOKE_EVAL_MANIFEST.read_text(encoding="utf-8"))
    rows = manifest["tasks"]
    assert len(rows) == 3
    for row in rows:
        assert (_REPO_ROOT / row["source"]).exists()
    resolved = resolve_task_rows(["all"], rows, _SMOKE_EVAL_MANIFEST)
    assert [r["task_id"] for r in resolved] == [r["task_id"] for r in rows]


# --------------------------------------------------------------------------- #
# Config-driven multi-model run (stub agent factory, no API)
# --------------------------------------------------------------------------- #
def test_run_from_config_drives_per_model_tasks(tmp_path):
    run_config = {
        "models": {
            "stub": {
                "provider": "claude",
                "model": "stub-model",
                "tasks": [str(default_maze_path("V01_empty_room.json"))],
            }
        }
    }
    cfg_path = tmp_path / "run_config.json"
    cfg_path.write_text(json.dumps(run_config), encoding="utf-8")
    artifacts = tmp_path / "artifacts"

    def factory(name, model_cfg):
        return ReplayAgent(v01_empty_room_trajectory()), model_cfg["model"]

    payloads = run_from_config(
        run_config_path=cfg_path,
        manifest_path=_MANIFEST,
        seeds=[0],
        artifacts_root=artifacts,
        run_set_id="cfg",
        agent_factory=factory,
        difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
    )

    run_dir = (
        artifacts / "runs" / "validation_10_v01_empty_room" / "minigrid" / "stub-model" / "seed_0" / "default"
    )
    assert (run_dir / "episode.json").exists()
    assert (run_dir / "run_score.json").exists()
    assert payloads["scoring_calibration_summary"]["run_count"] == 1


# --------------------------------------------------------------------------- #
# Content-hash invalidation
# --------------------------------------------------------------------------- #
class CountingReplayAgent:
    """Cycles a fixed plan (one full pass per episode) and counts model calls."""

    def __init__(self, actions):
        self._actions = itertools.cycle(actions)
        self.last_usage = None
        self.calls = 0

    def __call__(self, messages):
        self.calls += 1
        self.last_usage = {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}
        return f"FINAL_OUTPUT: {next(self._actions)}"


def _single_task_manifest(tmp_path, source):
    manifest = {"tasks": [{
        "task_id": "copy_v01", "experiment": "test1", "condition": "default",
        "variant": "copy", "source": str(source), "expected_mechanisms": [],
    }]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_expected_static_hash_matches_scorer(tmp_path):
    source = default_maze_path("V06_chain_ks.json")
    cfg = load_scorer_config()
    _, static = score_task_file(source, output_dir=tmp_path / "t", config=cfg)
    spec = task_spec_from_payload(load_json(source))
    assert _expected_static_hash(spec, cfg) == static.to_dict()["inputs_hash"]


def test_canonical_paths_carry_inputs_hash(tmp_path):
    source = default_maze_path("V06_chain_ks.json")
    score_task_file(source, output_dir=tmp_path / "t")
    canonical = load_json(tmp_path / "t" / "canonical_paths.json")
    assert canonical.get("inputs_hash")


def test_runtime_cap_clamps_episode_max_steps_to_three_times_optimal(tmp_path):
    task_file = tmp_path / "task.json"
    shutil.copy(default_maze_path("V01_empty_room.json"), task_file)
    manifest = _single_task_manifest(tmp_path, task_file)
    artifacts = tmp_path / "artifacts"

    run_pipeline(
        manifest_path=manifest,
        experiment="test1",
        agent=ReplayAgent(v01_empty_room_trajectory()),
        agent_name="stub",
        artifacts_root=artifacts,
        run_set_id="r",
        difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
    )

    run_dir = artifacts / "runs" / "copy_v01" / "minigrid" / "stub" / "seed_0" / "default"
    episode = load_json(run_dir / "episode.json")
    sidecar = load_json(run_dir / "run_inputs.json")

    assert episode["task_spec"]["max_steps"] == 33
    assert episode["initial_state"]["max_steps"] == 33
    assert sidecar["runtime_max_steps_cap"] == {
        "multiplier": 3,
        "optimal_steps": 11,
        "original_max_steps": 100,
        "effective_max_steps": 33,
    }


def test_runtime_cap_does_not_raise_existing_smaller_max_steps():
    spec = task_spec_from_payload(load_json(default_maze_path("V01_empty_room.json")))
    canonical = {"bfs": {"optimal_steps": 11}}
    lowered = spec.__class__.from_dict({**spec.to_dict(), "max_steps": 20})

    capped = _runtime_capped_spec(lowered, canonical)

    assert capped.max_steps == 20


def test_unchanged_rerun_reuses_episode_and_static(tmp_path):
    task_file = tmp_path / "task.json"
    shutil.copy(default_maze_path("V01_empty_room.json"), task_file)
    manifest = _single_task_manifest(tmp_path, task_file)
    artifacts = tmp_path / "artifacts"
    agent = CountingReplayAgent(v01_empty_room_trajectory())

    run_pipeline(manifest_path=manifest, experiment="test1", agent=agent,
                 agent_name="stub", artifacts_root=artifacts, run_set_id="r",
                 difficulty_max_static_score=_STABLE_DIFFICULTY_MAX)
    calls_after_first = agent.calls
    assert calls_after_first > 0

    # Second identical run: episode cache hit -> agent not called again.
    run_pipeline(manifest_path=manifest, experiment="test1", agent=agent,
                 agent_name="stub", artifacts_root=artifacts, run_set_id="r",
                 difficulty_max_static_score=_STABLE_DIFFICULTY_MAX)
    assert agent.calls == calls_after_first


def test_run_config_generation_settings_invalidate_episode_cache(tmp_path):
    task_file = tmp_path / "task.json"
    shutil.copy(default_maze_path("V01_empty_room.json"), task_file)
    artifacts = tmp_path / "artifacts"
    cfg_path = tmp_path / "run_config.json"
    agent = CountingReplayAgent(v01_empty_room_trajectory())

    def write_run_config(max_tokens: int) -> None:
        cfg_path.write_text(
            json.dumps(
                {
                    "models": {
                        "stub": {
                            "provider": "claude",
                            "model": "stub-model",
                            "temperature": 0.0,
                            "max_tokens": max_tokens,
                            "tasks": [str(task_file)],
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

    def factory(name, model_cfg):
        return agent, model_cfg["model"]

    write_run_config(128)
    run_from_config(
        run_config_path=cfg_path,
        manifest_path=_MANIFEST,
        seeds=[0],
        artifacts_root=artifacts,
        run_set_id="r",
        agent_factory=factory,
        difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
    )
    calls_after_first = agent.calls
    run_dir = (
        artifacts / "runs" / "task" / "minigrid" / "stub-model" / "seed_0" / "default"
    )
    first_sidecar = load_json(run_dir / "run_inputs.json")
    assert first_sidecar["model_config"]["max_tokens"] == 128

    write_run_config(4096)
    run_from_config(
        run_config_path=cfg_path,
        manifest_path=_MANIFEST,
        seeds=[0],
        artifacts_root=artifacts,
        run_set_id="r",
        agent_factory=factory,
        difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
    )
    second_sidecar = load_json(run_dir / "run_inputs.json")

    assert agent.calls > calls_after_first
    assert second_sidecar["model_config"]["max_tokens"] == 4096
    assert second_sidecar["runtime_model_config"]["max_tokens"] == 4096
    assert second_sidecar["inputs_hash"] != first_sidecar["inputs_hash"]


def test_corrupted_sidecar_reruns_episode_cache(tmp_path):
    task_file = tmp_path / "task.json"
    shutil.copy(default_maze_path("V01_empty_room.json"), task_file)
    manifest = _single_task_manifest(tmp_path, task_file)
    artifacts = tmp_path / "artifacts"
    agent = CountingReplayAgent(v01_empty_room_trajectory())

    run_pipeline(manifest_path=manifest, experiment="test1", agent=agent,
                 agent_name="stub", artifacts_root=artifacts, run_set_id="r",
                 difficulty_max_static_score=_STABLE_DIFFICULTY_MAX)
    calls_after_first = agent.calls
    sidecar = artifacts / "runs" / "copy_v01" / "minigrid" / "stub" / "seed_0" / "default" / "run_inputs.json"
    sidecar.write_text("{", encoding="utf-8")

    run_pipeline(manifest_path=manifest, experiment="test1", agent=agent,
                 agent_name="stub", artifacts_root=artifacts, run_set_id="r",
                 difficulty_max_static_score=_STABLE_DIFFICULTY_MAX)

    assert agent.calls > calls_after_first
    assert load_json(sidecar)["inputs_hash"]


def test_max_steps_edit_above_runtime_cap_rescores_without_rerunning_episode(tmp_path):
    task_file = tmp_path / "task.json"
    shutil.copy(default_maze_path("V01_empty_room.json"), task_file)
    manifest = _single_task_manifest(tmp_path, task_file)
    artifacts = tmp_path / "artifacts"
    agent = CountingReplayAgent(v01_empty_room_trajectory())

    run_pipeline(manifest_path=manifest, experiment="test1", agent=agent,
                 agent_name="stub", artifacts_root=artifacts, run_set_id="r",
                 difficulty_max_static_score=_STABLE_DIFFICULTY_MAX)
    first_calls = agent.calls
    first_static_hash = load_json(artifacts / "tasks" / "copy_v01" / "scored_static.json")["inputs_hash"]

    # Mutate only the authored max_steps above the runtime cap. Static scoring
    # sees the original task and must refresh, but Stage 3 hashes/runs the
    # effective capped spec, which is unchanged at 3 * optimal.
    data = json.loads(task_file.read_text())
    data["max_steps"] = data["max_steps"] + 5
    task_file.write_text(json.dumps(data), encoding="utf-8")

    run_pipeline(manifest_path=manifest, experiment="test1", agent=agent,
                 agent_name="stub", artifacts_root=artifacts, run_set_id="r",
                 difficulty_max_static_score=_STABLE_DIFFICULTY_MAX)
    new_static_hash = load_json(artifacts / "tasks" / "copy_v01" / "scored_static.json")["inputs_hash"]
    assert new_static_hash != first_static_hash  # Stage 2 recomputed
    assert agent.calls == first_calls            # Stage 3 episode reused


def test_task_geometry_edit_invalidates_static_and_episode(tmp_path):
    task_file = tmp_path / "task.json"
    shutil.copy(default_maze_path("V01_empty_room.json"), task_file)
    manifest = _single_task_manifest(tmp_path, task_file)
    artifacts = tmp_path / "artifacts"
    agent = CountingReplayAgent(v01_empty_room_trajectory())

    run_pipeline(manifest_path=manifest, experiment="test1", agent=agent,
                 agent_name="stub", artifacts_root=artifacts, run_set_id="r",
                 difficulty_max_static_score=_STABLE_DIFFICULTY_MAX)
    first_calls = agent.calls
    first_static_hash = load_json(artifacts / "tasks" / "copy_v01" / "scored_static.json")["inputs_hash"]

    data = json.loads(task_file.read_text())
    data["maze"]["goal"] = [6, 5]
    data["goal"]["target"] = [6, 5]
    task_file.write_text(json.dumps(data), encoding="utf-8")

    run_pipeline(manifest_path=manifest, experiment="test1", agent=agent,
                 agent_name="stub", artifacts_root=artifacts, run_set_id="r",
                 difficulty_max_static_score=_STABLE_DIFFICULTY_MAX)
    new_static_hash = load_json(artifacts / "tasks" / "copy_v01" / "scored_static.json")["inputs_hash"]
    assert new_static_hash != first_static_hash
    assert agent.calls > first_calls


def test_scorer_config_change_rescore_without_rerunning_model(tmp_path):
    task_file = tmp_path / "task.json"
    shutil.copy(default_maze_path("V01_empty_room.json"), task_file)
    manifest = _single_task_manifest(tmp_path, task_file)
    artifacts = tmp_path / "artifacts"
    agent = CountingReplayAgent(v01_empty_room_trajectory())

    # Small baselines (below the run's token count) so token_efficiency stays < 1
    # and actually moves with the config.
    cfg_a = load_scorer_config()
    cfg_a.baseline_tokens = 1.0
    cfg_a.difficulty_max_static_score = _STABLE_DIFFICULTY_MAX
    run_pipeline(manifest_path=manifest, experiment="test1", agent=agent, agent_name="stub",
                 artifacts_root=artifacts, run_set_id="r", scorer_config=cfg_a)
    calls_after_first = agent.calls
    run_dir = artifacts / "runs" / "copy_v01" / "minigrid" / "stub" / "seed_0" / "default"
    eff_a = load_json(run_dir / "run_score.json")["signals"]["token_efficiency"]

    cfg_b = load_scorer_config()
    cfg_b.baseline_tokens = 5.0
    cfg_b.difficulty_max_static_score = _STABLE_DIFFICULTY_MAX
    run_pipeline(manifest_path=manifest, experiment="test1", agent=agent, agent_name="stub",
                 artifacts_root=artifacts, run_set_id="r", scorer_config=cfg_b)

    # Episode reused (model not re-called) but run_score reflects the new config.
    assert agent.calls == calls_after_first
    eff_b = load_json(run_dir / "run_score.json")["signals"]["token_efficiency"]
    assert eff_b != eff_a


# --------------------------------------------------------------------------- #
# Prompt variants are an axis distinct from the manifest condition
# --------------------------------------------------------------------------- #
def test_pipeline_keeps_prompt_variants_distinct(tmp_path):
    # Two prompt variants over one task must produce two distinct runs that do
    # not collapse onto the manifest condition (regression for the setdefault bug).
    manifest_path = _write_manifest(tmp_path)
    artifacts = tmp_path / "artifacts"

    payloads = run_pipeline(
        manifest_path=manifest_path,
        experiment="test1",
        agent=CountingReplayAgent(v01_empty_room_trajectory()),
        agent_name="replay-stub",
        seeds=[0],
        conditions="Prompt",  # implemented variants: standard, minimal, verbose
        artifacts_root=artifacts,
        run_set_id="variants",
        difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
    )

    task_id = "validation_10_v01_empty_room"
    base = artifacts / "runs" / task_id / "minigrid" / "replay-stub" / "seed_0"
    assert (base / "standard" / "episode.json").exists()
    assert (base / "minimal" / "episode.json").exists()
    assert (base / "verbose" / "episode.json").exists()

    rows = [
        json.loads(line)
        for line in (artifacts / "episode_runs.jsonl").read_text().strip().splitlines()
    ]
    assert {r["prompt_variant"] for r in rows} == {"standard", "minimal", "verbose"}
    # Same task-intrinsic condition, distinct prompt variants -> distinct rows.
    assert all(r["condition"] == "default" for r in rows)
    summary = payloads["scoring_calibration_summary"]
    assert summary["run_count"] == 3
    assert set(summary["success_rate_by_prompt_variant"]) == {"standard", "minimal", "verbose"}


def test_pipeline_can_run_one_condition_variant(tmp_path):
    manifest_path = _write_manifest(tmp_path)
    artifacts = tmp_path / "artifacts"

    payloads = run_pipeline(
        manifest_path=manifest_path,
        experiment="test1",
        agent=CountingReplayAgent(v01_empty_room_trajectory()),
        agent_name="replay-stub",
        seeds=[0],
        conditions="Observation format",
        prompt_variant="text_only",
        artifacts_root=artifacts,
        run_set_id="one-variant",
        difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
    )

    task_id = "validation_10_v01_empty_room"
    base = artifacts / "runs" / task_id / "minigrid" / "replay-stub" / "seed_0"
    assert (base / "text_only" / "episode.json").exists()
    assert not (base / "image_only").exists()
    assert not (base / "image_text").exists()
    assert payloads["scoring_calibration_summary"]["run_count"] == 1


def test_pipeline_writes_per_model_report(tmp_path):
    manifest_path = _write_manifest(tmp_path)
    artifacts = tmp_path / "artifacts"

    payloads = run_pipeline(
        manifest_path=manifest_path,
        experiment="test1",
        agent=ReplayAgent(v01_empty_room_trajectory()),
        agent_name="replay-stub",
        seeds=[0],
        artifacts_root=artifacts,
        run_set_id="smoke",
        difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
    )

    report_path = artifacts / "reports" / "smoke" / "models" / "replay-stub.json"
    assert report_path.exists()
    rep = json.loads(report_path.read_text())
    assert rep["schema_version"] == "0.1.0"
    assert rep["model_id"] == "replay-stub"
    assert rep["provisional"] is True
    assert rep["run_count"] == 1
    assert "overall" in rep and "by_experiment" in rep and "tasks" in rep
    assert payloads["model_reports"]["replay-stub"]["run_count"] == 1


def test_run_one_model_skips_unbeatable_tasks(tmp_path):
    # A task Stage 2 marks unbeatable must not enter Stage 3/4: no model call,
    # no run rows, no composites — without even resolving its (missing) source.
    from scripts.run_pipeline import _run_one_model

    calls = []

    def agent(messages):
        calls.append(messages)
        return "FINAL_OUTPUT: DONE"

    rows = [{"task_id": "dead", "source": "missing.json",
             "experiment": "test1", "condition": "default"}]
    run_rows, composites = _run_one_model(
        rows, agent, "m",
        manifest_path=tmp_path / "manifest.json",
        artifacts_root=tmp_path / "artifacts",
        static_by_task={"dead": {"is_beatable": False}},
        difficulty_max=1.0,
        config=load_scorer_config(),
        seeds=[0], conditions=None, force=False,
    )
    assert run_rows == []
    assert composites == {}
    assert calls == []  # ineligible task -> model never invoked


# --------------------------------------------------------------------------- #
# Distributed coordinator mode
# --------------------------------------------------------------------------- #
def _write_run_config(tmp_path: Path, models: dict) -> Path:
    path = tmp_path / "run_config.json"
    path.write_text(json.dumps({"models": models}), encoding="utf-8")
    return path


def _prepare_plan_with_overlay(tmp_path: Path, experiment_config: dict) -> dict:
    """Build a run-config with a top-level experiment_config overlay and
    resolve it into a distributed job plan."""
    from scripts.distributed_run_pipeline import prepare_job

    task = str(default_maze_path("V01_empty_room.json"))
    cfg_path = tmp_path / "run_config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "models": {
                    "stub": {
                        "provider": "claude",
                        "model": "stub-model",
                        "tasks": [task],
                    }
                },
                "experiment_config": experiment_config,
            }
        ),
        encoding="utf-8",
    )
    return prepare_job(
        run_config_path=cfg_path,
        manifest_path=_MANIFEST,
        seeds=[0],
        conditions=None,
        artifacts_root=tmp_path / "artifacts",
        run_set_id="dist",
        difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
    )


def test_distributed_units_carry_resolved_experiment_config(tmp_path):
    plan = _prepare_plan_with_overlay(tmp_path, {"progress_stall_k": 20})
    assert plan["units"], "expected units"
    for u in plan["units"]:
        assert u["experiment_config"]["progress_stall_k"] == 20
        assert u["episode_inputs_hash"]
        assert u["inputs_hash"]


def _dummy_run_archive(files: dict[str, str] | None = None) -> bytes:
    files = files or {
        "episode.json": "{}",
        "run_inputs.json": "{}",
        "run_score.json": "{}",
    }
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, content in files.items():
            raw = content.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
    return buffer.getvalue()


def test_distributed_prepare_supports_qwen_groups_without_hardcoding(tmp_path):
    from scripts.distributed_run_pipeline import prepare_job

    task = str(default_maze_path("V01_empty_room.json"))
    cfg_path = _write_run_config(
        tmp_path,
        {
            "qwen35": {
                "provider": "qwen",
                "model": "Qwen/Qwen3.5-35B",
                "group": "qwen-35b",
                "worker_count": 2,
                "hardware_profile": "a100-80gb",
                "worker_tags": ["qwen", "35b"],
                "max_in_flight": 2,
                "tasks": [task],
            },
            "qwen122": {
                "provider": "qwen",
                "model": "Qwen/Qwen3.5-122B",
                "group": "qwen-122b",
                "hardware_profile": "h100-8x",
                "worker_tags": ["qwen", "122b"],
                "tasks": [task],
            },
            "qwen36": {
                "provider": "qwen",
                "model": "Qwen/Qwen3.6-35B",
                "group": "qwen-36-35b",
                "tasks": [task],
            },
        },
    )

    plan = prepare_job(
        run_config_path=cfg_path,
        manifest_path=_MANIFEST,
        seeds=[0],
        conditions=None,
        artifacts_root=tmp_path / "artifacts",
        run_set_id="dist",
        difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
    )

    assert len(plan["units"]) == 3
    assert {u["model_group"] for u in plan["units"]} == {
        "qwen-35b", "qwen-122b", "qwen-36-35b",
    }
    assert plan["models"]["qwen122"]["hardware_profile"] == "h100-8x"
    assert plan["models"]["qwen35"]["max_in_flight"] == 2
    assert {u["model_config"]["model"] for u in plan["units"]} == {
        "Qwen/Qwen3.5-35B", "Qwen/Qwen3.5-122B", "Qwen/Qwen3.6-35B",
    }


def _valid_unit_archive(unit) -> bytes:
    """Build a minimal archive that passes the coordinator's verify (the three
    EXPECTED_RUN_FILES at root, each valid JSON)."""
    from scripts.distributed_run_pipeline import EXPECTED_RUN_FILES

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in EXPECTED_RUN_FILES:
            payload = {"unit_id": unit["unit_id"], "file": name}
            if name == "run_inputs.json":
                payload["inputs_hash"] = unit["episode_inputs_hash"]
            data = json.dumps(payload).encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_smoke_coordinator_work_steals_across_two_qwen_and_one_kimi(tmp_path):
    """The smoke job's coordinator must run two Qwen runners in parallel, keep
    Kimi units off Qwen workers (and vice versa), and hand the next maze to a
    runner as soon as it finishes (work-stealing)."""
    from scripts.distributed_run_pipeline import CoordinatorStore, prepare_job

    artifacts = tmp_path / "artifacts"
    plan = prepare_job(
        run_config_path=_SMOKE_EVAL_RUN_CONFIG,
        manifest_path=_SMOKE_EVAL_MANIFEST,
        seeds=[0],
        conditions=None,
        artifacts_root=artifacts,
        run_set_id="smoke",
        difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
    )
    # 3 mazes x 2 models = 6 units, split by group.
    units_by_group = Counter(u["model_group"] for u in plan["units"])
    assert units_by_group == {"qwen35-27b": 3, "kimi-api": 3}

    store = CoordinatorStore(artifacts)
    qwen_caps = {"model_groups": ["qwen35-27b"], "hardware_profile": "local-gpu"}
    kimi_caps = {"model_groups": ["kimi-api"]}
    for wid, caps in [("qwenA", qwen_caps), ("qwenB", qwen_caps), ("kimi1", kimi_caps)]:
        store.register({"worker_id": wid, "capabilities": caps})

    def claim(wid, caps):
        return store.assign(wid, caps)["unit"]

    # Both Qwen workers hold distinct Qwen units at once (parallel, in-flight cap 2).
    a1 = claim("qwenA", qwen_caps)
    b1 = claim("qwenB", qwen_caps)
    k1 = claim("kimi1", kimi_caps)
    assert a1 and b1 and k1
    assert a1["unit_id"] != b1["unit_id"]
    assert a1["model_group"] == "qwen35-27b" and b1["model_group"] == "qwen35-27b"
    assert k1["model_group"] == "kimi-api"

    # qwenA finishes first and immediately steals the next (third) Qwen unit while
    # qwenB is still busy on its first.
    store.upload("qwenA", a1["unit_id"], _valid_unit_archive(a1))
    a2 = claim("qwenA", qwen_caps)
    assert a2 is not None
    assert a2["model_group"] == "qwen35-27b"
    assert a2["unit_id"] not in {a1["unit_id"], b1["unit_id"]}

    # No Qwen units remain for qwenB to start a *new* one (3 total: a1 done, b1
    # active, a2 active) -> it just keeps its current unit.
    store.upload("qwenB", b1["unit_id"], _valid_unit_archive(b1))
    store.upload("qwenA", a2["unit_id"], _valid_unit_archive(a2))

    # Kimi worker drains all three Kimi units one-by-one (work-stealing of its own
    # queue); a Qwen worker never receives a Kimi unit.
    kimi_done = {k1["unit_id"]}
    store.upload("kimi1", k1["unit_id"], _valid_unit_archive(k1))
    while True:
        nxt = claim("kimi1", kimi_caps)
        if nxt is None:
            break
        assert nxt["model_group"] == "kimi-api"
        kimi_done.add(nxt["unit_id"])
        store.upload("kimi1", nxt["unit_id"], _valid_unit_archive(nxt))
    assert len(kimi_done) == 3

    # A Qwen worker is never handed a Kimi unit, and with everything verified the
    # coordinator hands out nothing more.
    assert claim("qwenA", qwen_caps) is None
    state = store.load_state()
    assert all(u["status"] == "verified" for u in state["units"].values())


def test_check_run_config_expectations_manifest_mismatch():
    rc = {"models": {}, "manifest": "gridworld/fixtures/manifest.json"}
    with pytest.raises(ValueError, match="manifest"):
        check_run_config_expectations(rc, _OGBENCH_50_MANIFEST, None)


def test_check_run_config_expectations_conditions_mismatch():
    rc = {"models": {}, "conditions": "Observation format"}
    with pytest.raises(ValueError, match="conditions"):
        check_run_config_expectations(rc, _MANIFEST, None)


def test_check_run_config_expectations_passes_when_matching():
    rc = {"models": {}, "manifest": str(_MANIFEST), "conditions": "Prompt"}
    check_run_config_expectations(rc, _MANIFEST, "Prompt")  # no raise


def test_check_run_config_expectations_no_keys_is_noop():
    check_run_config_expectations({"models": {}}, _MANIFEST, None)  # no raise


def test_launch_run_configs_declare_manifest_and_conditions():
    for cond, path in _PENDING_VALIDATION10_CONFIGS.items():
        rc = load_run_config(path)
        assert (_REPO_ROOT / rc["manifest"]).resolve() == _MANIFEST.resolve()
        assert rc["conditions"] == cond
        # The guard must accept the declared pairing without raising.
        check_run_config_expectations(rc, _MANIFEST, cond)


def test_conditional_run_configs_pair_conditional_eval_with_all_six_sets():
    # One run-config per condition set, each pinned to the conditional_eval
    # manifest and its own --conditions, so the H1/H2 guard cannot benchmark the
    # wrong suite/axis across paid models.
    assert set(_CONDITIONAL_CONFIGS) == {
        "Prompt",
        "Observation format",
        "Context window",
        "Action space",
        "Querying strategy",
        "In-context learning",
        "History mechanism",
    }
    catalog = json.loads(_CONDITIONAL_EVAL_MANIFEST.read_text(encoding="utf-8"))["tasks"]
    for cond, path in _CONDITIONAL_CONFIGS.items():
        rc = load_run_config(path)
        assert (_REPO_ROOT / rc["manifest"]).resolve() == _CONDITIONAL_EVAL_MANIFEST.resolve()
        assert rc["conditions"] == cond
        check_run_config_expectations(rc, _CONDITIONAL_EVAL_MANIFEST, cond)  # no raise
        assert set(rc["models"]) == {"qwen36_27b_vllm", "kimi_k26", "claude_opus"}
        assert {m["provider"] for m in rc["models"].values()} == {"qwen_vllm_api", "kimi", "claude"}
        # Opus 4.8 is the Claude model. The experimental sweep runs adaptive
        # thinking at effort:low, and MUST NOT send temperature (Opus 4.7+ reject
        # sampling params with a 400 — see interface/agents/claude.py).
        opus = rc["models"]["claude_opus"]
        assert opus["model"] == "claude-opus-4-8"
        assert "temperature" not in opus
        assert opus["enable_thinking"] is True
        assert opus["effort"] == "low"
        # Kimi runs thinking OFF on the 10 experimental configs (thinking-on
        # truncates before FINAL_OUTPUT at 4096 — see M6 smoke); Moonshot requires
        # temperature 0.6 in thinking-off mode.
        kimi = rc["models"]["kimi_k26"]
        assert kimi["temperature"] == 0.6
        assert kimi["enable_thinking"] is False
        for model_cfg in rc["models"].values():
            rows = resolve_task_rows(model_cfg["tasks"], catalog, _CONDITIONAL_EVAL_MANIFEST)
            assert len(rows) == 15
            # Cross-model comparability: the 10 experimental configs run all
            # three models at the SAME cap (unequal caps confounded
            # baseline_thinking — see FINDINGS §4). Qwen runs thinking OFF on
            # the experimental sweep.
            assert model_cfg["max_tokens"] == 4096
        assert rc["models"]["qwen36_27b_vllm"]["enable_thinking"] is False


def test_baseline_thinking_config_11_runs_full_thinking():
    """Config #11 = the shared baseline (Prompt/standard) re-run with FULL thinking
    on all three models, to isolate reasoning-depth value vs the effort:low sweep.
    Opus runs adaptive+effort:xhigh; paid max_tokens is raised to 8192 (M6
    exception) to leave room for chain-of-thought plus the FINAL_OUTPUT line."""
    path = _FIXTURES / "run_config.conditional_baseline_thinking_claude_kimi_qwen.json"
    rc = load_run_config(path)
    assert (_REPO_ROOT / rc["manifest"]).resolve() == _CONDITIONAL_EVAL_MANIFEST.resolve()
    assert rc["conditions"] == "Prompt"
    check_run_config_expectations(rc, _CONDITIONAL_EVAL_MANIFEST, "Prompt")  # no raise
    assert set(rc["models"]) == {"qwen36_27b_vllm", "kimi_k26", "claude_opus"}

    opus = rc["models"]["claude_opus"]
    assert opus["model"] == "claude-opus-4-8"
    assert "temperature" not in opus
    assert opus["enable_thinking"] is True
    assert opus["effort"] == "xhigh"
    assert opus["max_tokens"] == 8192

    # Config #11 keeps Kimi thinking ON with a large budget: Moonshot requires
    # temperature 1.0 in thinking mode, and 16384 tokens so reasoning + the
    # trailing FINAL_OUTPUT fit (4096/8192 truncated on easy mazes in the smoke).
    kimi = rc["models"]["kimi_k26"]
    assert kimi["temperature"] == 1.0
    assert kimi["enable_thinking"] is True
    assert kimi["max_tokens"] == 16384

    # ⚠️ These per-model caps are UNEQUAL (4k/8k/16k) — a fair within-model
    # budget but an invalid cross-model thinking comparison (FINDINGS §4).
    # The archived config is asserted as-run; a clean re-run needs one equal,
    # large cap for all three models.
    qwen = rc["models"]["qwen36_27b_vllm"]
    assert qwen["provider"] == "qwen_vllm_api"
    assert qwen["model"] == "Qwen/Qwen3.6-27B"
    assert qwen["enable_thinking"] is True
    assert qwen["max_tokens"] == 4096


def test_kimictx_configs_are_kimi_only_thinking_off_over_15_mazes():
    """SWEEP_TOPO=kimictx: a Kimi-only 3-worker fleet running the Context-window
    comparison (last3 / text_summary / text_summary_and_last3) thinking OFF. No
    Claude/Qwen — the fleet provisions 3 kimi-api VMs + coordinator."""
    catalog = json.loads(_CONDITIONAL_EVAL_MANIFEST.read_text(encoding="utf-8"))["tasks"]

    rc = load_run_config(_FIXTURES / "run_config.conditional_context_window_kimi.json")
    assert (_REPO_ROOT / rc["manifest"]).resolve() == _CONDITIONAL_EVAL_MANIFEST.resolve()
    assert rc["conditions"] == "Context window"
    check_run_config_expectations(rc, _CONDITIONAL_EVAL_MANIFEST, "Context window")  # no raise
    assert set(rc["models"]) == {"kimi_k26"}             # Kimi ONLY
    kimi = rc["models"]["kimi_k26"]
    assert kimi["provider"] == "kimi" and kimi["model"] == "kimi-k2.6"
    assert kimi["temperature"] == 0.6 and kimi["enable_thinking"] is False
    assert kimi["worker_count"] == 3                     # 3 Kimi worker VMs
    # max_in_flight is a PER-GROUP concurrency cap (_below_max_in_flight counts
    # active units across the whole model_group). It must equal worker_count or
    # the coordinator serializes the fleet — a cap of 1 starved 2 of 3 workers in
    # the kimictx smoke.
    assert kimi["max_in_flight"] == kimi["worker_count"] == 3
    assert kimi["max_tokens"] == 4096
    rows = resolve_task_rows(kimi["tasks"], catalog, _CONDITIONAL_EVAL_MANIFEST)
    assert len(rows) == 15

    # The provision/smoke config must match the fleet topology (3 kimi-api VMs).
    smoke = load_run_config(_FIXTURES / "run_config.smoke_kimi3.json")
    assert smoke["conditions"] is None
    assert set(smoke["models"]) == {"kimi_k26"}
    assert smoke["models"]["kimi_k26"]["worker_count"] == 3
    assert smoke["models"]["kimi_k26"]["enable_thinking"] is False


def test_smoke_eval_run_config_uses_two_qwen_one_kimi_workers():
    rc = load_run_config(_SMOKE_EVAL_RUN_CONFIG)
    assert (_REPO_ROOT / rc["manifest"]).resolve() == _SMOKE_EVAL_MANIFEST.resolve()
    check_run_config_expectations(rc, _SMOKE_EVAL_MANIFEST, None)  # no raise
    assert set(rc["models"]) == {"qwen35_27b_hf", "kimi_k26"}
    assert rc["models"]["qwen35_27b_hf"]["worker_count"] == 2
    # max_in_flight caps concurrent units per model group, so 2 parallel Qwen
    # runners need a Qwen cap of >= 2 or they serialize.
    assert rc["models"]["qwen35_27b_hf"]["max_in_flight"] == 2
    assert rc["models"]["kimi_k26"]["worker_count"] == 1


def _chain_spec():
    return task_spec_from_payload(load_json(default_maze_path("V06_chain_ks.json")))


def test_infra_only_model_config_keys_excluded_from_run_hash():
    """Infra/transport-only knobs must not bust the episode cache — editing them
    re-pays for every cached episode for no change in model output."""
    spec = _chain_spec()
    base = {"temperature": 0.0, "max_tokens": 128}
    base_hash = _expected_run_hash(spec, "m", 0, "minigrid", model_config=base)
    for infra in (
        {"timeout": 180},
        {"max_attempts": 8},
        {"device_map": {"": 0}},
        {"local_files_only": True},
        {"hardware_profile": "h100-8x"},
        {"worker_count": 4},
        {"max_in_flight": 2},
        {"max_model_len": 8192},
        {"gpu_memory_utilization": 0.88},
        {"tensor_parallel_size": 1},
        {"enable_prefix_caching": True},
        {"enforce_eager": True},
    ):
        cfg = {**base, **infra}
        assert _expected_run_hash(spec, "m", 0, "minigrid", model_config=cfg) == base_hash, infra


def test_generation_model_config_keys_change_run_hash():
    """Knobs that change what the model produces must still invalidate the cache."""
    spec = _chain_spec()
    base = {"temperature": 0.0, "max_tokens": 128}
    base_hash = _expected_run_hash(spec, "m", 0, "minigrid", model_config=base)
    for gen in (
        {"temperature": 0.7},
        {"max_tokens": 256},
        {"enable_thinking": True},
        {"load_in_4bit": True},
        {"torch_dtype": "bfloat16"},
        {"dtype": "float16"},
        {"quantization": "fp8"},
        {"sampling_kwargs": {"top_p": 0.9}},
    ):
        cfg = {**base, **gen}
        assert _expected_run_hash(spec, "m", 0, "minigrid", model_config=cfg) != base_hash, gen


def test_jsonable_rejects_non_primitive_instead_of_stringifying():
    """A type the hash recipe does not understand must raise, not be silently
    str()'d — a stringified object can collide or vary across machines and
    poison the cross-machine episode cache hash (M4)."""
    from scripts.run_pipeline import _jsonable

    class Opaque:
        pass

    with pytest.raises(TypeError):
        _jsonable(Opaque())
    # Known/handled types still pass through unchanged.
    assert _jsonable({"a": (1, 2), "b": {3, 4}}) == {"a": [1, 2], "b": [3, 4]}
    assert _jsonable(2.0) == 2


def test_run_hash_canonicalizes_numeric_spelling():
    """``0`` vs ``0.0`` (or ``128`` vs ``128.0``) is the same model call — it must
    not produce a different cache key."""
    spec = _chain_spec()
    a = _expected_run_hash(spec, "m", 0, "minigrid", model_config={"temperature": 0.0, "max_tokens": 128})
    b = _expected_run_hash(spec, "m", 0, "minigrid", model_config={"temperature": 0, "max_tokens": 128.0})
    assert a == b


# The six condition sets launched over the conditional eval, and the
# deduplicated rollout that runs the shared baseline exactly once.
_LAUNCH_CONDITION_SETS = [
    "Prompt",
    "Observation format",
    "Context window",
    "Action space",
    "Querying strategy",
    "In-context learning",
    "History mechanism",
]
# Each set's variant that EQUALS the fair default ExperimentConfig (the shared
# baseline). After the fair-default rebase, these are the "strong" variants.
_BASELINE_VARIANT = {
    "Prompt": "standard",
    "Observation format": "image_text",
    "Context window": "last3",
    "Action space": "egocentric",
    "Querying strategy": "step_by_step",
    "In-context learning": "one_shot",
    "History mechanism": "single_message",
}
_DEDUP_ROLLOUT = [
    ("Prompt", None),  # shared baseline ("standard") + minimal + verbose
    ("Observation format", "image_only"),
    ("Observation format", "text_only"),
    ("Context window", "current"),
    ("Context window", "text_summary"),
    ("Action space", "cardinal"),
    ("Querying strategy", "subgoal"),
    ("Querying strategy", "full_trajectory"),
    ("In-context learning", "zero_shot"),
    ("History mechanism", "multiturn"),
]
# Variants registered after the validation10 rollout above was launched;
# intentionally not part of that already-run batch.
_NOT_IN_VALIDATION10_ROLLOUT = {
    ("Context window", "text_summary_and_last3"),
}


def _config_key(cfg) -> str:
    return json.dumps(cfg.to_dict(), sort_keys=True)


def test_launch_condition_sets_expose_expected_variants():
    """Lock the variant inventory so a registry edit can't silently drop or
    rename a variable we mean to cover at launch."""
    assert {cs: condition_variant_names(cs) for cs in _LAUNCH_CONDITION_SETS} == {
        "Prompt": ["standard", "minimal", "verbose"],
        "Observation format": ["image_only", "text_only", "image_text"],
        "Context window": ["current", "last3", "text_summary", "text_summary_and_last3"],
        "Action space": ["egocentric", "cardinal"],
        "Querying strategy": ["step_by_step", "subgoal", "full_trajectory"],
        "In-context learning": ["zero_shot", "one_shot"],
        "History mechanism": ["single_message", "multiturn"],
    }


def test_baseline_variant_of_every_launch_set_is_the_default_config():
    """Every set's baseline variant builds the identical default config, which is
    what makes running it once (under "standard") valid for all axes."""
    from interface.config import ExperimentConfig

    base_key = _config_key(ExperimentConfig())
    for conditions, variant in _BASELINE_VARIANT.items():
        configs = _condition_configs(conditions, prompt_variant=variant)
        assert len(configs) == 1
        assert _config_key(configs[0][1]) == base_key


def test_dedup_rollout_covers_every_unique_variant_config_once():
    """The deduplicated rollout must cover every distinct prompt config across
    the four launch sets, and run the shared baseline exactly once."""
    all_unique = {
        _config_key(cfg)
        for cs in _LAUNCH_CONDITION_SETS
        for _name, cfg in _condition_configs(cs)
        if (cs, _name) not in _NOT_IN_VALIDATION10_ROLLOUT
    }
    rolled = [
        _config_key(cfg)
        for conditions, variant in _DEDUP_ROLLOUT
        for _name, cfg in _condition_configs(conditions, prompt_variant=variant)
    ]
    assert set(rolled) == all_unique  # nothing dropped
    assert len(rolled) == len(all_unique)  # baseline run once, no variant twice


def test_experiment_config_overlay_applies_under_condition_variant():
    """A top-level experiment_config overlay composes with a named condition
    variant: defaults < overlay < condition variant. The variant stays
    authoritative for the axis it declares (context_window here); the overlay
    field it doesn't touch (progress_stall_k) survives."""
    pairs = _condition_configs(
        "Context window", prompt_variant="text_summary",
        base_overrides={"progress_stall_k": 20},
    )
    assert len(pairs) == 1
    _, cfg = pairs[0]
    assert cfg.progress_stall_k == 20            # overlay preserved
    assert cfg.context_window == "text_summary"  # variant still authoritative


def test_run_from_config_experiment_config_overlay_reaches_run_inputs(tmp_path):
    """The top-level experiment_config overlay in a run-config JSON must flow
    all the way into the per-run sidecar, proving run_from_config actually
    threads base_overrides down through _run_one_model/_run_one_unit."""
    run_config = {
        "models": {
            "stub": {
                "provider": "claude",
                "model": "stub-model",
                "tasks": [str(default_maze_path("V01_empty_room.json"))],
            }
        },
        "experiment_config": {"progress_stall_k": 20},
    }
    cfg_path = tmp_path / "run_config.json"
    cfg_path.write_text(json.dumps(run_config), encoding="utf-8")
    artifacts = tmp_path / "artifacts"

    def factory(name, model_cfg):
        return ReplayAgent(v01_empty_room_trajectory()), model_cfg["model"]

    run_from_config(
        run_config_path=cfg_path,
        manifest_path=_MANIFEST,
        seeds=[0],
        artifacts_root=artifacts,
        run_set_id="cfg",
        agent_factory=factory,
        difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
    )

    run_dir = (
        artifacts / "runs" / "validation_10_v01_empty_room" / "minigrid" / "stub-model" / "seed_0" / "default"
    )
    sidecar = load_json(run_dir / "run_inputs.json")
    assert sidecar["experiment_config"]["progress_stall_k"] == 20


def test_progress_stall_k_reaches_stored_resolved_config(tmp_path):
    """Task 11 end-to-end proof: a run-config's top-level experiment_config
    overlay (progress_stall_k=20) must reach the SAME resolved config through
    both storage paths -- the local pipeline's run_inputs.json sidecar
    (run_from_config -> _run_one_model -> _run_one_unit) and a prepared
    distributed unit (prepare_job -> _condition_configs with the same
    overlay). Individually these paths are covered by
    test_run_from_config_experiment_config_overlay_reaches_run_inputs (Task 3)
    and test_distributed_units_carry_resolved_experiment_config (Task 4); this
    test is the integration proof that both channels reconstruct the
    identical resolved config from one run-config, not merely that each
    independently contains 20."""
    from scripts.distributed_run_pipeline import prepare_job

    run_config = {
        "models": {
            "stub": {
                "provider": "claude",
                "model": "stub-model",
                "tasks": [str(default_maze_path("V01_empty_room.json"))],
            }
        },
        "experiment_config": {"progress_stall_k": 20},
    }
    cfg_path = tmp_path / "run_config.json"
    cfg_path.write_text(json.dumps(run_config), encoding="utf-8")
    artifacts = tmp_path / "artifacts"

    def factory(name, model_cfg):
        return ReplayAgent(v01_empty_room_trajectory()), model_cfg["model"]

    run_from_config(
        run_config_path=cfg_path,
        manifest_path=_MANIFEST,
        seeds=[0],
        artifacts_root=artifacts,
        run_set_id="cfg",
        agent_factory=factory,
        difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
    )

    run_dir = (
        artifacts / "runs" / "validation_10_v01_empty_room" / "minigrid" / "stub-model" / "seed_0" / "default"
    )
    sidecar = load_json(run_dir / "run_inputs.json")
    stored_config = sidecar["experiment_config"]
    assert stored_config["progress_stall_k"] == 20

    plan = prepare_job(
        run_config_path=cfg_path,
        manifest_path=_MANIFEST,
        seeds=[0],
        conditions=None,
        artifacts_root=artifacts,
        run_set_id="dist",
        difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
    )
    assert plan["units"], "expected at least one distributed unit"
    reconstructed_config = plan["units"][0]["experiment_config"]
    assert reconstructed_config["progress_stall_k"] == 20

    # The local sidecar's stored config and the distributed unit's
    # reconstructed config must be the *same* resolved config end to end --
    # this is what would break (while each half-test above kept passing) if
    # Task 3's overlay and Task 4's unit-resolution ever drifted apart.
    assert stored_config == reconstructed_config


def test_run_from_config_rejects_invalid_experiment_config_before_model_call(tmp_path):
    """An unknown/invalid experiment_config key must fail fast during job
    preparation, before any (paid) model call happens."""

    def _boom(name, model_cfg):
        raise AssertionError("model call must not happen when the overlay is invalid")

    run_config = {
        "models": {
            "stub": {
                "provider": "claude",
                "model": "stub-model",
                "tasks": [str(default_maze_path("V01_empty_room.json"))],
            }
        },
        "experiment_config": {"not_a_real_field": True},
    }
    cfg_path = tmp_path / "run_config.json"
    cfg_path.write_text(json.dumps(run_config), encoding="utf-8")

    with pytest.raises(TypeError):
        run_from_config(
            run_config_path=cfg_path,
            manifest_path=_MANIFEST,
            seeds=[0],
            artifacts_root=tmp_path / "artifacts",
            run_set_id="cfg",
            agent_factory=_boom,
            difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
        )


def test_distributed_prepare_honors_prompt_variant(tmp_path):
    """coordinator-prepare can plan a single variant so the shared baseline is
    not re-run once per condition set (the deduplicated launch rollout)."""
    from scripts.distributed_run_pipeline import prepare_job

    task = str(default_maze_path("V01_empty_room.json"))
    cfg_path = _write_run_config(
        tmp_path,
        {"stub": {"provider": "claude", "model": "stub-model", "group": "stub", "tasks": [task]}},
    )

    def _prepare(prompt_variant, sub):
        return prepare_job(
            run_config_path=cfg_path,
            manifest_path=_MANIFEST,
            seeds=[0],
            conditions="Observation format",
            prompt_variant=prompt_variant,
            artifacts_root=tmp_path / sub,
            run_set_id="dist",
            difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
        )

    full = _prepare(None, "full")
    assert {u["prompt_variant"] for u in full["units"]} == {"image_only", "text_only", "image_text"}

    one = _prepare("text_only", "one")
    assert one["prompt_variants"] == ["text_only"]
    assert {u["prompt_variant"] for u in one["units"]} == {"text_only"}

    with pytest.raises(ValueError, match="Unknown prompt variant"):
        _prepare("bogus", "bogus")


def test_distributed_assignment_filters_group_and_reassigns_stale(tmp_path):
    from scripts.distributed_run_pipeline import CoordinatorStore, prepare_job, state_path

    task = str(default_maze_path("V01_empty_room.json"))
    cfg_path = _write_run_config(
        tmp_path,
        {
            "a": {"provider": "qwen", "model": "model-a", "group": "group-a", "tasks": [task]},
            "b": {"provider": "qwen", "model": "model-b", "group": "group-b", "tasks": [task]},
        },
    )
    artifacts = tmp_path / "artifacts"
    prepare_job(
        run_config_path=cfg_path,
        manifest_path=_MANIFEST,
        seeds=[0],
        conditions=None,
        artifacts_root=artifacts,
        run_set_id="dist",
        difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
    )
    store = CoordinatorStore(artifacts, stale_after_seconds=1.0)

    w1 = store.register({"worker_id": "w1", "capabilities": {"model_group": "group-a"}})["worker_id"]
    first = store.assign(w1)["unit"]
    assert first["model_group"] == "group-a"
    assert store.assign(w1)["unit"]["unit_id"] == first["unit_id"]

    state = json.loads(state_path(artifacts).read_text(encoding="utf-8"))
    state["units"][first["unit_id"]]["heartbeat_at"] = 0
    state_path(artifacts).write_text(json.dumps(state), encoding="utf-8")

    w2 = store.register({"worker_id": "w2", "capabilities": {"model_group": "group-a"}})["worker_id"]
    reassigned = store.assign(w2)["unit"]
    assert reassigned["unit_id"] == first["unit_id"]

    w3 = store.register({"worker_id": "w3", "capabilities": {"model_group": "group-b"}})["worker_id"]
    group_b = store.assign(w3)["unit"]
    assert group_b["model_group"] == "group-b"


def test_failed_unit_is_reassigned_until_attempt_cap(tmp_path):
    """A transient worker crash must not permanently strand a paid unit, but a
    poison unit must stop being re-handed-out once the attempt cap is hit."""
    from scripts.distributed_run_pipeline import CoordinatorStore, prepare_job

    task = str(default_maze_path("V01_empty_room.json"))
    cfg_path = _write_run_config(
        tmp_path, {"a": {"provider": "qwen", "model": "m", "group": "g", "tasks": [task]}},
    )
    artifacts = tmp_path / "artifacts"
    prepare_job(
        run_config_path=cfg_path, manifest_path=_MANIFEST, seeds=[0], conditions=None,
        artifacts_root=artifacts, run_set_id="dist", difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
    )
    store = CoordinatorStore(artifacts, max_unit_attempts=2)

    w1 = store.register({"worker_id": "w1", "capabilities": {"model_group": "g"}})["worker_id"]
    u1 = store.assign(w1)["unit"]
    assert u1 is not None
    store.fail(w1, u1["unit_id"], "crash")

    w2 = store.register({"worker_id": "w2", "capabilities": {"model_group": "g"}})["worker_id"]
    u2 = store.assign(w2)["unit"]
    assert u2 is not None and u2["unit_id"] == u1["unit_id"]  # reassigned after failure
    store.fail(w2, u2["unit_id"], "crash again")

    w3 = store.register({"worker_id": "w3", "capabilities": {"model_group": "g"}})["worker_id"]
    assert store.assign(w3)["unit"] is None  # attempt cap reached -> terminal


def test_distributed_upload_validates_and_extracts_archive(tmp_path):
    from scripts.distributed_run_pipeline import CoordinatorStore, prepare_job

    task = str(default_maze_path("V01_empty_room.json"))
    cfg_path = _write_run_config(
        tmp_path,
        {"stub": {"provider": "claude", "model": "stub-model", "group": "stub", "tasks": [task]}},
    )
    artifacts = tmp_path / "artifacts"
    prepare_job(
        run_config_path=cfg_path,
        manifest_path=_MANIFEST,
        seeds=[0],
        conditions=None,
        artifacts_root=artifacts,
        run_set_id="dist",
        difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
    )
    store = CoordinatorStore(artifacts)
    worker_id = store.register({"worker_id": "w", "capabilities": {"model_group": "stub"}})["worker_id"]
    unit = store.assign(worker_id)["unit"]

    result = store.upload(worker_id, unit["unit_id"], _valid_unit_archive(unit))

    assert result["status"] == "verified"
    run_dir = artifacts / unit["run_dir"]
    assert (run_dir / "episode.json").exists()
    assert (run_dir / "run_inputs.json").exists()
    assert (run_dir / "run_score.json").exists()


def _stub_unit_store(tmp_path, *, storage=None):
    from scripts.distributed_run_pipeline import CoordinatorStore, prepare_job

    task = str(default_maze_path("V01_empty_room.json"))
    cfg_path = _write_run_config(
        tmp_path,
        {"stub": {"provider": "claude", "model": "stub-model", "group": "stub", "tasks": [task]}},
    )
    artifacts = tmp_path / "artifacts"
    prepare_job(
        run_config_path=cfg_path,
        manifest_path=_MANIFEST,
        seeds=[0],
        conditions=None,
        artifacts_root=artifacts,
        run_set_id="dist",
        difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
    )
    store = CoordinatorStore(artifacts, storage=storage)
    wid = store.register({"worker_id": "w", "capabilities": {"model_group": "stub"}})["worker_id"]
    unit = store.assign(wid)["unit"]
    return artifacts, store, wid, unit


def test_distributed_upload_mirrors_verified_run_to_bucket(tmp_path, monkeypatch):
    from scripts import distributed_run_pipeline as dist

    artifacts, store, wid, unit = _stub_unit_store(
        tmp_path, storage=dist.StorageConfig(bucket="gs://bkt/runs")
    )
    calls = []
    monkeypatch.setattr(dist, "mirror_to_bucket", lambda src, dest, **kw: calls.append((str(src), dest)))

    result = store.upload(wid, unit["unit_id"], _valid_unit_archive(unit))

    assert result["status"] == "verified"
    assert len(calls) == 1
    src, dest = calls[0]
    assert src == str(artifacts / unit["run_dir"])
    assert dest == "gs://bkt/runs/dist/" + unit["run_dir"]
    assert store.load_state()["units"][unit["unit_id"]]["gcs_uri"] == dest


def test_distributed_upload_records_pending_when_mirror_fails(tmp_path, monkeypatch):
    from scripts import distributed_run_pipeline as dist

    _artifacts, store, wid, unit = _stub_unit_store(
        tmp_path, storage=dist.StorageConfig(bucket="gs://bkt/runs")
    )

    def boom(src, dest, **kw):
        raise RuntimeError("network down")

    monkeypatch.setattr(dist, "mirror_to_bucket", boom)

    # A failed mirror must not lose the (paid) verified run: the upload still
    # succeeds and the result is flagged for a later re-push.
    result = store.upload(wid, unit["unit_id"], _valid_unit_archive(unit))
    assert result["status"] == "verified"
    us = store.load_state()["units"][unit["unit_id"]]
    assert us["status"] == "verified"
    assert us["gcs_pending"] is True
    assert "network down" in us["gcs_error"]


def test_distributed_upload_without_storage_does_not_mirror(tmp_path, monkeypatch):
    from scripts import distributed_run_pipeline as dist

    _artifacts, store, wid, unit = _stub_unit_store(tmp_path)  # storage=None
    monkeypatch.setattr(
        dist, "mirror_to_bucket",
        lambda *a, **k: pytest.fail("mirror_to_bucket must not run without storage"),
    )

    result = store.upload(wid, unit["unit_id"], _valid_unit_archive(unit))
    assert result["status"] == "verified"
    us = store.load_state()["units"][unit["unit_id"]]
    assert "gcs_uri" not in us and "gcs_pending" not in us


def test_distributed_finalize_mirrors_aggregate_to_bucket(tmp_path, monkeypatch):
    from scripts import distributed_run_pipeline as dist

    manifest_path = _write_manifest(tmp_path)
    cfg_path = _write_run_config(
        tmp_path,
        {"stub": {"provider": "claude", "model": "replay-stub", "group": "stub",
                  "tasks": [str(default_maze_path("V01_empty_room.json"))]}},
    )
    art = tmp_path / "coordinator"
    storage = dist.StorageConfig(bucket="gs://bkt/runs")
    dist.prepare_job(
        run_config_path=cfg_path, manifest_path=manifest_path, seeds=[0], conditions=None,
        artifacts_root=art, run_set_id="dist", difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
    )
    store = dist.CoordinatorStore(art, storage=storage)
    wid = store.register({"worker_id": "w", "capabilities": {"model_group": "stub"}})["worker_id"]
    unit = store.assign(wid)["unit"]
    dist.run_assigned_unit(
        unit, artifacts_root=tmp_path / "worker",
        agent_factory=lambda n, mc: (ReplayAgent(v01_empty_room_trajectory()), mc["model"]),
    )

    dests = []
    monkeypatch.setattr(dist, "mirror_to_bucket", lambda src, dest, **kw: dests.append(dest))
    store.upload(
        wid, unit["unit_id"],
        dist.package_run_archive(unit, artifacts_root=tmp_path / "worker").read_bytes(),
    )
    dist.finalize_job(artifacts_root=art, storage=storage)

    assert any(d.endswith("/dist/episode_runs.jsonl") for d in dests)
    assert any(d.endswith("/dist/reports/dist") for d in dests)


def test_safe_heartbeat_swallows_transient_errors():
    from scripts.distributed_run_pipeline import _safe_heartbeat

    class _BadClient:
        def heartbeat(self, *a):
            raise RuntimeError("coordinator down")

    # Must not raise: a heartbeat blip can't be allowed to kill the heartbeat thread.
    assert _safe_heartbeat(_BadClient(), "w", "u") is False

    class _GoodClient:
        def __init__(self):
            self.calls = 0

        def heartbeat(self, *a):
            self.calls += 1

    good = _GoodClient()
    assert _safe_heartbeat(good, "w", "u") is True
    assert good.calls == 1


def test_distributed_upload_rejects_truncated_json(tmp_path):
    artifacts, store, wid, unit = _stub_unit_store(tmp_path)
    bad = _dummy_run_archive(
        {"episode.json": "{ truncated", "run_inputs.json": "{}", "run_score.json": "{}"}
    )
    with pytest.raises(ValueError, match="not valid JSON"):
        store.upload(wid, unit["unit_id"], bad)
    # A failed verification must not leave a half-extracted run dir.
    assert not (artifacts / unit["run_dir"] / "episode.json").exists()


def test_distributed_upload_rejects_wrong_episode_inputs_hash(tmp_path):
    artifacts, store, wid, unit = _stub_unit_store(tmp_path)
    bad = _dummy_run_archive({
        "episode.json": "{}",
        "run_inputs.json": json.dumps({"inputs_hash": "wrong-inputs"}),
        "run_score.json": "{}",
    })

    with pytest.raises(ValueError, match="inputs_hash mismatch"):
        store.upload(wid, unit["unit_id"], bad)

    assert not (artifacts / unit["run_dir"] / "episode.json").exists()


def test_episode_log_write_leaves_no_tmp_and_valid_json(tmp_path):
    manifest_path = _write_manifest(tmp_path)
    artifacts = tmp_path / "artifacts"
    run_pipeline(
        manifest_path=manifest_path, experiment="test1",
        agent=CountingReplayAgent(v01_empty_room_trajectory()), agent_name="stub",
        seeds=[0], artifacts_root=artifacts, run_set_id="r",
        difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
    )
    run_dir = artifacts / "runs" / "validation_10_v01_empty_room" / "minigrid" / "stub" / "seed_0" / "default"
    assert load_json(run_dir / "episode.json")  # valid JSON, fully written
    assert not list(run_dir.glob("*.tmp"))  # atomic write cleaned up its temp


def test_expected_run_hash_is_stable_across_processes(tmp_path):
    """Coordinator and workers run on different machines/interpreters; the run
    hash must be byte-identical or every worker re-pays for cached episodes."""
    import subprocess
    import sys
    import textwrap

    source = str(default_maze_path("V06_chain_ks.json"))
    snippet = textwrap.dedent(
        f"""
        from scorer.io import load_json, task_spec_from_payload
        from scripts.run_pipeline import _expected_run_hash
        spec = task_spec_from_payload(load_json(r"{source}"))
        print(_expected_run_hash(
            spec, "model-x", 0, "minigrid",
            condition_set=None, prompt_variant="default",
            experiment_config={{"a": 1, "b": [2, 3]}},
            model_config={{"temperature": 0.0, "max_tokens": 128, "z": 1, "a": 2}},
        ))
        """
    )
    outs = [
        subprocess.run(
            [sys.executable, "-c", snippet], cwd=str(_REPO_ROOT),
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        for _ in range(2)
    ]
    assert outs[0] == outs[1] and len(outs[0]) == 64


def test_max_in_flight_blocks_second_concurrent_assignment(tmp_path):
    from scripts.distributed_run_pipeline import CoordinatorStore, prepare_job

    t1 = str(default_maze_path("V01_empty_room.json"))
    t2 = str(default_maze_path("V06_chain_ks.json"))
    cfg_path = _write_run_config(
        tmp_path,
        {"g": {"provider": "qwen", "model": "m", "group": "g", "max_in_flight": 1, "tasks": [t1, t2]}},
    )
    artifacts = tmp_path / "artifacts"
    prepare_job(
        run_config_path=cfg_path, manifest_path=_MANIFEST, seeds=[0], conditions=None,
        artifacts_root=artifacts, run_set_id="dist", difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
    )
    store = CoordinatorStore(artifacts)
    w1 = store.register({"worker_id": "w1", "capabilities": {"model_group": "g"}})["worker_id"]
    w2 = store.register({"worker_id": "w2", "capabilities": {"model_group": "g"}})["worker_id"]
    assert store.assign(w1)["unit"] is not None       # 1 in flight
    assert store.assign(w2)["unit"] is None            # capped -> no second concurrent unit


def test_distributed_upload_rejects_path_traversal_member(tmp_path):
    artifacts, store, wid, unit = _stub_unit_store(tmp_path)
    evil = _dummy_run_archive(
        {"episode.json": "{}", "run_inputs.json": "{}", "run_score.json": "{}", "../escape.json": "{}"}
    )
    with pytest.raises(ValueError, match="Unsafe archive path"):
        store.upload(wid, unit["unit_id"], evil)
    assert not (artifacts.parent / "escape.json").exists()


@pytest.mark.parametrize("manifest", sorted(_FIXTURES.glob("manifest*.json")))
def test_all_committed_manifest_sources_resolve(manifest):
    """A source-path typo in a committed manifest must fail here, not after
    spinning workers / spending tokens."""
    for row in load_manifest(manifest):
        _resolve_source(row, manifest)  # raises FileNotFoundError on a bad path


def test_distributed_finalize_is_idempotent(tmp_path):
    from scripts.distributed_run_pipeline import (
        CoordinatorStore, finalize_job, package_run_archive, prepare_job, run_assigned_unit,
    )

    manifest_path = _write_manifest(tmp_path)
    cfg_path = _write_run_config(
        tmp_path,
        {"stub": {"provider": "claude", "model": "replay-stub", "group": "stub",
                  "tasks": [str(default_maze_path("V01_empty_room.json"))]}},
    )
    art = tmp_path / "coordinator"
    prepare_job(
        run_config_path=cfg_path, manifest_path=manifest_path, seeds=[0], conditions=None,
        artifacts_root=art, run_set_id="dist", difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
    )
    store = CoordinatorStore(art)
    wid = store.register({"worker_id": "w", "capabilities": {"model_group": "stub"}})["worker_id"]
    unit = store.assign(wid)["unit"]
    run_assigned_unit(
        unit, artifacts_root=tmp_path / "worker",
        agent_factory=lambda n, mc: (ReplayAgent(v01_empty_room_trajectory()), mc["model"]),
    )
    store.upload(wid, unit["unit_id"],
                 package_run_archive(unit, artifacts_root=tmp_path / "worker").read_bytes())

    first = finalize_job(artifacts_root=art)
    second = finalize_job(artifacts_root=art)  # a retry after a crash must not double-count
    rows = (art / "episode_runs.jsonl").read_text().strip().splitlines()
    assert first["run_count"] == second["run_count"] == 1
    assert len(rows) == 1


def test_run_assigned_unit_rejects_stale_plan_missing_experiment_config(tmp_path):
    """A worker resuming a job_plan.json prepared before units carried a
    resolved experiment_config must get a clear, actionable RuntimeError
    telling it to regenerate the plan via coordinator-prepare — not a bare
    KeyError from the from_dict call."""
    from scripts.distributed_run_pipeline import CoordinatorStore, prepare_job, run_assigned_unit

    manifest_path = _write_manifest(tmp_path)
    cfg_path = _write_run_config(
        tmp_path,
        {"stub": {"provider": "claude", "model": "replay-stub", "group": "stub",
                  "tasks": [str(default_maze_path("V01_empty_room.json"))]}},
    )
    art = tmp_path / "coordinator"
    prepare_job(
        run_config_path=cfg_path, manifest_path=manifest_path, seeds=[0], conditions=None,
        artifacts_root=art, run_set_id="dist", difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
    )
    store = CoordinatorStore(art)
    wid = store.register({"worker_id": "w", "capabilities": {"model_group": "stub"}})["worker_id"]
    unit = store.assign(wid)["unit"]
    assert "experiment_config" in unit  # sanity: freshly prepared plans do carry it

    # Simulate a job_plan.json persisted before the resolved-config change.
    del unit["experiment_config"]

    with pytest.raises(RuntimeError) as excinfo:
        run_assigned_unit(
            unit, artifacts_root=tmp_path / "worker",
            agent_factory=lambda n, mc: (ReplayAgent(v01_empty_room_trajectory()), mc["model"]),
        )
    message = str(excinfo.value)
    assert "coordinator-prepare" in message
    assert "regenerate" in message


def test_distributed_worker_upload_finalize_local_integration(tmp_path):
    from scripts.distributed_run_pipeline import (
        CoordinatorStore,
        finalize_job,
        package_run_archive,
        prepare_job,
        run_assigned_unit,
    )

    manifest_path = _write_manifest(tmp_path)
    cfg_path = _write_run_config(
        tmp_path,
        {
            "stub": {
                "provider": "claude",
                "model": "replay-stub",
                "group": "stub",
                "tasks": [str(default_maze_path("V01_empty_room.json"))],
            }
        },
    )
    coordinator_artifacts = tmp_path / "coordinator"
    prepare_job(
        run_config_path=cfg_path,
        manifest_path=manifest_path,
        seeds=[0],
        conditions=None,
        artifacts_root=coordinator_artifacts,
        run_set_id="dist",
        difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
    )
    store = CoordinatorStore(coordinator_artifacts)
    worker_id = store.register({"worker_id": "w", "capabilities": {"model_group": "stub"}})["worker_id"]
    unit = store.assign(worker_id)["unit"]

    def factory(name, model_cfg):
        return ReplayAgent(v01_empty_room_trajectory()), model_cfg["model"]

    worker_artifacts = tmp_path / "worker"
    run_assigned_unit(unit, artifacts_root=worker_artifacts, agent_factory=factory)
    worker_episode = load_json(worker_artifacts / unit["run_dir"] / "episode.json")
    worker_sidecar = load_json(worker_artifacts / unit["run_dir"] / "run_inputs.json")
    assert worker_episode["task_spec"]["max_steps"] == 33
    assert worker_sidecar["runtime_max_steps_cap"]["effective_max_steps"] == 33
    archive = package_run_archive(unit, artifacts_root=worker_artifacts)
    store.upload(worker_id, unit["unit_id"], archive.read_bytes())

    result = finalize_job(artifacts_root=coordinator_artifacts)

    assert result["run_count"] == 1
    assert result["missing_units"] == []
    rows = [
        json.loads(line)
        for line in (coordinator_artifacts / "episode_runs.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["agent_or_model"] == "replay-stub"
    assert rows[0]["prompt_variant"] == "default"
    assert (
        coordinator_artifacts / "reports" / "dist" / "models" / "replay-stub.json"
    ).exists()


def test_distributed_coordinator_runs_api_client_locally(tmp_path):
    from scripts.distributed_run_pipeline import (
        CoordinatorStore,
        finalize_job,
        prepare_job,
        run_coordinator_api_client,
    )

    manifest_path = _write_manifest(tmp_path)
    cfg_path = _write_run_config(
        tmp_path,
        {
            "claude-api": {
                "provider": "claude",
                "model": "replay-api",
                "group": "api-clients",
                "tasks": [str(default_maze_path("V01_empty_room.json"))],
            }
        },
    )
    artifacts = tmp_path / "coordinator"
    prepare_job(
        run_config_path=cfg_path,
        manifest_path=manifest_path,
        seeds=[0],
        conditions=None,
        artifacts_root=artifacts,
        run_set_id="api",
        difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
    )

    def factory(name, model_cfg):
        return ReplayAgent(v01_empty_room_trajectory()), model_cfg["model"]

    result = run_coordinator_api_client(
        artifacts_root=artifacts,
        model_group="api-clients",
        agent_factory=factory,
    )
    finalized = finalize_job(artifacts_root=artifacts)
    status = CoordinatorStore(artifacts).status()

    assert result["completed"] == 1
    assert status["units"] == {"verified": 1}
    assert finalized["run_count"] == 1
    rows = [
        json.loads(line)
        for line in (artifacts / "episode_runs.jsonl").read_text().splitlines()
    ]
    assert rows[0]["agent_or_model"] == "replay-api"


def test_distributed_prepare_preserves_progress_on_rerun(tmp_path):
    """Re-running prepare with identical inputs must not wipe completed (paid) work."""
    from scripts.distributed_run_pipeline import prepare_job, state_path

    task = str(default_maze_path("V01_empty_room.json"))
    cfg_path = _write_run_config(
        tmp_path, {"a": {"provider": "qwen", "model": "m", "group": "g", "tasks": [task]}},
    )
    artifacts = tmp_path / "artifacts"
    common = dict(
        run_config_path=cfg_path, manifest_path=_MANIFEST, seeds=[0], conditions=None,
        artifacts_root=artifacts, run_set_id="dist", difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
    )
    prepare_job(**common)

    state = json.loads(state_path(artifacts).read_text(encoding="utf-8"))
    unit_id = next(iter(state["units"]))
    state["units"][unit_id]["status"] = "verified"
    state_path(artifacts).write_text(json.dumps(state), encoding="utf-8")

    prepare_job(**common)  # identical inputs -> same job_id -> state must be preserved

    reread = json.loads(state_path(artifacts).read_text(encoding="utf-8"))
    assert reread["units"][unit_id]["status"] == "verified"


def test_explicit_job_id_does_not_preserve_units_after_resolved_config_changes(tmp_path):
    """A durable job name is a namespace, not permission to reuse stale work.

    The fully resolved experiment config participates in the canonical episode
    hash and therefore in unit identity. Re-preparing the same explicit job ID
    with a different watchdog K must replace the completed unit with fresh
    pending work.
    """
    from scripts.distributed_run_pipeline import prepare_job, state_path

    task = str(default_maze_path("V01_empty_room.json"))
    cfg_path = tmp_path / "run_config.json"
    artifacts = tmp_path / "artifacts"

    def write_config(k):
        cfg_path.write_text(
            json.dumps(
                {
                    "experiment_config": {"progress_stall_k": k},
                    "models": {
                        "a": {
                            "provider": "qwen",
                            "model": "m",
                            "group": "g",
                            "tasks": [task],
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

    common = dict(
        run_config_path=cfg_path,
        manifest_path=_MANIFEST,
        seeds=[0],
        conditions=None,
        artifacts_root=artifacts,
        run_set_id="dist",
        difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
        job_id="durable-watchdog-job",
    )

    write_config(20)
    first_plan = prepare_job(**common)
    first_id = first_plan["units"][0]["unit_id"]
    state = json.loads(state_path(artifacts).read_text(encoding="utf-8"))
    state["units"][first_id]["status"] = "verified"
    state_path(artifacts).write_text(json.dumps(state), encoding="utf-8")

    write_config(30)
    second_plan = prepare_job(**common)
    second_id = second_plan["units"][0]["unit_id"]
    reread = json.loads(state_path(artifacts).read_text(encoding="utf-8"))

    assert second_id != first_id
    assert first_id not in reread["units"]
    assert reread["units"][second_id]["status"] == "pending"


def test_distributed_unit_inputs_hash_covers_complete_episode_and_scoring_recipe():
    from scripts.distributed_run_pipeline import _unit_inputs_hash

    scorer_config = load_scorer_config()
    common = {
        "episode_inputs_hash": "episode-a",
        "task_row": {"task_id": "t", "condition": "a"},
        "canonical_paths": {"inputs_hash": "canonical-input", "route": [1, 2]},
        "scored_static": {"inputs_hash": "static-input", "static_score": 10.0},
        "scorer_config": scorer_config,
        "difficulty_max_static_score": 100.0,
    }
    baseline = _unit_inputs_hash(**common)

    changes = [
        {**common, "episode_inputs_hash": "episode-b"},
        {**common, "task_row": {"task_id": "t", "condition": "b"}},
        {
            **common,
            # The complete artifact is hashed, not only its embedded input hash.
            "canonical_paths": {
                "inputs_hash": "canonical-input",
                "route": [1, 3],
            },
        },
        {
            **common,
            "scored_static": {
                "inputs_hash": "static-input",
                "static_score": 11.0,
            },
        },
        {**common, "difficulty_max_static_score": 101.0},
    ]
    changed_scorer = load_scorer_config()
    changed_scorer.baseline_tokens += 1
    changes.append({**common, "scorer_config": changed_scorer})

    assert all(_unit_inputs_hash(**changed) != baseline for changed in changes)


def test_distributed_api_client_continues_past_failures_and_reports_them(tmp_path):
    """One failing API unit must not abort the rest; failures are recorded, not raised."""
    from scripts.distributed_run_pipeline import CoordinatorStore, prepare_job, run_coordinator_api_client

    task = str(default_maze_path("V01_empty_room.json"))
    cfg_path = _write_run_config(
        tmp_path,
        {
            "good": {"provider": "claude", "model": "replay-good", "group": "good-api", "tasks": [task]},
            "bad": {"provider": "claude", "model": "boom", "group": "bad-api", "tasks": [task]},
        },
    )
    artifacts = tmp_path / "coordinator"
    prepare_job(
        run_config_path=cfg_path, manifest_path=_write_manifest(tmp_path), seeds=[0], conditions=None,
        artifacts_root=artifacts, run_set_id="api", difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
    )

    class _BoomAgent:
        last_usage = None

        def __call__(self, messages):
            raise RuntimeError("boom")

    def factory(name, model_cfg):
        if model_cfg["model"] == "boom":
            return _BoomAgent(), model_cfg["model"]
        return ReplayAgent(v01_empty_room_trajectory()), model_cfg["model"]

    # max_units=0 -> drain all pending; the bad unit must not abort the good one.
    result = run_coordinator_api_client(artifacts_root=artifacts, max_units=0, agent_factory=factory)

    assert result["completed"] == 1  # the good unit still ran
    assert result["failures"]  # the bad unit was recorded, not raised
    assert all(f["unit_id"] for f in result["failures"])
    status = CoordinatorStore(artifacts).status()
    assert status["units"].get("verified") == 1


def test_distributed_finalize_requires_complete_work_by_default(tmp_path):
    from scripts.distributed_run_pipeline import finalize_job, prepare_job

    task = str(default_maze_path("V01_empty_room.json"))
    cfg_path = _write_run_config(
        tmp_path,
        {"stub": {"provider": "claude", "model": "stub-model", "group": "stub", "tasks": [task]}},
    )
    artifacts = tmp_path / "artifacts"
    prepare_job(
        run_config_path=cfg_path,
        manifest_path=_MANIFEST,
        seeds=[0],
        conditions=None,
        artifacts_root=artifacts,
        run_set_id="dist",
        difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
    )

    with pytest.raises(RuntimeError, match="Missing distributed work units"):
        finalize_job(artifacts_root=artifacts)

    partial = finalize_job(artifacts_root=artifacts, allow_partial=True)
    assert partial["run_count"] == 0
    assert len(partial["missing_units"]) == 1


def test_cross_model_run_configs_must_declare_equal_token_caps(tmp_path):
    """Unequal per-model max_tokens confounded baseline_thinking (FINDINGS §4):
    the 4k/8k/16k spread measured budget, not reasoning. Cross-model configs
    must use one cap — or opt out explicitly so the asymmetry is on record."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    base = {
        "manifest": str(manifest),
        "models": {
            "a": {"provider": "claude", "max_tokens": 8192},
            "b": {"provider": "kimi", "max_tokens": 16384},
        },
    }

    with pytest.raises(ValueError, match="unequal max_tokens"):
        check_run_config_expectations(base, manifest, None)

    # Equal caps pass.
    equal = json.loads(json.dumps(base))
    equal["models"]["b"]["max_tokens"] = 8192
    check_run_config_expectations(equal, manifest, None)  # no raise

    # Single-model configs are unconstrained.
    solo = {"manifest": str(manifest), "models": {"a": {"max_tokens": 4096}}}
    check_run_config_expectations(solo, manifest, None)  # no raise

    # The explicit opt-out documents an intentional asymmetry.
    declared = json.loads(json.dumps(base))
    declared["allow_unequal_max_tokens"] = True
    check_run_config_expectations(declared, manifest, None)  # no raise


def test_baseline_thinking_fixture_declares_its_unequal_caps():
    path = _FIXTURES / "run_config.conditional_baseline_thinking_claude_kimi_qwen.json"
    rc = load_run_config(path)
    assert rc.get("allow_unequal_max_tokens") is True
