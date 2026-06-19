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
    _expected_static_hash,
    condition_variant_names,
    load_run_config,
    resolve_task_rows,
    run_from_config,
    run_pipeline,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIXTURES = _REPO_ROOT / "gridworld" / "fixtures"
_MANIFEST = _FIXTURES / "manifest.json"
_OGBENCH_50_MANIFEST = _FIXTURES / "manifest.ogbench_50_smbd.json"
_COORDINATOR_SMOKE_MANIFEST = _FIXTURES / "manifest.coordinator_smoke_validation10_ogbench.json"
_COORDINATOR_SMOKE_RUN_CONFIG = _FIXTURES / "run_config.coordinator_smoke_qwen_kimi.json"
_PENDING_VALIDATION10_CONFIGS = {
    "Prompt": _FIXTURES / "run_config.validation10_prompt_claude_kimi_qwen.json",
    "Observation format": _FIXTURES / "run_config.validation10_observation_format_claude_kimi_qwen.json",
    "Context window": _FIXTURES / "run_config.validation10_context_window_claude_kimi_qwen.json",
    "Querying strategy": _FIXTURES / "run_config.validation10_querying_strategy_claude_kimi_qwen.json",
}
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
        "Prompt": 2,
        "Observation format": 3,
        "Context window": 2,
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
    assert "manifest.ogbench_50_smbd.json" in smoke["selection"]["b_family_omission"]


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


def test_task_edit_invalidates_static_and_episode(tmp_path):
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

    # Mutate the task spec -> both static and run hashes must change.
    data = json.loads(task_file.read_text())
    data["max_steps"] = data["max_steps"] + 5
    task_file.write_text(json.dumps(data), encoding="utf-8")

    run_pipeline(manifest_path=manifest, experiment="test1", agent=agent,
                 agent_name="stub", artifacts_root=artifacts, run_set_id="r",
                 difficulty_max_static_score=_STABLE_DIFFICULTY_MAX)
    new_static_hash = load_json(artifacts / "tasks" / "copy_v01" / "scored_static.json")["inputs_hash"]
    assert new_static_hash != first_static_hash  # Stage 2 recomputed
    assert agent.calls > first_calls             # Stage 3 episode re-run


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
        conditions="Prompt",  # implemented variants: standard, verbose
        artifacts_root=artifacts,
        run_set_id="variants",
        difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
    )

    task_id = "validation_10_v01_empty_room"
    base = artifacts / "runs" / task_id / "minigrid" / "replay-stub" / "seed_0"
    assert (base / "standard" / "episode.json").exists()
    assert (base / "verbose" / "episode.json").exists()

    rows = [
        json.loads(line)
        for line in (artifacts / "episode_runs.jsonl").read_text().strip().splitlines()
    ]
    assert {r["prompt_variant"] for r in rows} == {"standard", "verbose"}
    # Same task-intrinsic condition, distinct prompt variants -> distinct rows.
    assert all(r["condition"] == "default" for r in rows)
    summary = payloads["scoring_calibration_summary"]
    assert summary["run_count"] == 2
    assert set(summary["success_rate_by_prompt_variant"]) == {"standard", "verbose"}


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

    result = store.upload(worker_id, unit["unit_id"], _dummy_run_archive())

    assert result["status"] == "verified"
    run_dir = artifacts / unit["run_dir"]
    assert (run_dir / "episode.json").exists()
    assert (run_dir / "run_inputs.json").exists()
    assert (run_dir / "run_score.json").exists()


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
