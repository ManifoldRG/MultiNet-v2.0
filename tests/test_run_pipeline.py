"""End-to-end test for the bare-bones run pipeline using a replay stub agent.

Runs are live-model-only in production, but the runner accepts any callable
``messages -> str`` agent, so a deterministic replay stub exercises the full
Stage 1->5 chain (real MiniGrid backend, episode log, and scorer) with no API.
"""

from __future__ import annotations

import json
from pathlib import Path

from interface.loader import default_maze_path
from interface.smoke_tests.plans import v01_empty_room_trajectory

from scripts.run_pipeline import resolve_task_rows, run_from_config, run_pipeline

_MANIFEST = Path(__file__).resolve().parents[1] / "gridworld" / "fixtures" / "manifest.json"


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
import itertools
import shutil

from scorer import load_scorer_config, score_task_file
from scorer.io import load_json, task_spec_from_payload
from scripts.run_pipeline import _expected_static_hash


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
                 agent_name="stub", artifacts_root=artifacts, run_set_id="r")
    calls_after_first = agent.calls
    assert calls_after_first > 0

    # Second identical run: episode cache hit -> agent not called again.
    run_pipeline(manifest_path=manifest, experiment="test1", agent=agent,
                 agent_name="stub", artifacts_root=artifacts, run_set_id="r")
    assert agent.calls == calls_after_first


def test_task_edit_invalidates_static_and_episode(tmp_path):
    task_file = tmp_path / "task.json"
    shutil.copy(default_maze_path("V01_empty_room.json"), task_file)
    manifest = _single_task_manifest(tmp_path, task_file)
    artifacts = tmp_path / "artifacts"
    agent = CountingReplayAgent(v01_empty_room_trajectory())

    run_pipeline(manifest_path=manifest, experiment="test1", agent=agent,
                 agent_name="stub", artifacts_root=artifacts, run_set_id="r")
    first_calls = agent.calls
    first_static_hash = load_json(artifacts / "tasks" / "copy_v01" / "scored_static.json")["inputs_hash"]

    # Mutate the task spec -> both static and run hashes must change.
    data = json.loads(task_file.read_text())
    data["max_steps"] = data["max_steps"] + 5
    task_file.write_text(json.dumps(data), encoding="utf-8")

    run_pipeline(manifest_path=manifest, experiment="test1", agent=agent,
                 agent_name="stub", artifacts_root=artifacts, run_set_id="r")
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
    run_pipeline(manifest_path=manifest, experiment="test1", agent=agent, agent_name="stub",
                 artifacts_root=artifacts, run_set_id="r", scorer_config=cfg_a)
    calls_after_first = agent.calls
    run_dir = artifacts / "runs" / "copy_v01" / "minigrid" / "stub" / "seed_0" / "default"
    eff_a = load_json(run_dir / "run_score.json")["signals"]["token_efficiency"]

    cfg_b = load_scorer_config()
    cfg_b.baseline_tokens = 5.0
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
