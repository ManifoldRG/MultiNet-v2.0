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
        "task_id", "experiment", "condition", "backend", "agent_or_model", "seed",
        "success", "terminated", "truncated", "reward", "steps", "optimal_steps",
        "optimality_ratio", "path_choice", "mechanism_interaction_order",
        "failure_point", "tokens", "raw_output_ref",
    ):
        assert field in row, f"missing episode_runs field: {field}"
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
