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

from scripts.run_pipeline import run_pipeline


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
