"""Phase provenance on episode artifacts (Task B3).

A run-config may carry a top-level ``"phase": {"pass": int, "label": str}`` block
that labels the whole run invocation (the Qwen two-tier rerun: phase-1 wide/8k,
phase-2 deep/64k). Phase is *provenance*, not a generation input, so it must:

  * appear on ``episode.json`` (top-level ``pass`` + the caps it ran under),
  * appear in ``run_inputs.json`` (extras),
  * appear as a ``pass`` column on the ``episode_runs.jsonl`` row (default 1),
  * and NEVER change any episode/unit input hash (or already-paid distributed
    units get orphaned and re-run — real money).
"""

from __future__ import annotations

import json
from pathlib import Path

from interface.loader import default_maze_path
from interface.smoke_tests.plans import v01_empty_room_trajectory
from scorer.io import load_json

from pipeline import episode_metrics
from scripts.run_pipeline import run_from_config

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST = _REPO_ROOT / "gridworld" / "fixtures" / "manifest.json"
_STABLE_DIFFICULTY_MAX = 1000.0
_TASK_ID = "validation_10_v01_empty_room"


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


def _factory(name, model_cfg):
    return ReplayAgent(v01_empty_room_trajectory()), model_cfg["model"]


def _base_run_config() -> dict:
    return {
        "models": {
            "qwen_phase": {
                "provider": "qwen_vllm",
                "model": "qwen-stub",
                "max_tokens": 64000,
                "max_model_len": 96000,
                "tasks": [str(default_maze_path("V01_empty_room.json"))],
            }
        }
    }


def _write_cfg(path: Path, cfg: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def _run(tmp_path: Path, cfg: dict, artifacts_name: str = "artifacts") -> Path:
    artifacts = tmp_path / artifacts_name
    run_from_config(
        run_config_path=_write_cfg(tmp_path / f"{artifacts_name}_config.json", cfg),
        manifest_path=_MANIFEST,
        seeds=[0],
        artifacts_root=artifacts,
        run_set_id="phase",
        agent_factory=_factory,
        difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
    )
    return artifacts


def _run_dir(artifacts: Path) -> Path:
    return artifacts / "runs" / _TASK_ID / "minigrid" / "qwen-stub" / "seed_0" / "default"


# --------------------------------------------------------------------------- #
# Provenance recording
# --------------------------------------------------------------------------- #
def test_phase_reaches_episode_run_inputs_and_row(tmp_path):
    cfg = _base_run_config()
    cfg["phase"] = {"pass": 2, "label": "qwen_phase2"}
    artifacts = _run(tmp_path, cfg)

    run_dir = _run_dir(artifacts)
    episode = load_json(run_dir / "episode.json")
    assert episode["pass"] == 2
    assert episode["phase_label"] == "qwen_phase2"
    # Caps it ran under, recorded on the episode itself.
    assert episode["max_tokens"] == 64000
    assert episode["max_model_len"] == 96000

    sidecar = load_json(run_dir / "run_inputs.json")
    assert sidecar["phase"] == {"pass": 2, "label": "qwen_phase2"}

    rows = [
        json.loads(line)
        for line in (artifacts / "episode_runs.jsonl").read_text().strip().splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["pass"] == 2
    assert rows[0]["max_tokens"] == 64000


def test_absent_phase_defaults_row_pass_1(tmp_path):
    artifacts = _run(tmp_path, _base_run_config())

    run_dir = _run_dir(artifacts)
    episode = load_json(run_dir / "episode.json")
    # No phase declared -> no phase stamp on the episode (kept clean).
    assert "pass" not in episode
    assert "phase_label" not in episode

    sidecar = load_json(run_dir / "run_inputs.json")
    assert "phase" not in sidecar

    rows = [
        json.loads(line)
        for line in (artifacts / "episode_runs.jsonl").read_text().strip().splitlines()
    ]
    assert rows[0]["pass"] == 1  # build_run_row default
    assert rows[0]["max_tokens"] is None


# --------------------------------------------------------------------------- #
# build_run_row column defaults (unit)
# --------------------------------------------------------------------------- #
def test_build_run_row_pass_and_max_tokens_columns():
    canonical = {"bfs": {"optimal_steps": 0}}
    manifest_row = {"task_id": "t", "experiment": "test1", "condition": "default"}

    base = {"success": True, "end_reason": "success", "steps_used": 0, "transcript": []}
    row = episode_metrics.build_run_row(
        base, canonical, manifest_row, agent_or_model="m", seed=0
    )
    assert row["pass"] == 1  # default when absent
    assert row["max_tokens"] is None

    stamped = dict(base, **{"pass": 2, "max_tokens": 64000})
    row2 = episode_metrics.build_run_row(
        stamped, canonical, manifest_row, agent_or_model="m", seed=0
    )
    assert row2["pass"] == 2
    assert row2["max_tokens"] == 64000


# --------------------------------------------------------------------------- #
# HASH INVARIANCE — the load-bearing property
# --------------------------------------------------------------------------- #
def test_episode_inputs_hash_invariant_to_phase(tmp_path):
    """Adding/removing the top-level phase block must not change the per-run
    ``inputs_hash`` in run_inputs.json (that hash keys the episode cache and the
    distributed upload verification)."""
    without = _run(tmp_path, _base_run_config(), artifacts_name="no_phase")

    cfg = _base_run_config()
    cfg["phase"] = {"pass": 2, "label": "qwen_phase2"}
    with_phase = _run(tmp_path, cfg, artifacts_name="with_phase")

    h_without = load_json(_run_dir(without) / "run_inputs.json")["inputs_hash"]
    h_with = load_json(_run_dir(with_phase) / "run_inputs.json")["inputs_hash"]
    assert h_without == h_with


def test_distributed_unit_ids_invariant_to_phase(tmp_path):
    """prepare_job derives job_id from the run_config and unit_id from job_id.
    A top-level phase block must not churn job_id or any unit_id, or a
    re-labeled phase orphans already-paid units."""
    from scripts.distributed_run_pipeline import prepare_job

    def run_plan(cfg: dict, name: str) -> dict:
        artifacts = tmp_path / name
        return prepare_job(
            run_config_path=_write_cfg(tmp_path / f"{name}_config.json", cfg),
            manifest_path=_MANIFEST,
            seeds=[0],
            conditions=None,
            artifacts_root=artifacts,
            run_set_id="dist",
            difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
        )

    plan_without = run_plan(_base_run_config(), "d_no_phase")
    cfg = _base_run_config()
    cfg["phase"] = {"pass": 2, "label": "qwen_phase2"}
    plan_with = run_plan(cfg, "d_with_phase")

    assert plan_without["job_id"] == plan_with["job_id"]
    ids_without = sorted(u["unit_id"] for u in plan_without["units"])
    ids_with = sorted(u["unit_id"] for u in plan_with["units"])
    assert ids_without == ids_with
    assert ids_without  # non-empty
