"""An infra-killed unit is resumable, not verified.

An episode that ends on ``agent_error:*`` (dead transport, billing, an API 400)
is a partial run. The coordinator used to verify it like a finished one, so a
partly-killed run looked complete. Now it records the end reason, marks the
unit resumable and requeues it (no attempt burned), reports it separately with
its stall counter and steps left, and keeps it out of the final rows.
"""

from __future__ import annotations

import base64
import io
import json
import tarfile
from pathlib import Path

import pytest

from interface.agents.billing import BillingExhausted
from interface.loader import default_maze_path
from interface.smoke_tests.plans import v01_empty_room_trajectory
from scripts.distributed_run_pipeline import (
    CoordinatorStore,
    finalize_job,
    package_run_archive,
    prepare_job,
    run_assigned_unit,
)

_REPO = Path(__file__).resolve().parents[1]
_MANIFEST = _REPO / "gridworld" / "fixtures" / "manifest.json"
STALL_K = 30


def _prepare(tmp_path: Path) -> Path:
    cfg = tmp_path / "rc.json"
    cfg.write_text(json.dumps({
        "experiment_config": {"progress_stall_k": STALL_K},
        "models": {"stub": {"provider": "claude", "model": "stub-model", "group": "stub",
                            "tasks": [str(default_maze_path("V01_empty_room.json"))]}},
    }), encoding="utf-8")
    artifacts = tmp_path / "coordinator"
    prepare_job(run_config_path=cfg, manifest_path=_MANIFEST, seeds=[0], conditions=None,
                artifacts_root=artifacts, run_set_id="life", difficulty_max_static_score=1000.0)
    return artifacts


class _Agent:
    """Replays V01's plan; raises ``fail_with`` on query ``fail_at`` (1-based)."""

    def __init__(self, fail_at=None, fail_with=None):
        self._actions = iter(v01_empty_room_trajectory())
        self._fail_at, self._fail_with = fail_at, fail_with
        self.calls = 0
        self.last_usage = None

    def __call__(self, messages):
        self.calls += 1
        if self._fail_at is not None and self.calls >= self._fail_at:
            raise self._fail_with
        self.last_usage = {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}
        return f"FINAL_OUTPUT: {next(self._actions, 'DONE')}"


def _run_and_upload(artifacts: Path, tmp_path: Path, agent: _Agent, worker: str = "w") -> tuple[CoordinatorStore, dict, dict]:
    store = CoordinatorStore(artifacts)
    worker_id = store.register({"worker_id": worker, "capabilities": {"model_group": "stub"}})["worker_id"]
    unit = store.assign(worker_id)["unit"]
    root = tmp_path / f"worker_{worker}_{agent.calls}_{id(agent)}"
    run_assigned_unit(unit, artifacts_root=root, agent_factory=lambda k, c: (agent, "stub-model"))
    result = store.upload(worker_id, unit["unit_id"], package_run_archive(unit, artifacts_root=root).read_bytes())
    return store, unit, result


def test_a_finished_episode_still_verifies_and_records_its_end_reason(tmp_path):
    store, unit, result = _run_and_upload(_prepare(tmp_path), tmp_path, _Agent())
    assert result["status"] == "verified"
    state = store.load_state()["units"][unit["unit_id"]]
    assert state["status"] == "verified" and state["end_reason"] == "success"
    assert store.status()["infra_killed"] == []


def test_an_agent_error_upload_is_resumable_not_verified(tmp_path):
    store, unit, result = _run_and_upload(_prepare(tmp_path), tmp_path, _Agent(3, RuntimeError("connection reset")))
    assert result["status"] == "resumable"
    state = store.load_state()["units"][unit["unit_id"]]
    assert state["status"] == "resumable"
    assert state["end_reason"].startswith("agent_error:RuntimeError: connection reset")
    kill = state["infra_kill"]
    assert kill["steps_used"] == 2 and kill["query_count"] == 2
    assert kill["steps_left"] == kill["max_steps"] - 2
    assert kill["stall_k"] == STALL_K and 0 <= kill["stall_count"] <= STALL_K


def test_status_lists_infra_killed_units_separately(tmp_path):
    store, unit, _ = _run_and_upload(_prepare(tmp_path), tmp_path, _Agent(3, RuntimeError("connection reset")))
    status = store.status()
    assert status["units"] == {"resumable": 1}
    [row] = status["infra_killed"]
    assert row["unit_id"] == unit["unit_id"] and row["task_id"] == unit["task_id"]
    assert row["model_id"] == "stub-model" and row["status"] == "resumable"
    assert row["stall"] == f"{row['stall_count']}/{STALL_K}"
    assert row["steps_left"] > 0 and row["queries"] == 2 and row["resumes"] == 0


def test_a_resumable_unit_is_handed_out_again_with_its_archive_and_no_attempt_burned(tmp_path):
    store, unit, _ = _run_and_upload(_prepare(tmp_path), tmp_path, _Agent(3, RuntimeError("connection reset")))
    attempts = store.load_state()["units"][unit["unit_id"]]["attempts"]
    again = store.assign("w2", {"model_group": "stub"})["unit"]
    assert again["unit_id"] == unit["unit_id"]
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(again["resume"]["archive_b64"])), mode="r:gz") as tar:
        episode = json.load(tar.extractfile("episode.json"))
        run_inputs = json.load(tar.extractfile("run_inputs.json"))
    assert episode["end_reason"].startswith("agent_error:")
    assert run_inputs["inputs_hash"] == unit["episode_inputs_hash"]
    assert again["resume"]["resume_count"] == 1
    state = store.load_state()["units"][unit["unit_id"]]
    assert state["attempts"] == attempts and state["resume_count"] == 1


def test_a_unit_killed_past_max_resumes_fails(tmp_path):
    artifacts = _prepare(tmp_path)
    statuses = []
    for i in range(3):  # MAX_RESUMES = 2: the third kill is terminal
        _store, _unit, result = _run_and_upload(artifacts, tmp_path, _Agent(3, RuntimeError(f"reset {i}")), worker=f"w{i}")
        statuses.append(result["status"])
    assert statuses == ["resumable", "resumable", "failed"]
    store = CoordinatorStore(artifacts)
    [row] = store.status()["infra_killed"]
    assert row["status"] == "failed" and "reset 2" in row["end_reason"]
    assert store.assign("w9", {"model_group": "stub"})["unit"] is None


def test_billing_kills_do_not_count_against_max_resumes(tmp_path):
    artifacts = _prepare(tmp_path)
    statuses = []
    for i in range(4):
        store, _u, result = _run_and_upload(
            artifacts, tmp_path, _Agent(3, BillingExhausted("claude", f"credit balance too low {i}")), worker=f"w{i}")
        statuses.append(result["status"])
        store.unpark("claude")  # a billing kill parks the provider until a top-up
    assert statuses == ["resumable"] * 4


def test_finalize_refuses_while_units_are_infra_killed_and_reports_them_when_partial(tmp_path):
    artifacts = _prepare(tmp_path)
    _run_and_upload(artifacts, tmp_path, _Agent(3, RuntimeError("connection reset")))
    with pytest.raises(RuntimeError, match="infra-killed"):
        finalize_job(artifacts_root=artifacts, run_set_id="life")
    result = finalize_job(artifacts_root=artifacts, run_set_id="life", allow_partial=True)
    assert result["run_count"] == 0
    [row] = result["infra_killed"]
    assert row["status"] == "resumable" and row["steps_left"] > 0
    assert CoordinatorStore(artifacts).load_state()["units"][row["unit_id"]]["status"] == "resumable"
