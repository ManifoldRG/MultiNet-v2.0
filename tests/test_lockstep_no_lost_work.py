"""Lockstep fleet: never lose paid work.

Each test pins a way the mechanics-15 3D run (mech15-3d-20261023) lost or
stranded paid work on a lockstep (batch API) worker: a billing stop that deleted
the units' checkpoints, a finished unit held in memory through the next round's
batch, and a restarted worker that got back only one of the units it held.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from interface.agents.billing import BillingExhausted
from interface.agents.reply import Reply
from interface.config import ExperimentConfig
from interface.episode_checkpoint import resume_stepper
from interface.loader import default_maze_path
from interface.smoke_tests.plans import v01_empty_room_trajectory
from pipeline.run_stage3 import build_episode_runner
from scripts import distributed_run_pipeline as drp
from scripts.distributed_run_pipeline import CoordinatorStore, prepare_job, run_assigned_unit, run_lockstep_worker

_REPO = Path(__file__).resolve().parents[1]
_MANIFEST = _REPO / "gridworld" / "fixtures" / "manifest.json"
PLAN = v01_empty_room_trajectory()
V01 = default_maze_path("V01_empty_room.json")
CAPS = {"model_group": "claude-api"}
_USAGE = {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}


def _job(tmp_path: Path, seeds=(0,)) -> CoordinatorStore:
    cfg = tmp_path / "rc.json"
    cfg.write_text(json.dumps({"models": {"opus": {"provider": "claude", "model": "stub-model",
                                                   "group": "claude-api", "tasks": [str(V01)]}}}), encoding="utf-8")
    artifacts = tmp_path / "coordinator"
    prepare_job(run_config_path=cfg, manifest_path=_MANIFEST, seeds=list(seeds), conditions=None,
                artifacts_root=artifacts, run_set_id="nolost", difficulty_max_static_score=1000.0)
    return CoordinatorStore(artifacts)


class _PlanBatch:
    """Batch agent replaying V01's plan, ``start`` steps in (a resumed run); every
    episode in a round gets the same step. Round ``fail_at`` raises ``error``."""

    def __init__(self, start=0, fail_at=None, error=None):
        self.start, self.fail_at, self.error = start, fail_at, error
        self.rounds = 0

    def generate_batch(self, batch):
        self.rounds += 1
        if self.fail_at is not None and self.rounds >= self.fail_at:
            raise self.error
        step = self.start + self.rounds - 1
        token = PLAN[step] if step < len(PLAN) else "DONE"
        return [Reply(text=f"FINAL_OUTPUT: {token}", usage=_USAGE) for _ in batch]


class _PlanSerial:
    """The serial counterpart: call ``fail_at`` (1-based) raises ``error``."""

    def __init__(self, start=0, fail_at=None, error=None):
        self.start, self.fail_at, self.error = start, fail_at, error
        self.calls = 0
        self.last_usage = None

    def __call__(self, messages):
        self.calls += 1
        if self.fail_at is not None and self.calls >= self.fail_at:
            raise self.error
        self.last_usage = _USAGE
        step = self.start + self.calls - 1
        return f"FINAL_OUTPUT: {PLAN[step] if step < len(PLAN) else 'DONE'}"


def _lockstep(store: CoordinatorStore, root: Path, agent, max_batches: int = 1, **kwargs) -> dict:
    return run_lockstep_worker(artifacts_root=root, capabilities=CAPS, max_batches=max_batches, client=store,
                               agent_factory=lambda k, c: (agent, "stub-model"), **kwargs)


def _billing() -> BillingExhausted:
    return BillingExhausted("claude", "credit balance too low")


# --------------------------------------------------------------------------- #
# A billing stop keeps the checkpoint
# --------------------------------------------------------------------------- #
def test_a_billing_stop_keeps_each_units_checkpoint_at_the_uploaded_boundary(tmp_path):
    store = _job(tmp_path, seeds=(0, 1))
    root = tmp_path / "lock"
    _lockstep(store, root, _PlanBatch(fail_at=3, error=_billing()), max_batches=2)

    uploads = drp._uploads_root(store.artifacts_root)
    for unit in store.load_plan()["units"]:
        assert store.load_state()["units"][unit["unit_id"]]["status"] == "resumable"
        checkpoint = root / unit["run_dir"] / "checkpoint.json"
        assert checkpoint.exists()  # a continuation (resume_from_archive --unit-id) starts here
        partial = json.loads((root / unit["run_dir"] / "episode.json").read_text())
        runner = build_episode_runner(V01, ExperimentConfig.from_dict(unit["experiment_config"]), int(unit["seed"]),
                                      max_steps=partial["task_spec"]["max_steps"])
        stepper = resume_stepper(checkpoint, runner=runner)
        assert stepper.query_count == partial["query_count"] == 2
        assert stepper.step_index == partial["steps_used"]
        with tarfile.open(uploads / f"{unit['unit_id']}.tar.gz") as tar:
            assert not [n for n in tar.getnames() if n.startswith("checkpoint.json")]  # worker-local only


def test_a_non_billing_error_still_drops_the_checkpoint(tmp_path):
    store = _job(tmp_path)
    root = tmp_path / "lock"
    result = _lockstep(store, root, _PlanBatch(fail_at=3, error=RuntimeError("Anthropic Batches HTTP 500")))
    unit = store.load_plan()["units"][0]
    assert set(result["failed"]) == {unit["unit_id"]}  # retried up to the attempt cap, as before
    assert store.load_state()["units"][unit["unit_id"]]["status"] == "failed"
    assert not (root / unit["run_dir"] / "checkpoint.json").exists()


def test_a_kept_checkpoint_never_shadows_a_newer_partial(tmp_path):
    """W1 parks the unit at query 2 and keeps its checkpoint. Another worker
    resumes it and parks it again at query 3. When W1 gets the unit back it must
    continue from the coordinator's newer partial, not re-pay query 3 from its own
    older checkpoint."""
    store = _job(tmp_path)
    w1 = tmp_path / "w1"
    _lockstep(store, w1, _PlanBatch(fail_at=3, error=_billing()))
    unit_id = store.load_plan()["units"][0]["unit_id"]
    store.unpark("claude")

    again = store.assign("w2", CAPS)["unit"]
    run_assigned_unit(again, artifacts_root=tmp_path / "w2",
                      agent_factory=lambda k, c: (_PlanSerial(start=2, fail_at=2, error=_billing()), "stub-model"))
    store.upload("w2", unit_id, drp.package_run_archive(again, artifacts_root=tmp_path / "w2").read_bytes())
    assert store.load_state()["units"][unit_id]["infra_kill"]["query_count"] == 3
    store.unpark("claude")

    agent = _PlanBatch(start=3)
    _lockstep(store, w1, agent)
    state = store.load_state()["units"][unit_id]
    episode = json.loads((store.artifacts_root / again["run_dir"] / "episode.json").read_text())
    assert state["status"] == "verified" and episode["success"] is True
    assert episode["resumed_from"]["killed_at_query"] == 3
    assert agent.rounds == len(PLAN) - 1 - 3  # only the queries after the newer partial were paid for


# --------------------------------------------------------------------------- #
# A finished unit is uploaded before the next round's batch
# --------------------------------------------------------------------------- #
class _CrashOnRound:
    """Wraps a batch agent; round ``crash_on`` kills the process mid-batch."""

    def __init__(self, inner, crash_on):
        self._inner, self._crash_on = inner, crash_on
        self.rounds = 0

    def generate_batch(self, batch):
        self.rounds += 1
        if self.rounds == self._crash_on:
            raise KeyboardInterrupt  # the worker VM dies while the batch is in flight
        return self._inner.generate_batch(batch)


def test_a_finished_unit_is_uploaded_before_the_next_rounds_batch(tmp_path):
    """corr_a finishes after round 1 while corr_b still needs a query. Round 2's
    batch can run for hours; corr_a must already be on the coordinator when the
    worker dies during it, not in memory."""
    from tests.test_lockstep_worker import ScriptedBatchAgent, _scripts_for, _write_job

    artifacts, plan = _write_job(tmp_path)
    store = CoordinatorStore(artifacts)
    agent = _CrashOnRound(ScriptedBatchAgent(_scripts_for()), crash_on=2)
    with pytest.raises(KeyboardInterrupt):
        run_lockstep_worker(artifacts_root=tmp_path / "w", capabilities={"model_group": "claude-batch"},
                            max_batches=2, client=store, agent_factory=lambda k, c: (agent, "stub-batch-model"))
    by_task = {u["task_id"]: u["unit_id"] for u in plan["units"]}
    state = store.load_state()["units"]
    assert state[by_task["corr_a"]]["status"] == "verified"
    assert state[by_task["corr_b"]]["status"] == "running"


class _FlakyUploads(CoordinatorStore):
    """The first upload fails (a coordinator 502); later ones land."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.failures_left = 1

    def upload(self, worker_id, unit_id, archive_bytes):
        if self.failures_left:
            self.failures_left -= 1
            raise RuntimeError("Coordinator request failed (502)")
        return super().upload(worker_id, unit_id, archive_bytes)


class _Script:
    """Batch agent answering ``tokens`` in round order."""

    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.rounds = 0

    def generate_batch(self, batch):
        self.rounds += 1
        token = self.tokens[self.rounds - 1] if self.rounds <= len(self.tokens) else "DONE"
        return [Reply(text=f"FINAL_OUTPUT: {token}", usage=_USAGE) for _ in batch]


def test_a_finished_unit_whose_upload_failed_keeps_its_checkpoint(tmp_path):
    """The checkpoint goes only once the result is on the coordinator. When the
    upload fails, the retry resumes at the last query boundary: it re-pays the
    final query, not the whole episode."""
    _job(tmp_path)
    store = _FlakyUploads(tmp_path / "coordinator")
    queries = len(PLAN) - 1
    agent = _Script(PLAN[:queries] + [PLAN[queries - 1]])  # the retry re-asks the last query
    _lockstep(store, tmp_path / "w", agent)
    unit = store.load_plan()["units"][0]
    episode = json.loads((store.artifacts_root / unit["run_dir"] / "episode.json").read_text())
    assert store.load_state()["units"][unit["unit_id"]]["status"] == "verified"
    assert episode["success"] is True and episode["query_count"] == queries
    assert agent.rounds == queries + 1
