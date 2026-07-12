"""prepare_job state-merge on a same-job re-prepare (reused-fleet re-run).

Invariant: re-preparing the same job (e.g. next-batch on a reused fleet after
fixing an infra bug) must KEEP completed units (verified/uploaded) so paid work
isn't re-run, but RESET every non-terminal unit (failed/stale/assigned/running/
pending) to fresh pending/attempts=0 — otherwise a prior run that failed all
units at the attempt cap permanently strands the batch (coordinator dispatches
nothing, workers idle).
"""
from __future__ import annotations

from scripts.distributed_run_pipeline import _reset_state_for_rerun


def _fresh(uids):
    return {"job_id": "j",
            "units": {u: {"status": "pending", "attempts": 0, "worker_id": None}
                      for u in uids}}


def test_keeps_verified_and_uploaded_resets_the_rest():
    fresh = _fresh(["a", "b", "c", "d", "e"])
    existing = {"job_id": "j", "units": {
        "a": {"status": "verified", "attempts": 1, "worker_id": "w1"},
        "b": {"status": "uploaded", "attempts": 1, "worker_id": "w1"},
        "c": {"status": "failed", "attempts": 3, "worker_id": "w2"},   # at cap
        "d": {"status": "running", "attempts": 2, "worker_id": "w3"},  # crashed mid-run
        "e": {"status": "stale", "attempts": 1, "worker_id": "w3"},
    }}
    out = _reset_state_for_rerun(existing, fresh)
    # completed work preserved
    assert out["units"]["a"]["status"] == "verified"
    assert out["units"]["b"]["status"] == "uploaded"
    # everything else reset so it can be re-dispatched
    for u in ("c", "d", "e"):
        assert out["units"][u]["status"] == "pending"
        assert out["units"][u]["attempts"] == 0


def test_new_units_not_in_existing_are_pending():
    fresh = _fresh(["a", "x"])          # x is new this prepare
    existing = {"job_id": "j", "units": {"a": {"status": "verified", "attempts": 1}}}
    out = _reset_state_for_rerun(existing, fresh)
    assert out["units"]["a"]["status"] == "verified"
    assert out["units"]["x"]["status"] == "pending"
