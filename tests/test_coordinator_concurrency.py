"""Coordinator assign must hand a served-vLLM worker up to worker_concurrency DISTINCT
units at once (so its episodes actually run in parallel), while a worker that declares
no concurrency keeps the original one-unit-per-worker (idempotent re-return) behaviour."""
import json

from scripts.distributed_run_pipeline import (
    CoordinatorStore,
    _distributed_root,
    plan_path,
    state_path,
)


def _setup(tmp_path, n_units=8):
    _distributed_root(tmp_path).mkdir(parents=True, exist_ok=True)
    units = []
    for i in range(n_units):
        tid = f"t{i}"
        td = tmp_path / "tasks" / tid
        td.mkdir(parents=True, exist_ok=True)
        (td / "canonical_paths.json").write_text("{}")
        (td / "scored_static.json").write_text("{}")
        units.append({"unit_id": f"u{i}", "model_group": "g", "task_id": tid,
                      "seed": 0, "prompt_variant": "current", "task_payload": {}})
    plan_path(tmp_path).write_text(json.dumps({
        "job_id": "j", "units": units, "scorer_config": {},
        "difficulty_max_static_score": 1000}))
    state_path(tmp_path).write_text(json.dumps({
        "job_id": "j", "workers": {},
        "units": {f"u{i}": {"status": "pending", "attempts": 0, "worker_id": None}
                  for i in range(n_units)},
    }))
    return tmp_path


def test_concurrent_worker_gets_distinct_units_up_to_concurrency(tmp_path):
    store = CoordinatorStore(_setup(tmp_path, 8))
    caps = {"model_group": "g", "worker_concurrency": 4}
    got = [store.assign("w1", caps)["unit"]["unit_id"] for _ in range(4)]
    assert len(set(got)) == 4                       # 4 DISTINCT units in flight
    fifth = store.assign("w1", caps)["unit"]["unit_id"]
    assert fifth in got                             # at cap -> re-return an active one


def test_serial_worker_gets_one_unit_re_returned(tmp_path):
    store = CoordinatorStore(_setup(tmp_path, 8))
    caps = {"model_group": "g"}                     # no worker_concurrency -> defaults to 1
    first = store.assign("w1", caps)["unit"]["unit_id"]
    second = store.assign("w1", caps)["unit"]["unit_id"]
    assert first == second                          # same unit re-returned (serial)
