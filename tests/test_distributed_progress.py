from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.distributed_run_pipeline import CoordinatorStore, plan_path


def _write_plan(root: Path, n_units: int, group: str = "qwen36-27b", max_in_flight: int = 2) -> dict:
    units = [
        {
            "unit_id": f"u{i}",
            "task_id": f"t{i}",
            "model_group": group,
            "hardware_profile": "local-gpu",
            "max_in_flight": max_in_flight,
        }
        for i in range(n_units)
    ]
    plan = {
        "job_id": "job_test",
        "units": units,
        "scorer_config": {},
        "difficulty_max_static_score": 1000.0,
    }
    p = plan_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(plan))
    return plan


def _caps() -> dict:
    return {"model_groups": ["qwen36-27b"], "hardware_profile": "local-gpu"}


def _store(tmp_path: Path, monkeypatch, n_units: int) -> CoordinatorStore:
    _write_plan(tmp_path, n_units)
    store = CoordinatorStore(tmp_path)
    # assign() builds a payload that reads per-task files; we only care about
    # which unit is selected, so return the raw unit dict instead.
    monkeypatch.setattr(store, "_unit_payload", lambda plan, unit: dict(unit))
    return store


def test_work_stealing_two_workers_three_units(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch, n_units=3)

    a = store.assign("A", _caps())["unit"]["unit_id"]
    b = store.assign("B", _caps())["unit"]["unit_id"]
    assert a != b                      # parallel pickup of two distinct units

    # group is at max_in_flight=2 -> a fresh worker gets nothing yet
    assert store.assign("C", _caps())["unit"] is None

    # A finishes its unit -> frees an in-flight slot
    state = store.load_state()
    state["units"][a]["status"] = "verified"
    store.save_state(state)

    third = store.assign("A", _caps())["unit"]["unit_id"]
    assert third not in {a, b}         # work-stealing: A claims the 3rd unit

    # all units now assigned/verified -> empty queue -> clean None
    assert store.assign("C", _caps())["unit"] is None
