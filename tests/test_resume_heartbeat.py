"""A standard-API unit resume heartbeats as the worker it resumes for.

``scripts/resume_from_archive.py --unit-id`` can drive one lockstep unit for
hours. Without heartbeats the coordinator marks the unit stale after
``STALE_AFTER`` and hands it out again as a fresh attempt: a second, paid run of
the same unit, whose late upload the coordinator then refuses.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from interface.loader import default_maze_path
from scripts import resume_from_archive
from scripts.distributed_run_pipeline import CoordinatorStore, prepare_job

_REPO = Path(__file__).resolve().parents[1]
_MANIFEST = _REPO / "gridworld" / "fixtures" / "manifest.json"


def _assigned_unit(tmp_path: Path, stale_after: float):
    cfg = tmp_path / "rc.json"
    cfg.write_text(json.dumps({"models": {
        "stub": {"provider": "scripted", "model": "stub", "group": "g",
                 "tasks": [str(default_maze_path("V01_empty_room.json"))]},
    }}), encoding="utf-8")
    artifacts = tmp_path / "coordinator"
    prepare_job(run_config_path=cfg, manifest_path=_MANIFEST, seeds=[0], conditions=None,
                artifacts_root=artifacts, run_set_id="hb", difficulty_max_static_score=1000.0)
    store = CoordinatorStore(artifacts, stale_after_seconds=stale_after)
    unit = store.assign("w1", {"model_group": "g"})["unit"]
    assert unit is not None
    return store, unit


class _InProcessClient:
    def __init__(self, store: CoordinatorStore):
        self.store = store
        self.beats: list[int | None] = []

    def heartbeat(self, worker_id, unit_id=None, progress=None):
        self.beats.append(progress)
        return self.store.heartbeat(worker_id, unit_id, progress)


def test_without_heartbeats_the_unit_is_handed_out_again(tmp_path):
    store, unit = _assigned_unit(tmp_path, stale_after=0.2)
    time.sleep(0.4)
    again = store.assign("w2", {"model_group": "g"})["unit"]
    assert again is not None and again["unit_id"] == unit["unit_id"]  # the double-drive


def test_a_resume_heartbeats_so_its_unit_is_not_handed_out_again(tmp_path):
    store, unit = _assigned_unit(tmp_path, stale_after=0.2)
    client = _InProcessClient(store)
    progress = {"q": 3}
    with resume_from_archive.heartbeat_while(client, "w1", unit["unit_id"], lambda: progress["q"], every=0.05):
        time.sleep(0.25)
        progress["q"] = 5
        time.sleep(0.25)
        assert store.assign("w2", {"model_group": "g"})["unit"] is None
    assert client.beats[0] == 3 and client.beats[-1] == 5
    assert store.load_state()["units"][unit["unit_id"]]["progress"] == 5
    count = len(client.beats)
    time.sleep(0.15)
    assert len(client.beats) == count  # stops with the resume


def test_a_failing_heartbeat_never_stops_the_resume(tmp_path):
    class _Down:
        calls = 0

        def heartbeat(self, *args, **kwargs):
            _Down.calls += 1
            raise OSError("coordinator unreachable")

    logged: list[str] = []
    with resume_from_archive.heartbeat_while(_Down(), "w1", "u1", lambda: 0, every=0.02, log=logged.append):
        time.sleep(0.1)
    assert _Down.calls >= 2 and logged and "heartbeat" in logged[0]
