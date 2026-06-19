"""Tests for the gsutil-backed bucket mirroring used by the coordinator."""

from __future__ import annotations

import json

import pytest

from scripts.gcs_storage import StorageConfig, gcs_destination, mirror_to_bucket


class _Result:
    def __init__(self, returncode: int):
        self.returncode = returncode
        self.stdout = ""
        self.stderr = "" if returncode == 0 else "boom"


class _FakeRun:
    def __init__(self, returncodes):
        self.returncodes = list(returncodes)
        self.commands: list[list[str]] = []

    def __call__(self, cmd, capture_output=True, text=True, timeout=None):
        self.commands.append(cmd)
        return _Result(self.returncodes.pop(0))


def test_gcs_destination_joins_cleanly():
    assert gcs_destination("gs://b/prefix/", "rs", "runs/x/") == "gs://b/prefix/rs/runs/x"
    assert gcs_destination("gs://b", "", "runs/x") == "gs://b/runs/x"
    assert gcs_destination("gs://b/") == "gs://b"


def test_storage_config_from_file_and_active(tmp_path):
    p = tmp_path / "storage.json"
    p.write_text(json.dumps({"bucket": "gs://b/p", "enabled": True}), encoding="utf-8")
    cfg = StorageConfig.from_file(p)
    assert cfg.bucket == "gs://b/p"
    assert cfg.active

    p.write_text(json.dumps({"bucket": "gs://b/p", "enabled": False}), encoding="utf-8")
    assert not StorageConfig.from_file(p).active

    p.write_text(json.dumps({"enabled": True}), encoding="utf-8")
    assert not StorageConfig.from_file(p).active  # no bucket -> inactive


def test_mirror_directory_uses_rsync(tmp_path):
    run = _FakeRun([0])
    src = tmp_path / "run"
    src.mkdir()
    mirror_to_bucket(src, "gs://b/p/run", run=run, sleep=lambda s: None)
    assert run.commands == [["gsutil", "-m", "rsync", "-r", str(src), "gs://b/p/run"]]


def test_mirror_file_uses_cp(tmp_path):
    run = _FakeRun([0])
    src = tmp_path / "episode_runs.jsonl"
    src.write_text("{}\n", encoding="utf-8")
    mirror_to_bucket(src, "gs://b/p/episode_runs.jsonl", run=run, sleep=lambda s: None)
    assert run.commands == [["gsutil", "cp", str(src), "gs://b/p/episode_runs.jsonl"]]


def test_mirror_retries_then_succeeds(tmp_path):
    run = _FakeRun([1, 0])
    src = tmp_path / "run"
    src.mkdir()
    mirror_to_bucket(src, "gs://b/p/run", run=run, sleep=lambda s: None)
    assert len(run.commands) == 2


def test_mirror_raises_after_max_attempts(tmp_path):
    run = _FakeRun([1, 1, 1])
    src = tmp_path / "run"
    src.mkdir()
    with pytest.raises(RuntimeError, match="gsutil"):
        mirror_to_bucket(src, "gs://b/p/run", run=run, max_attempts=3, sleep=lambda s: None)
    assert len(run.commands) == 3
