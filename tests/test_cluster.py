from __future__ import annotations

import json

import pytest

from deploy import REPO_ROOT
from deploy.cluster import ClusterConfigError, load_cluster


def test_loads_example_inventory():
    cluster = load_cluster(REPO_ROOT / "deploy" / "cluster.example.json")
    assert cluster.coordinator_url.startswith("http://")
    assert cluster.coordinator.port > 0
    assert len(cluster.workers) >= 1
    assert "kimi-api" in cluster.model_groups()
    assert cluster.worker(cluster.workers[0].name).model_group


def test_missing_coordinator_raises(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"workers": []}), encoding="utf-8")
    with pytest.raises(ClusterConfigError):
        load_cluster(p)


def test_missing_worker_field_raises(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(
        json.dumps(
            {
                "coordinator": {"host": "h", "port": 8765, "artifacts_root": "a", "run_set_id": "r"},
                "workers": [{"name": "w1", "host": "h2", "hardware_profile": "local-gpu"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ClusterConfigError):
        load_cluster(p)


def test_bad_json_raises(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(ClusterConfigError):
        load_cluster(p)


def test_duplicate_names_raise(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(
        json.dumps(
            {
                "coordinator": {"host": "h", "port": 1, "artifacts_root": "a", "run_set_id": "r"},
                "workers": [{"name": "dup", "host": "h", "model_group": "g", "hardware_profile": "p"}],
                "api_clients": [{"name": "dup", "model_group": "g2"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ClusterConfigError):
        load_cluster(p)


def test_missing_file_raises(tmp_path):
    with pytest.raises(ClusterConfigError):
        load_cluster(tmp_path / "nonexistent.json")
