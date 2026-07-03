from __future__ import annotations

import json
from pathlib import Path

from deploy.cluster import ApiClient, Cluster, Coordinator, Worker
from deploy.print_launch_commands import build_commands, group_mismatches


def _cluster(artifacts_root: str, storage=None):
    return Cluster(
        coordinator=Coordinator(host="10.0.0.1", port=8765, artifacts_root=artifacts_root, run_set_id="rs"),
        workers=[Worker(name="qwen-1", host="10.0.0.2", model_group="qwen36-27b", hardware_profile="local-gpu")],
        api_clients=[ApiClient(name="kimi", model_group="kimi-api")],
        storage_config=storage,
    )


def test_build_commands_wires_coordinator_url_and_flags():
    cmds = build_commands(
        _cluster("artifacts/x", storage="gridworld/fixtures/storage_config.example.json"),
        run_config="rc.json",
        manifest="m.json",
        conditions="Prompt",
        prompt_variant="minimal",
    )
    assert "--distributed-role coordinator-prepare" in cmds["coordinator-prepare"]
    assert '--conditions "Prompt"' in cmds["coordinator-prepare"]
    assert "--prompt-variant minimal" in cmds["coordinator-prepare"]
    assert "--coordinator-url http://10.0.0.1:8765" in cmds["worker:qwen-1"]
    assert "--model-group qwen36-27b" in cmds["worker:qwen-1"]
    assert "coordinator-run-api-client" in cmds["api-client:kimi"]
    assert "--storage-config gridworld/fixtures/storage_config.example.json" in cmds["coordinator-serve"]


def test_group_mismatch_detected_against_plan(tmp_path):
    art = tmp_path / "art"
    (art / "distributed").mkdir(parents=True)
    plan = {"models": {"qwen": {"model_group": "qwen36-27b"}}}  # note: no kimi-api group
    (art / "distributed" / "job_plan.json").write_text(json.dumps(plan), encoding="utf-8")

    warnings = group_mismatches(_cluster(str(art)))
    assert any("kimi-api" in w for w in warnings)


def test_group_mismatch_empty_when_no_plan(tmp_path):
    assert group_mismatches(_cluster(str(tmp_path / "missing"))) == []


def test_build_commands_omits_optional_flags_when_absent():
    cmds = build_commands(_cluster("artifacts/x"), run_config="rc.json", manifest="m.json")
    assert "--conditions" not in cmds["coordinator-prepare"]
    assert "--prompt-variant" not in cmds["coordinator-prepare"]
    assert "--storage-config" not in cmds["coordinator-serve"]
