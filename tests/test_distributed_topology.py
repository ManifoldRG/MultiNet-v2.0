from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.distributed_topology import sanitize_vm_name, derive_topology


def _cfg(models: dict) -> dict:
    return {"description": "t", "models": models}


def test_sanitize_lowercases_and_replaces():
    assert sanitize_vm_name("Run_ID.123") == "run-id-123"


def test_sanitize_strips_and_truncates():
    long = "x" * 80
    out = sanitize_vm_name(long + "_END")
    assert len(out) <= 63
    assert not out.startswith("-") and not out.endswith("-")


def test_gpu_only_topology():
    cfg = _cfg({"qwen36_27b_fp8_vllm": {
        "provider": "qwen_vllm", "hardware_profile": "local-gpu",
        "group": "qwen36-27b", "worker_count": 2,
    }})
    t = derive_topology(cfg, "run1")
    assert t["coordinator"]["name"] == "run1-coord"
    assert [w["name"] for w in t["workers"]] == ["run1-qwen36-27b-0", "run1-qwen36-27b-1"]
    assert all(w["kind"] == "gpu" for w in t["workers"])
    assert t["has_gpu"] is True
    assert t["required_credentials"] == []


def test_mixed_topology_and_credentials():
    cfg = _cfg({
        "qwen": {"provider": "qwen_vllm", "hardware_profile": "local-gpu",
                 "group": "qwen36-27b", "worker_count": 2},
        "kimi": {"provider": "kimi", "hardware_profile": "api-client",
                 "group": "kimi-api", "worker_count": 1},
    })
    t = derive_topology(cfg, "run2")
    kinds = sorted((w["kind"], w["model_group"]) for w in t["workers"])
    assert kinds == [("api", "kimi-api"), ("gpu", "qwen36-27b"), ("gpu", "qwen36-27b")]
    assert t["has_gpu"] is True
    assert t["required_credentials"] == ["MOONSHOT_API_KEY"]


def test_all_api_topology_no_gpu():
    cfg = _cfg({
        "claude": {"provider": "claude", "hardware_profile": "api-client",
                   "group": "claude-api", "worker_count": 1},
        "kimi": {"provider": "kimi", "hardware_profile": "api-client",
                 "group": "kimi-api", "worker_count": 1},
    })
    t = derive_topology(cfg, "run3")
    assert t["has_gpu"] is False
    assert all(w["kind"] == "api" for w in t["workers"])
    assert t["required_credentials"] == ["ANTHROPIC_API_KEY", "MOONSHOT_API_KEY"]


def test_worker_count_defaults_to_one():
    cfg = _cfg({"q": {"provider": "qwen_vllm", "hardware_profile": "local-gpu", "group": "g"}})
    t = derive_topology(cfg, "r")
    assert len(t["workers"]) == 1


def test_derive_topology_raises_on_missing_group():
    cfg = _cfg({"q": {"provider": "qwen_vllm", "hardware_profile": "local-gpu", "worker_count": 1}})
    with pytest.raises(ValueError, match="missing required 'group'"):
        derive_topology(cfg, "r")


REPO = Path(__file__).resolve().parent.parent


def _run_cli(args, env=None, cfg=None, tmp_path=None):
    if cfg is not None:
        p = tmp_path / "rc.json"
        p.write_text(json.dumps(cfg))
        args = [str(p)] + args
    e = os.environ.copy()
    e.pop("MOONSHOT_API_KEY", None)
    e.pop("ANTHROPIC_API_KEY", None)
    if env:
        e.update(env)
    return subprocess.run(
        [sys.executable, "-m", "scripts.distributed_topology", *args],
        capture_output=True, text=True, cwd=REPO, env=e,
    )


def test_cli_prints_topology_json(tmp_path):
    cfg = {"models": {"q": {"provider": "qwen_vllm", "hardware_profile": "local-gpu",
                            "group": "qwen36-27b", "worker_count": 2}}}
    r = _run_cli(["run9"], cfg=cfg, tmp_path=tmp_path)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["coordinator"]["name"] == "run9-coord"
    assert len(out["workers"]) == 2


def test_cli_check_credentials_missing_fails(tmp_path):
    cfg = {"models": {"k": {"provider": "kimi", "hardware_profile": "api-client",
                            "group": "kimi-api", "worker_count": 1}}}
    r = _run_cli(["run10", "--check-credentials"], cfg=cfg, tmp_path=tmp_path)
    assert r.returncode == 3
    assert "MOONSHOT_API_KEY" in r.stderr


def test_cli_check_credentials_present_passes(tmp_path):
    cfg = {"models": {"k": {"provider": "kimi", "hardware_profile": "api-client",
                            "group": "kimi-api", "worker_count": 1}}}
    r = _run_cli(["run11", "--check-credentials"], env={"MOONSHOT_API_KEY": "x"},
                 cfg=cfg, tmp_path=tmp_path)
    assert r.returncode == 0, r.stderr


def test_claude_smoke_fixture_topology():
    import json
    from pathlib import Path
    from scripts.distributed_topology import derive_topology
    cfg = json.loads(Path("gridworld/fixtures/run_config.smoke_claude_sonnet.json").read_text())
    topo = derive_topology(cfg, "claude-smoke")
    assert topo["has_gpu"] is False
    assert topo["required_credentials"] == ["ANTHROPIC_API_KEY"]
    assert len(topo["workers"]) == 1
    w = topo["workers"][0]
    assert w["kind"] == "api" and w["model_group"] == "claude-api" and w["provider"] == "claude"
    assert w["name"] == "claude-smoke-claude-api-0"


def test_worker_carries_model_field():
    from scripts.distributed_topology import derive_topology
    cfg = {"models": {
        "m_gpu": {"provider": "qwen", "model": "Qwen/Qwen3.6-27B",
                  "group": "qwen36-27b", "hardware_profile": "local-gpu", "worker_count": 1},
        "m_api": {"provider": "claude", "model": "claude-sonnet-4-6",
                  "group": "claude-api", "worker_count": 1},
    }}
    topo = derive_topology(cfg, "r")
    by_group = {w["model_group"]: w for w in topo["workers"]}
    assert by_group["qwen36-27b"]["model"] == "Qwen/Qwen3.6-27B"
    assert by_group["claude-api"]["model"] == "claude-sonnet-4-6"


def test_gpu_worker_count_override_multiplies_only_gpu_workers():
    cfg = {"models": {
        "qwen": {"group": "qwen36-27b", "provider": "qwen_vllm",
                 "hardware_profile": "local-gpu", "worker_count": 1},
        "kimi": {"group": "kimi-api", "provider": "kimi", "worker_count": 1},
        "claude": {"group": "opus", "provider": "claude", "worker_count": 1},
    }}
    from scripts.distributed_topology import derive_topology
    topo = derive_topology(cfg, "r1", gpu_worker_count=3)
    gpu = [w for w in topo["workers"] if w["kind"] == "gpu"]
    api = [w for w in topo["workers"] if w["kind"] == "api"]
    assert len(gpu) == 3           # override applied
    assert len(api) == 2           # kimi + claude untouched
    assert {w["name"] for w in gpu} == {"r1-qwen36-27b-0", "r1-qwen36-27b-1", "r1-qwen36-27b-2"}


def test_gpu_worker_count_none_preserves_config_counts():
    cfg = {"models": {"qwen": {"group": "q", "provider": "qwen_vllm",
                               "hardware_profile": "local-gpu", "worker_count": 1}}}
    from scripts.distributed_topology import derive_topology
    assert len([w for w in derive_topology(cfg, "r1")["workers"] if w["kind"] == "gpu"]) == 1


def test_api_only_split_config_has_no_gpu_workers():
    # The SWEEP_TOPO=api split: Kimi+Claude only -> 0 GPU workers, so the launcher
    # skips the A100 hunt and brings up coordinator + 2 e2 API VMs.
    cfg = json.loads(Path("gridworld/fixtures/run_config.conditional_prompt_claude_kimi.json").read_text())
    topo = derive_topology(cfg, "sid")
    assert topo["has_gpu"] is False
    assert all(w["kind"] == "api" for w in topo["workers"])
    assert len(topo["workers"]) == 2
    assert set(topo["required_credentials"]) == {"ANTHROPIC_API_KEY", "MOONSHOT_API_KEY"}
