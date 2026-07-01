from __future__ import annotations

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
