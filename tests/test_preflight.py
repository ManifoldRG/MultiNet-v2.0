# tests/test_preflight.py
from __future__ import annotations

from deploy.cluster import ApiClient, Cluster, Coordinator, Worker
from deploy.preflight import Probes, run_preflight


def _cluster():
    return Cluster(
        coordinator=Coordinator(host="10.0.0.1", port=8765, artifacts_root="/tmp/art", run_set_id="rs"),
        workers=[Worker(name="qwen-1", host="10.0.0.2", model_group="qwen36-27b", hardware_profile="local-gpu")],
        api_clients=[ApiClient(name="kimi", model_group="kimi-api")],
        storage_config=None,
    )


def _probes(**overrides):
    base = dict(
        gpu=lambda: (True, "NVIDIA A100-SXM4-80GB"),
        vram_free_gb=lambda: 79.0,
        weights_present=lambda g: True,
        kernels=lambda: {"flash_attn": True, "fla": True, "causal_conv1d": True},
        http_ok=lambda url: True,
        port_free=lambda h, p: True,
        gsutil_present=lambda: True,
        keys=lambda: {"anthropic": "a", "kimi": "k"},
    )
    base.update(overrides)
    return Probes(**base)


def test_worker_all_pass():
    results, code = run_preflight(_cluster(), "qwen-1", "worker", _probes())
    assert code == 0
    assert all(r.ok for r in results if r.fatal)


def test_worker_no_gpu_is_fatal():
    results, code = run_preflight(_cluster(), "qwen-1", "worker", _probes(gpu=lambda: (False, "no nvidia-smi")))
    assert code == 1


def test_worker_low_vram_is_fatal():
    results, code = run_preflight(_cluster(), "qwen-1", "worker", _probes(vram_free_gb=lambda: 20.0))
    assert code == 1


def test_worker_missing_weights_is_fatal():
    results, code = run_preflight(_cluster(), "qwen-1", "worker", _probes(weights_present=lambda g: False))
    assert code == 1


def test_worker_missing_kernel_is_warn_only():
    probes = _probes(kernels=lambda: {"flash_attn": False, "fla": True, "causal_conv1d": True})
    results, code = run_preflight(_cluster(), "qwen-1", "worker", probes)
    assert code == 0
    assert any((not r.ok) and (not r.fatal) for r in results)


def test_worker_coordinator_unreachable_is_fatal():
    results, code = run_preflight(_cluster(), "qwen-1", "worker", _probes(http_ok=lambda url: False))
    assert code == 1


def test_coordinator_port_busy_is_warn_only():
    results, code = run_preflight(_cluster(), None, "coordinator", _probes(port_free=lambda h, p: False))
    assert code == 0


def test_api_client_missing_key_is_fatal():
    probes = _probes(keys=lambda: {"anthropic": "a", "kimi": None})
    results, code = run_preflight(_cluster(), None, "api-client", probes)
    assert code == 1
