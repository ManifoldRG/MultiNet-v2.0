"""Bash tests for the phase-parameterized `vllm serve` arg rendering (Qwen two-tier).

lib/vllm_serve_args.sh::vllm_serve_args renders the served-vLLM argument string
from three env knobs. With the env UNSET it must reproduce today's exact
hard-coded arg string (phase 1); with the phase-2 env it must switch the KV knobs.
See docs/qwen-two-tier-rerun-design.md.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Today's hard-coded phase-1 arg string (lib/distributed_start.sh gpu branch,
# verbatim, modulo ordering which is irrelevant to vllm).
PHASE1 = ("--port 8000 --gpu-memory-utilization 0.9 --max-model-len 16384 "
          "--max-num-seqs 64 --dtype bfloat16 --trust-remote-code")

# Every shell script this task touches — must stay syntactically valid.
TOUCHED = [
    "lib/vllm_serve_args.sh",
    "lib/distributed_start.sh",
    "launch_qwen_smoke.sh",
    "sweep_run.sh",
]


def _render(env: dict | None = None) -> str:
    e = os.environ.copy()
    for k in ("QWEN_MAX_MODEL_LEN", "QWEN_MAX_NUM_SEQS", "QWEN_GPU_MEMORY_UTILIZATION"):
        e.pop(k, None)
    if env:
        e.update(env)
    r = subprocess.run(
        ["bash", "-c", "source lib/vllm_serve_args.sh; vllm_serve_args"],
        capture_output=True, text=True, cwd=REPO, env=e,
    )
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def test_unset_env_reproduces_todays_exact_string():
    assert _render() == PHASE1


def test_phase2_env_switches_kv_knobs():
    out = _render({"QWEN_MAX_MODEL_LEN": "96000", "QWEN_MAX_NUM_SEQS": "3"})
    assert "--max-model-len 96000" in out
    assert "--max-num-seqs 3" in out
    # the non-phase knobs are untouched
    assert "--dtype bfloat16" in out
    assert "--trust-remote-code" in out
    assert "--gpu-memory-utilization 0.9" in out


def test_gpu_memory_utilization_knob():
    out = _render({"QWEN_GPU_MEMORY_UTILIZATION": "0.75"})
    assert "--gpu-memory-utilization 0.75" in out


def test_touched_scripts_pass_bash_n():
    for rel in TOUCHED:
        r = subprocess.run(["bash", "-n", rel], capture_output=True, text=True, cwd=REPO)
        assert r.returncode == 0, f"{rel}: {r.stderr}"


def test_reload_and_subcommand_wiring_present():
    """reload_gpu_worker exists and stops before it relaunches; sweep exposes the
    subcommand and gates it on BATCH_CAP."""
    r = subprocess.run(
        ["bash", "-c", "source lib/distributed_start.sh 2>/dev/null; type reload_gpu_worker"],
        capture_output=True, text=True, cwd=REPO,
    )
    assert r.returncode == 0, r.stderr
    # Reload ordering: stop_gpu_worker → pkill+bounded-wait (race fix) → start_worker.
    body = r.stdout
    assert "stop_gpu_worker" in body and "start_worker" in body
    assert "pkill -f 'vllm serve'" in body, "reload must kill a lingering serve parent"
    assert "pgrep -f 'vllm serve'" in body, "reload must bounded-wait for the parent to clear"
    i_stop = body.index("stop_gpu_worker")
    i_kill = body.index("pkill -f 'vllm serve'")
    i_start = body.index("start_worker")
    assert i_stop < i_kill < i_start, "pkill+wait must sit between stop and relaunch"

    # sweep subcommand dispatch present.
    disp = subprocess.run(
        ["bash", "-c", "grep -n 'reload-qwen-phase2' sweep_run.sh"],
        capture_output=True, text=True, cwd=REPO,
    )
    assert disp.returncode == 0 and "reload-qwen-phase2" in disp.stdout


def test_api_worker_role_override_wiring():
    """start_worker's API branch honors API_WORKER_ROLE (worker|lockstep-worker),
    rejects other values, and forwards API_WORKER_CONCURRENCY as
    --worker-concurrency (MAX_BATCHES for the lockstep runner)."""
    r = subprocess.run(
        ["bash", "-c", "source lib/distributed_start.sh 2>/dev/null; type start_worker"],
        capture_output=True, text=True, cwd=REPO,
    )
    assert r.returncode == 0, r.stderr
    body = r.stdout
    assert 'API_WORKER_ROLE:-worker' in body, "role must default to today's serial worker"
    assert "lockstep-worker" in body, "lockstep role must be selectable on API VMs"
    assert "invalid API_WORKER_ROLE" in body, "unknown roles must fail closed"
    assert 'API_WORKER_CONCURRENCY:-1' in body, "concurrency must default to 1 (serial parity)"
    # The remote command uses the forwarded role/concurrency, not a hardcoded
    # role (the \$ is the heredoc escape as stored in the function body).
    assert '--distributed-role "\\$API_ROLE"' in body
    assert '--worker-concurrency "\\$API_CONC"' in body
