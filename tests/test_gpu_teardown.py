"""Unit tests for lib/gpu_teardown.sh::gpu_teardown_local.

The load-bearing invariant: after SIGTERM'ing the worker + GPU compute-apps, the
function polls `nvidia-smi` until the GPU reports NO compute-apps and returns 0;
if the GPU never frees within the timeout it returns nonzero (FAIL-CLOSED) and
does NOT escalate to SIGKILL (SIGKILL wedges CUDA). `nvidia-smi`/`pkill` are
shimmed onto PATH so this runs on a GPU-less CI box.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "lib" / "gpu_teardown.sh"


def _bin(tmp_path):
    b = tmp_path / "bin"
    b.mkdir()
    # pkill + external kill (used via `xargs kill`) always succeed
    for name in ("pkill", "kill"):
        p = b / name
        p.write_text("#!/usr/bin/env bash\nexit 0\n")
        p.chmod(0o755)
    return b


def _run(tmp_path, nvidia_smi_body: str, timeout=2, interval=1):
    b = _bin(tmp_path)
    ns = b / "nvidia-smi"
    ns.write_text("#!/usr/bin/env bash\n" + nvidia_smi_body)
    ns.chmod(0o755)
    cmd = f'source "{SCRIPT}"; gpu_teardown_local'
    return subprocess.run(
        ["bash", "-c", cmd],
        env={"PATH": f"{b}:/usr/bin:/bin",
             "GPU_TEARDOWN_TIMEOUT": str(timeout),
             "GPU_TEARDOWN_INTERVAL": str(interval)},
        capture_output=True, text=True, timeout=30)


def test_returns_zero_when_gpu_frees_after_kill(tmp_path):
    # first call lists a stuck pid; subsequent calls report the GPU free
    counter = tmp_path / "count"
    body = (
        f'c="{counter}"; n=$(cat "$c" 2>/dev/null || echo 0); echo $((n+1)) > "$c"\n'
        'if [ "$n" -eq 0 ]; then echo "1234"; fi\n')
    r = _run(tmp_path, body)
    assert r.returncode == 0, r.stderr


def test_fail_closed_when_gpu_never_frees(tmp_path):
    # nvidia-smi always reports a compute-app -> never frees -> nonzero, no hang
    r = _run(tmp_path, 'echo "1234"\n', timeout=2, interval=1)
    assert r.returncode != 0
    assert "still" in r.stderr.lower()
