"""Fleet VMs for 3D runs (next-run-fixes item 20): a run-config whose render
block names mujoco3d gets MuJoCo + OSMesa on every VM, one mujoco version
everywhere (it is in every 3D unit's hash), before any coordinator or worker
starts. All cloud and git calls are stubbed."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def bash(snippet: str, env: dict | None = None) -> subprocess.CompletedProcess:
    e = os.environ.copy()
    e.pop("MAX_RUN_DURATION", None)
    if env:
        e.update(env)
    return subprocess.run(["bash", "-c", snippet], capture_output=True, text=True, cwd=REPO, env=e)


def _run_config(tmp_path: Path, render: dict | None) -> str:
    cfg = {"models": {"q": {"provider": "qwen_vllm", "hardware_profile": "local-gpu",
                            "group": "qwen36-27b", "worker_count": 1}}}
    if render is not None:
        cfg["render"] = render
    p = tmp_path / ("rc_3d.json" if render else "rc_2d.json")
    p.write_text(json.dumps(cfg))
    return str(p)


def test_setup_script_syntax_ok():
    r = bash("bash -n deploy/setup_mujoco3d.sh")
    assert r.returncode == 0, r.stderr


def test_needs_mujoco3d_reads_the_render_backend(tmp_path):
    three_d = _run_config(tmp_path, {"backend": "mujoco3d", "camera": "top_down"})
    two_d = _run_config(tmp_path, None)
    r = bash(f"source ./launch_distributed.sh; needs_mujoco3d {three_d} && echo 3D; "
             f"needs_mujoco3d {two_d} || echo 2D")
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["3D", "2D"]


def _version_gcloud(tmp_path: Path) -> Path:
    """Fake gcloud: `compute ssh <vm>` runs the setup and prints the vm's version
    ($VERSION_<vm> with '-' as '_', default 3.13.0); every call is logged."""
    log = tmp_path / "g.log"
    fake = tmp_path / "gcloud"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        'echo "GCLOUD $*" >> "$GCLOUD_LOG"\n'
        'if [[ "$1 $2" == "compute ssh" ]]; then\n'
        '  var="VERSION_${3//-/_}"; echo "pip noise"; echo "mujoco3d-ok ${!var:-3.13.0}"\n'
        "fi\n"
    )
    fake.chmod(0o755)
    return log


def _setup(tmp_path: Path, **versions: str) -> tuple[subprocess.CompletedProcess, str]:
    log = _version_gcloud(tmp_path)
    env = {"PATH": f"{tmp_path}:{os.environ['PATH']}", "GCLOUD_LOG": str(log), **versions}
    r = bash("source ./launch_distributed.sh; GPU_VMS=(r-q-0); "
             "setup_mujoco3d_on_vms z1 r-coord r-q-0 r-api-0", env=env)
    return r, log.read_text() if log.exists() else ""


def test_setup_runs_on_every_vm_in_its_own_venv(tmp_path):
    r, log = _setup(tmp_path)
    assert r.returncode == 0, r.stderr
    calls = [line for line in log.splitlines() if "setup_mujoco3d.sh" in line]
    assert len(calls) == 3
    assert "r-coord" in calls[0] and ".venv-multinet" in calls[0]
    assert "r-q-0" in calls[1] and ".venv-qwen-vllm" in calls[1]
    assert "r-api-0" in calls[2] and ".venv-multinet" in calls[2]
    assert "mujoco 3.13.0" in r.stderr


def test_setup_refuses_a_worker_on_another_mujoco_version(tmp_path):
    r, _log = _setup(tmp_path, VERSION_r_api_0="3.14.0")
    assert r.returncode != 0
    assert "r-api-0" in r.stderr and "3.14.0" in r.stderr and "3.13.0" in r.stderr


def test_setup_fails_when_a_vm_prints_no_version(tmp_path):
    log = tmp_path / "g.log"
    fake = tmp_path / "gcloud"
    fake.write_text('#!/usr/bin/env bash\necho "GCLOUD $*" >> "$GCLOUD_LOG"\necho "pip noise"\n')
    fake.chmod(0o755)
    env = {"PATH": f"{tmp_path}:{os.environ['PATH']}", "GCLOUD_LOG": str(log)}
    r = bash("source ./launch_distributed.sh; setup_mujoco3d_on_vms z1 r-coord", env=env)
    assert r.returncode != 0
    assert "r-coord" in r.stderr


def _launch(tmp_path: Path, run_config: str, setup_body: str) -> tuple[subprocess.CompletedProcess, Path]:
    gcloud = tmp_path / "gcloud"
    gcloud.write_text('#!/usr/bin/env bash\necho "GCLOUD $*"\n')
    gcloud.chmod(0o755)
    git = tmp_path / "git"
    git.write_text('#!/usr/bin/env bash\ncase "$1" in diff) exit 0;; rev-parse) echo SHA;; *) exit 0;; esac\n')
    git.chmod(0o755)
    calls = tmp_path / "calls.log"
    env = {"PATH": f"{tmp_path}:{os.environ['PATH']}", "MAX_RUN_DURATION": "6h", "RUN_ID": "r3",
           "RUNS_DIR": str(tmp_path / ".runs"), "RUN_CONFIG": run_config, "MANIFEST": "m",
           "ZONES": "zoneA", "CALLS": str(calls)}
    snippet = (
        "source ./launch_distributed.sh; "
        "hunt_zones() { echo zoneA; }; wait_for_ssh() { return 0; }; "
        "sync_and_verify() { echo SYNC >> \"$CALLS\"; }; arm_watchdog() { return 0; }; "
        "assert_no_resource_policy() { return 0; }; internal_ip() { echo 10.0.0.2; }; "
        f"setup_mujoco3d_on_vms() {{ echo \"SETUP $*\" >> \"$CALLS\"; {setup_body} }}; "
        "derive_names; start_coordinator() { echo START >> \"$CALLS\"; }; "
        "start_worker() { echo WORKER >> \"$CALLS\"; }; main"
    )
    return bash(snippet, env=env), calls


def test_3d_launch_sets_up_every_vm_after_code_sync_and_before_any_start(tmp_path):
    rc = _run_config(tmp_path, {"backend": "mujoco3d", "camera": "first_person"})
    r, calls = _launch(tmp_path, rc, "return 0;")
    assert r.returncode == 0, r.stderr
    lines = calls.read_text().splitlines()
    assert lines[0] == "SYNC" and lines[1].startswith("SETUP zoneA r3-coord ") and lines[2] == "START"


def test_2d_launch_skips_the_3d_setup(tmp_path):
    r, calls = _launch(tmp_path, _run_config(tmp_path, None), "return 0;")
    assert r.returncode == 0, r.stderr
    assert not any(line.startswith("SETUP") for line in calls.read_text().splitlines())


def test_a_failed_3d_setup_stops_the_fleet_before_any_start(tmp_path):
    rc = _run_config(tmp_path, {"backend": "mujoco3d", "camera": "first_person"})
    r, calls = _launch(tmp_path, rc, "return 1;")
    assert r.returncode != 0
    assert "instances stop" in r.stdout and "instances delete" not in r.stdout
    assert "START" not in calls.read_text()
