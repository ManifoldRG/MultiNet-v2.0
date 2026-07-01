"""Bash tests for launch_distributed.sh (provisioner). All cloud/git calls are stubbed."""
from __future__ import annotations

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


def _fake_gcloud(tmp_path: Path) -> None:
    fake = tmp_path / "gcloud"
    fake.write_text('#!/usr/bin/env bash\necho "GCLOUD $*"\n')
    fake.chmod(0o755)


def test_syntax_ok():
    r = bash("bash -n ./launch_distributed.sh")
    assert r.returncode == 0, r.stderr


def test_launch_without_max_run_duration_aborts(tmp_path):
    _fake_gcloud(tmp_path)
    env = {"PATH": f"{tmp_path}:{os.environ['PATH']}",
           "RUN_CONFIG": "gridworld/fixtures/run_config.smoke_eval_qwen_kimi.json",
           "MANIFEST": "gridworld/fixtures/manifest.smoke_eval.json"}
    r = bash("bash ./launch_distributed.sh", env=env)
    assert r.returncode != 0
    assert "MAX_RUN_DURATION" in r.stderr


def test_stop_subcommand_reads_manifest(tmp_path):
    _fake_gcloud(tmp_path)
    manifest_dir = tmp_path / ".runs" / "r1"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text(
        '{"run_id":"r1","zone":"z1","coordinator":{"name":"r1-coord"},'
        '"workers":[{"name":"r1-w-0","kind":"gpu","model_group":"g"}]}'
    )
    env = {"PATH": f"{tmp_path}:{os.environ['PATH']}", "RUN_ID": "r1",
           "RUNS_DIR": str(tmp_path / ".runs")}
    r = bash("bash ./launch_distributed.sh stop", env=env)
    assert r.returncode == 0, r.stderr
    assert "instances stop" in r.stdout
    assert "r1-coord" in r.stdout and "r1-w-0" in r.stdout
    assert "instances delete" not in r.stdout


def _stateful_fake_gcloud(tmp_path: Path, fail_create_in: str) -> None:
    """Fake gcloud that fails `instances create` when --zone == fail_create_in,
    logging every call to $GCLOUD_LOG."""
    fake = tmp_path / "gcloud"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        'echo "GCLOUD $*" >> "$GCLOUD_LOG"\n'
        'if [[ "$1 $2" == "compute instances" && "$3" == "create" ]]; then\n'
        '  for a in "$@"; do [[ "$prev" == "--zone" ]] && z="$a"; prev="$a"; done\n'
        f'  if [[ "$z" == "{fail_create_in}" ]]; then echo "ZONE_RESOURCE_POOL_EXHAUSTED" >&2; exit 1; fi\n'
        "fi\n"
        "exit 0\n"
    )
    fake.chmod(0o755)


def test_hunt_rolls_back_first_zone_and_lands_in_second(tmp_path):
    glog = tmp_path / "g.log"
    _stateful_fake_gcloud(tmp_path, fail_create_in="zoneA")
    snippet = (
        "source ./launch_distributed.sh; "
        "ZONES='zoneA zoneB'; "
        'hunt_zones coord g0 g1'
    )
    env = {"PATH": f"{tmp_path}:{os.environ['PATH']}", "GCLOUD_LOG": str(glog),
           "MAX_RUN_DURATION": "6h", "ZONES": "zoneA zoneB"}
    r = bash(snippet, env=env)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().splitlines()[-1] == "zoneB"   # winning zone echoed last
    log = glog.read_text()
    assert ("instances delete g0 --zone zoneA" in log) or ("instances delete" in log and "zoneA" in log)
    assert "instances create coord --zone zoneB" in log


def test_hunt_all_stocked_out_returns_nonzero_and_leaves_nothing(tmp_path):
    glog = tmp_path / "g.log"
    # Fail creates in BOTH zones by making fail_create_in match a regex-y sentinel:
    fake = tmp_path / "gcloud"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        'echo "GCLOUD $*" >> "$GCLOUD_LOG"\n'
        'if [[ "$3" == "create" ]]; then echo EXHAUSTED >&2; exit 1; fi\n'
        "exit 0\n"
    )
    fake.chmod(0o755)
    env = {"PATH": f"{tmp_path}:{os.environ['PATH']}", "GCLOUD_LOG": str(glog),
           "MAX_RUN_DURATION": "6h"}
    r = bash("source ./launch_distributed.sh; ZONES='zoneA zoneB'; hunt_zones coord g0", env=env)
    assert r.returncode != 0
    log = glog.read_text()
    # Rollback delete attempted in every zone tried.
    assert "instances delete" in log and "zoneA" in log and "zoneB" in log


def test_require_clean_tree_blocks_dirty(tmp_path):
    # Stub git to report a dirty tree (diff returns nonzero).
    gitstub = tmp_path / "git"
    gitstub.write_text('#!/usr/bin/env bash\n[[ "$1" == "diff" ]] && exit 1\nexit 0\n')
    gitstub.chmod(0o755)
    env = {"PATH": f"{tmp_path}:{os.environ['PATH']}"}
    r = bash("source ./launch_distributed.sh; require_clean_tree", env=env)
    assert r.returncode != 0
    # ALLOW_DIRTY overrides.
    r2 = bash("source ./launch_distributed.sh; require_clean_tree",
              env={**env, "ALLOW_DIRTY": "1"})
    assert r2.returncode == 0, r2.stderr


def test_verify_aborts_on_sha_mismatch(tmp_path):
    # git: archive prints nothing; show prints fixed content; rev-parse irrelevant here.
    gitstub = tmp_path / "git"
    gitstub.write_text(
        '#!/usr/bin/env bash\n'
        'case "$1" in\n'
        '  archive) printf "";;\n'
        '  show) printf "PIPELINE_BYTES";;\n'
        '  *) exit 0;;\n'
        'esac\n'
    )
    gitstub.chmod(0o755)
    # gcloud ssh: read-back returns the WRONG sha -> mismatch.
    gcloudstub = tmp_path / "gcloud"
    gcloudstub.write_text(
        '#!/usr/bin/env bash\n'
        'for a in "$@"; do last="$a"; done\n'
        'case "$last" in\n'
        '  *deployed_sha*) echo "WRONGSHA";;\n'
        '  *sha256sum*) echo "deadbeef  scripts/distributed_run_pipeline.py";;\n'
        '  *) :;;\n'
        'esac\n'
        'exit 0\n'
    )
    gcloudstub.chmod(0o755)
    env = {"PATH": f"{tmp_path}:{os.environ['PATH']}"}
    r = bash("source ./launch_distributed.sh; sync_and_verify TARGETSHA zoneA vm0", env=env)
    assert r.returncode != 0
    assert "mismatch" in (r.stderr + r.stdout).lower()
