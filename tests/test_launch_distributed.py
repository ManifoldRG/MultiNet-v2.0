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


def test_verify_aborts_on_content_hash_mismatch(tmp_path):
    # git: archive prints nothing; show prints fixed content so local sha256 is deterministic.
    gitstub = tmp_path / "git"
    gitstub.write_text(
        '#!/usr/bin/env bash\n'
        'case "$1" in\n'
        '  archive) printf "";;\n'
        '  show) printf "REALCONTENT";;\n'
        '  *) exit 0;;\n'
        'esac\n'
    )
    gitstub.chmod(0o755)
    # gcloud ssh: deployed_sha returns the CORRECT sha (first check passes);
    # sha256sum returns a hash that will NOT equal sha256("REALCONTENT").
    gcloudstub = tmp_path / "gcloud"
    gcloudstub.write_text(
        '#!/usr/bin/env bash\n'
        'for a in "$@"; do last="$a"; done\n'
        'case "$last" in\n'
        '  *deployed_sha*) echo "TARGETSHA";;\n'
        '  *sha256sum*) echo "deadbeef  scripts/distributed_run_pipeline.py";;\n'
        '  *) :;;\n'
        'esac\n'
        'exit 0\n'
    )
    gcloudstub.chmod(0o755)
    env = {"PATH": f"{tmp_path}:{os.environ['PATH']}"}
    r = bash("source ./launch_distributed.sh; sync_and_verify TARGETSHA zoneA vm0", env=env)
    assert r.returncode != 0
    assert "content mismatch" in (r.stderr + r.stdout).lower()


def test_require_clean_tree_blocks_staged_dirty(tmp_path):
    # Stub git: unstaged diff (no --cached) → exit 0 (clean); staged diff (--cached) → exit 1 (dirty).
    gitstub = tmp_path / "git"
    gitstub.write_text(
        '#!/usr/bin/env bash\n'
        'if [[ "$1" == "diff" ]]; then for a in "$@"; do [[ "$a" == "--cached" ]] && exit 1; done; exit 0; fi\n'
        'exit 0\n'
    )
    gitstub.chmod(0o755)
    env = {"PATH": f"{tmp_path}:{os.environ['PATH']}"}
    r = bash("source ./launch_distributed.sh; require_clean_tree", env=env)
    assert r.returncode != 0


def _gpu_only_config(tmp_path: Path) -> str:
    """A self-contained GPU-only run_config: a local-gpu model (→ hunt path) and
    no API model (→ no credential gate). Returns the path."""
    import json
    p = tmp_path / "rc_gpu.json"
    p.write_text(json.dumps({"models": {
        "q": {"provider": "qwen_vllm", "hardware_profile": "local-gpu",
              "group": "qwen36-27b", "worker_count": 2}}}))
    return str(p)


def test_post_start_failure_stops_not_deletes(tmp_path):
    _fake_gcloud(tmp_path)   # all gcloud calls succeed (echo only)
    # git stub: clean tree + archive/show succeed.
    gitstub = tmp_path / "git"
    gitstub.write_text(
        '#!/usr/bin/env bash\n'
        'case "$1" in diff) exit 0;; rev-parse) echo SHA;; archive) printf "";; '
        'show) printf "B";; *) exit 0;; esac\n'
    )
    gitstub.chmod(0o755)
    env = {"PATH": f"{tmp_path}:{os.environ['PATH']}", "MAX_RUN_DURATION": "6h",
           "RUN_ID": "rx", "RUNS_DIR": str(tmp_path / ".runs"),
           "RUN_CONFIG": _gpu_only_config(tmp_path), "MANIFEST": "dummy-manifest",
           "ZONES": "zoneA", "GCLOUD_LOG": str(tmp_path / "g.log")}
    # Override the heavy/cloud bits; force start_coordinator to FAIL after create.
    snippet = (
        "source ./launch_distributed.sh; "
        "hunt_zones() { echo zoneA; }; "
        "wait_for_ssh() { return 0; }; "
        "sync_and_verify() { return 0; }; "
        "arm_watchdog() { return 0; }; "
        "assert_no_resource_policy() { return 0; }; "
        "internal_ip() { echo 10.0.0.2; }; "
        "derive_names; "
        "start_coordinator() { echo 'BOOM' >&2; return 1; }; "
        "main"
    )
    r = bash(snippet, env=env)
    assert r.returncode != 0
    assert "instances stop" in r.stdout            # STOP on the failure path
    assert "instances delete" not in r.stdout      # never DELETE a possibly-data VM


def test_successful_provision_writes_manifest(tmp_path):
    _fake_gcloud(tmp_path)
    gitstub = tmp_path / "git"
    gitstub.write_text(
        '#!/usr/bin/env bash\n'
        'case "$1" in diff) exit 0;; rev-parse) echo SHA123;; *) exit 0;; esac\n'
    )
    gitstub.chmod(0o755)
    runs = tmp_path / ".runs"
    env = {"PATH": f"{tmp_path}:{os.environ['PATH']}", "MAX_RUN_DURATION": "6h",
           "RUN_ID": "rok", "RUNS_DIR": str(runs),
           "RUN_CONFIG": _gpu_only_config(tmp_path), "MANIFEST": "dummy-manifest",
           "ZONES": "zoneA"}
    snippet = (
        "source ./launch_distributed.sh; "
        "hunt_zones() { echo zoneA; }; wait_for_ssh() { return 0; }; "
        "sync_and_verify() { return 0; }; arm_watchdog() { return 0; }; "
        "assert_no_resource_policy() { return 0; }; internal_ip() { echo 10.0.0.2; }; "
        "derive_names; start_coordinator() { return 0; }; start_worker() { return 0; }; main"
    )
    r = bash(snippet, env=env)
    assert r.returncode == 0, r.stderr
    import json
    mf = json.loads((runs / "rok" / "manifest.json").read_text())
    assert mf["run_id"] == "rok" and mf["zone"] == "zoneA"
    assert mf["code_sha"] == "SHA123"
    assert mf["coordinator"]["name"] == "rok-coord"
    assert {w["kind"] for w in mf["workers"]} <= {"gpu", "api"}
    assert "internal_ip" in mf["coordinator"]


def test_hunt_zone_capture_is_clean_zone_only(tmp_path):
    """ZONE capture must contain only the winning zone name — no gcloud stdout noise."""
    fake = tmp_path / "gcloud"
    fake.write_text('#!/usr/bin/env bash\n[[ "$3" == "create" ]] && echo "Created [https://example/vm]."\nexit 0\n')
    fake.chmod(0o755)
    env = {"PATH": f"{tmp_path}:{os.environ['PATH']}", "MAX_RUN_DURATION": "6h"}
    r = bash('source ./launch_distributed.sh; ZONES=zoneA; '
             'z="$(hunt_zones coord g0)"; printf "CAPTURED=[%s]\\n" "$z"', env=env)
    assert r.returncode == 0, r.stderr
    assert "CAPTURED=[zoneA]" in r.stdout   # exactly the zone — no gcloud/log noise captured


def test_gpu_only_no_empty_vm_name(tmp_path):
    """ALL_VMS must contain no empty strings in a GPU-only run (no API VMs).

    wait_for_ssh rejects an empty first argument; the provision must still
    succeed (returncode 0) and NAME_EMPTY must never appear in output.
    Before the guarded-append fix this fails: the empty API_VMS expansion
    inserts a blank element and wait_for_ssh emits NAME_EMPTY → abort."""
    _fake_gcloud(tmp_path)
    gitstub = tmp_path / "git"
    gitstub.write_text(
        '#!/usr/bin/env bash\n'
        'case "$1" in diff) exit 0;; rev-parse) echo SHA456;; *) exit 0;; esac\n'
    )
    gitstub.chmod(0o755)
    runs = tmp_path / ".runs"
    env = {"PATH": f"{tmp_path}:{os.environ['PATH']}", "MAX_RUN_DURATION": "6h",
           "RUN_ID": "rgpu", "RUNS_DIR": str(runs),
           "RUN_CONFIG": _gpu_only_config(tmp_path), "MANIFEST": "dummy-manifest",
           "ZONES": "zoneA"}
    snippet = (
        "source ./launch_distributed.sh; "
        "hunt_zones() { echo zoneA; }; "
        "wait_for_ssh() { [[ -n \"$1\" ]] || { echo NAME_EMPTY >&2; return 1; }; return 0; }; "
        "sync_and_verify() { return 0; }; "
        "arm_watchdog() { return 0; }; "
        "assert_no_resource_policy() { return 0; }; "
        "internal_ip() { echo 10.0.0.2; }; "
        "start_coordinator() { return 0; }; "
        "start_worker() { return 0; }; "
        "derive_names; main"
    )
    r = bash(snippet, env=env)
    combined = r.stdout + r.stderr
    assert "NAME_EMPTY" not in combined, f"empty VM name reached wait_for_ssh:\n{combined}"
    assert r.returncode == 0, f"provision failed (expected success):\n{r.stderr}"


def test_launch_sources_real_start_hooks():
    # After sourcing launch_distributed.sh, start_coordinator must be the REAL recipe
    # (coordinator-serve), not the old `ssh ... true` stub.
    r = bash("source ./launch_distributed.sh 2>/dev/null; type start_coordinator")
    assert r.returncode == 0, r.stderr
    assert "coordinator-serve" in r.stdout
    assert "start_worker" in bash(
        "source ./launch_distributed.sh 2>/dev/null; type start_worker").stdout
    # the stub returned true with no real command; the real one dispatches on kind
    assert "hardware-profile" in bash(
        "source ./launch_distributed.sh 2>/dev/null; type start_worker").stdout


def test_sync_ships_and_verifies_submodule_content(tmp_path):
    # git archive skips submodules; sync_code_to_vm must also archive each
    # submodule (enumerated via ls-tree commit entries) at its pinned sha, and
    # verify_code_on_vm must confirm the submodule tree landed on the VM.
    gclog = tmp_path / "gclog"
    gitstub = tmp_path / "git"
    gitstub.write_text(
        '#!/usr/bin/env bash\n'
        'case "$1" in\n'
        '  archive) printf "SUPERARCHIVE";;\n'
        '  show) printf "PIPEBYTES";;\n'                      # $sha:scripts/distributed_run_pipeline.py
        '  ls-tree) printf "160000 commit deadbeefcafe\\togbench\\n";;\n'
        '  rev-parse) printf "subsha123";;\n'                 # $sha:ogbench
        '  diff) exit 0;;\n'
        '  -C) case "$3" in archive) printf "SUBARCHIVE";; *) exit 0;; esac;;\n'
        '  *) exit 0;;\n'
        'esac\n'
    )
    gitstub.chmod(0o755)
    gcloudstub = tmp_path / "gcloud"
    gcloudstub.write_text(
        '#!/usr/bin/env bash\n'
        'cat > /dev/null\n'  # drain stdin so the piped git-archive never SIGPIPEs
        'cmd=""; prev=""\n'
        'for a in "$@"; do [[ "$prev" == "--command" ]] && cmd="$a"; prev="$a"; done\n'
        'echo "$cmd" >> "$GCLOG"\n'
        'case "$cmd" in\n'
        '  *deployed_sha*) echo "TARGETSHA";;\n'
        '  *sha256sum*distributed_run_pipeline.py*)\n'
        '     echo "$(printf PIPEBYTES | sha256sum | awk "{print \\$1}")  x";;\n'
        '  *) : ;;\n'                                          # tar extract + find presence -> exit 0
        'esac\n'
        'exit 0\n'
    )
    gcloudstub.chmod(0o755)
    env = {"PATH": f"{tmp_path}:{os.environ['PATH']}", "GCLOG": str(gclog)}
    r = bash("source ./launch_distributed.sh; sync_and_verify TARGETSHA zoneA vm0", env=env)
    assert r.returncode == 0, (r.stderr + r.stdout)
    log = gclog.read_text()
    # submodule tree shipped to the VM's ogbench path...
    assert "tar -x -C ~/MultiNet-v2.0/ogbench" in log, log
    # ...and its presence fail-closed-verified.
    assert "find ~/MultiNet-v2.0/ogbench" in log, log


def test_sync_aborts_when_submodule_missing_on_vm(tmp_path):
    # If the submodule tree did not land, verify_code_on_vm must fail closed.
    gitstub = tmp_path / "git"
    gitstub.write_text(
        '#!/usr/bin/env bash\n'
        'case "$1" in\n'
        '  archive) printf "SUPERARCHIVE";;\n'
        '  show) printf "PIPEBYTES";;\n'
        '  ls-tree) printf "160000 commit deadbeefcafe\\togbench\\n";;\n'
        '  rev-parse) printf "subsha123";;\n'
        '  diff) exit 0;;\n'
        '  -C) case "$3" in archive) printf "SUBARCHIVE";; *) exit 0;; esac;;\n'
        '  *) exit 0;;\n'
        'esac\n'
    )
    gitstub.chmod(0o755)
    gcloudstub = tmp_path / "gcloud"
    gcloudstub.write_text(
        '#!/usr/bin/env bash\n'
        'cat > /dev/null\n'  # drain stdin so the piped git-archive never SIGPIPEs
        'cmd=""; prev=""\n'
        'for a in "$@"; do [[ "$prev" == "--command" ]] && cmd="$a"; prev="$a"; done\n'
        'case "$cmd" in\n'
        '  *deployed_sha*) echo "TARGETSHA"; exit 0;;\n'
        '  *sha256sum*distributed_run_pipeline.py*)\n'
        '     echo "$(printf PIPEBYTES | sha256sum | awk "{print \\$1}")  x"; exit 0;;\n'
        '  *find*MultiNet-v2.0/ogbench*) exit 1;;\n'           # submodule dir empty/missing
        '  *) exit 0;;\n'
        'esac\n'
    )
    gcloudstub.chmod(0o755)
    env = {"PATH": f"{tmp_path}:{os.environ['PATH']}"}
    r = bash("source ./launch_distributed.sh; sync_and_verify TARGETSHA zoneA vm0", env=env)
    assert r.returncode != 0
    assert "submodule" in (r.stderr + r.stdout).lower()
