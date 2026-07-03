"""Bash tests for lib/distributed_start.sh (coordinator/worker start recipes).
All cloud is stubbed: a fake `gcloud` on PATH records --command args and the
heredoc stdin. No real cloud or API calls."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def bash(snippet: str, env: dict | None = None) -> subprocess.CompletedProcess:
    e = os.environ.copy()
    if env:
        e.update(env)
    return subprocess.run(["bash", "-c", snippet], capture_output=True, text=True, cwd=REPO, env=e)


def _fake_gcloud_capture(tmp_path: Path) -> None:
    """Records `ARGS: ...` (all argv) and the heredoc stdin between STDIN markers."""
    fake = tmp_path / "gcloud"
    fake.write_text(
        '#!/usr/bin/env bash\n'
        'echo "ARGS: $*" >> "$GCLOUD_LOG"\n'
        'echo "STDIN_BEGIN" >> "$GCLOUD_LOG"\n'
        'cat >> "$GCLOUD_LOG"\n'
        'echo "STDIN_END" >> "$GCLOUD_LOG"\n'
        'exit 0\n'
    )
    fake.chmod(0o755)


def test_coordinator_prepares_serves_healthchecks(tmp_path):
    _fake_gcloud_capture(tmp_path)
    glog = tmp_path / "g.log"; glog.write_text("")
    snippet = (
        'source ./lib/distributed_start.sh; '
        'COORD=r1-coord; ZONE=z1; RUN_ID=r1; '
        'RUN_CONFIG=gridworld/fixtures/run_config.smoke_claude_sonnet.json; '
        'MANIFEST=gridworld/fixtures/manifest.smoke_eval.json; '
        'start_coordinator'
    )
    r = bash(snippet, env={"PATH": f"{tmp_path}:{os.environ['PATH']}", "GCLOUD_LOG": str(glog)})
    assert r.returncode == 0, r.stderr
    log = glog.read_text()
    # env injected on the --command; recipe body in the heredoc stdin
    assert "RUN_ID='r1'" in log
    assert "coordinator-prepare" in log
    assert "--run-config" in log and "run_config.smoke_claude_sonnet.json" in log
    assert "--manifest" in log and "manifest.smoke_eval.json" in log
    assert "--seeds" in log and "--difficulty-max-static-score" in log
    # the quoted heredoc keeps $RUN_ID literal (expanded remotely from the injected env)
    assert "--run-set-id" in log and "artifacts/$RUN_ID" in log
    assert "coordinator-serve" in log and "--host 0.0.0.0 --port 8765" in log
    # coordinator-serve.log path is load-bearing (Spec-2 supervisor snapshot_logs expects it)
    assert "coordinator-serve.log" in log
    assert "127.0.0.1:8765/status" in log


_CLAUDE_TOPO = ('{"workers":[{"name":"r1-claude-api-0","kind":"api",'
                '"model_group":"claude-api","provider":"claude","model":"claude-sonnet-4-6"}]}')
_KIMI_TOPO = ('{"workers":[{"name":"r1-kimi-api-0","kind":"api",'
              '"model_group":"kimi-api","provider":"kimi","model":"kimi-k2.6"}]}')
_QWEN_TOPO = ('{"workers":[{"name":"r1-qwen36-27b-0","kind":"gpu",'
              '"model_group":"qwen36-27b","provider":"qwen","model":"Qwen/Qwen3.6-27B"}]}')


def test_worker_field_reads_topo(tmp_path):
    snippet = (f"source ./lib/distributed_start.sh; TOPO_JSON='{_CLAUDE_TOPO}'; "
               'echo "K=$(worker_field r1-claude-api-0 kind) '
               'G=$(worker_field r1-claude-api-0 model_group) '
               'P=$(worker_field r1-claude-api-0 provider)"')
    r = bash(snippet)
    assert "K=api" in r.stdout and "G=claude-api" in r.stdout and "P=claude" in r.stdout


def test_api_worker_claude_exports_anthropic_key(tmp_path):
    _fake_gcloud_capture(tmp_path)
    glog = tmp_path / "g.log"; glog.write_text("")
    snippet = (f"source ./lib/distributed_start.sh; ZONE=z1; RUN_ID=r1; TOPO_JSON='{_CLAUDE_TOPO}'; "
               "start_worker r1-claude-api-0 10.0.0.2")
    r = bash(snippet, env={"PATH": f"{tmp_path}:{os.environ['PATH']}",
                           "GCLOUD_LOG": str(glog), "ANTHROPIC_API_KEY": "sk-ant-DUMMY"})
    assert r.returncode == 0, r.stderr
    log = glog.read_text()
    assert "--hardware-profile api-client" in log
    assert "--model-group" in log and "claude-api" in log
    # the unquoted heredoc keeps \$COORD_IP literal; the IP arrives via the injected env
    assert "COORD_IP='10.0.0.2'" in log and "coordinator-url" in log
    assert ".venv-multinet" in log
    # SECURITY (load-bearing): the key is delivered ONLY via the remote heredoc stdin —
    # never on the --command argv (which leaks to `ps`/gcloud logs) or the launcher stdout.
    args_line = next(line for line in log.splitlines() if line.startswith("ARGS:"))
    assert "sk-ant-DUMMY" not in args_line               # not in --command argv
    stdin = log.split("STDIN_BEGIN", 1)[1].split("STDIN_END", 1)[0]
    assert "export ANTHROPIC_API_KEY=sk-ant-DUMMY" in stdin   # delivered via remote stdin
    assert "sk-ant-DUMMY" not in r.stdout                # launcher never echoes it


def test_api_worker_kimi_exports_moonshot_key(tmp_path):
    _fake_gcloud_capture(tmp_path)
    glog = tmp_path / "g.log"; glog.write_text("")
    snippet = (f"source ./lib/distributed_start.sh; ZONE=z1; RUN_ID=r1; TOPO_JSON='{_KIMI_TOPO}'; "
               "start_worker r1-kimi-api-0 10.0.0.2")
    r = bash(snippet, env={"PATH": f"{tmp_path}:{os.environ['PATH']}",
                           "GCLOUD_LOG": str(glog), "MOONSHOT_API_KEY": "sk-moon-DUMMY"})
    assert r.returncode == 0, r.stderr
    assert "export MOONSHOT_API_KEY=sk-moon-DUMMY" in glog.read_text()


def test_gpu_worker_uses_vllm_venv_and_offline(tmp_path):
    _fake_gcloud_capture(tmp_path)
    glog = tmp_path / "g.log"; glog.write_text("")
    snippet = (f"source ./lib/distributed_start.sh; ZONE=z1; RUN_ID=r1; TOPO_JSON='{_QWEN_TOPO}'; "
               "start_worker r1-qwen36-27b-0 10.0.0.2")
    r = bash(snippet, env={"PATH": f"{tmp_path}:{os.environ['PATH']}", "GCLOUD_LOG": str(glog)})
    assert r.returncode == 0, r.stderr
    log = glog.read_text()
    assert ".venv-qwen-vllm" in log
    assert "HF_HUB_OFFLINE=1" in log
    assert "--hardware-profile local-gpu" in log
    assert "--local-model-cache" in log and "Qwen/Qwen3.6-27B" in log


def test_coordinator_passes_conditions_and_prompt_variant(tmp_path):
    """The conditional sweep needs --conditions (and, for dedup batches,
    --prompt-variant) or run_pipeline's H1/H2 guard rejects the prepare. The
    values are injected on the --command env and consumed by guarded appends in
    the remote heredoc."""
    _fake_gcloud_capture(tmp_path)
    glog = tmp_path / "g.log"; glog.write_text("")
    snippet = (
        'source ./lib/distributed_start.sh; '
        'COORD=r1-coord; ZONE=z1; RUN_ID=r1; '
        'RUN_CONFIG=gridworld/fixtures/run_config.conditional_prompt_claude_kimi_qwen.json; '
        'MANIFEST=gridworld/fixtures/manifest.conditional_eval.json; '
        'start_coordinator'
    )
    r = bash(snippet, env={"PATH": f"{tmp_path}:{os.environ['PATH']}", "GCLOUD_LOG": str(glog),
                           "CONDITIONS": "Prompt", "PROMPT_VARIANT": "minimal"})
    assert r.returncode == 0, r.stderr
    log = glog.read_text()
    # values plumbed through the --command env (load-bearing)
    assert "CONDITIONS='Prompt'" in log
    assert "PROMPT_VARIANT='minimal'" in log
    # guarded appends present in the remote recipe
    assert "--conditions" in log and "--prompt-variant" in log
    assert "coordinator-prepare" in log


def test_coordinator_injects_empty_conditions_when_unset(tmp_path):
    """A non-conditional run (no CONDITIONS/PROMPT_VARIANT) must inject empty
    values so the guarded appends skip both flags — preserving the plain
    coordinator-prepare invocation."""
    _fake_gcloud_capture(tmp_path)
    glog = tmp_path / "g.log"; glog.write_text("")
    snippet = (
        'source ./lib/distributed_start.sh; '
        'COORD=r1-coord; ZONE=z1; RUN_ID=r1; '
        'RUN_CONFIG=gridworld/fixtures/run_config.smoke_claude_sonnet.json; '
        'MANIFEST=gridworld/fixtures/manifest.smoke_eval.json; '
        'start_coordinator'
    )
    r = bash(snippet, env={"PATH": f"{tmp_path}:{os.environ['PATH']}", "GCLOUD_LOG": str(glog)})
    assert r.returncode == 0, r.stderr
    log = glog.read_text()
    assert "CONDITIONS=''" in log
    assert "PROMPT_VARIANT=''" in log


def test_api_worker_unknown_provider_fails(tmp_path):
    _fake_gcloud_capture(tmp_path)
    bad = ('{"workers":[{"name":"r1-x-0","kind":"api","model_group":"x-api",'
           '"provider":"mystery","model":"m"}]}')
    snippet = (f"source ./lib/distributed_start.sh; ZONE=z1; RUN_ID=r1; TOPO_JSON='{bad}'; "
               "start_worker r1-x-0 10.0.0.2")
    r = bash(snippet, env={"PATH": f"{tmp_path}:{os.environ['PATH']}", "GCLOUD_LOG": str(tmp_path / 'g.log')})
    assert r.returncode != 0
    assert "no credential mapping" in r.stderr
