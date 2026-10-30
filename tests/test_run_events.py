"""Run events for the dashboard (scripts/run_events.py, deploy/monitoring/).

What the coordinator writes as episodes start and end, that writing it can never
get in a run's way, and that the dashboard config reads the fields the events
carry. Cloud Logging is faked: no network, no credentials.
"""

from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import time
import urllib.error
from pathlib import Path

import pytest

import scripts.distributed_run_pipeline as drp
import scripts.run_events as run_events
from interface.loader import default_maze_path
from interface.smoke_tests.plans import v01_empty_room_trajectory
from scripts import run_pipeline
from scripts.distributed_run_pipeline import (
    CoordinatorStore,
    package_run_archive,
    plan_path,
    prepare_job,
    run_assigned_unit,
)
from scripts.run_events import LOG_ID, RunEvents

_REPO = Path(__file__).resolve().parents[1]
_MANIFEST = _REPO / "gridworld" / "fixtures" / "manifest.json"
_MONITORING = _REPO / "deploy" / "monitoring"


class _Post:
    """Fake Cloud Logging write: records each request; ``fail`` holds exceptions
    to raise, one per call, before it starts succeeding."""

    def __init__(self, fail=()):
        self.fail = list(fail)
        self.requests: list[dict] = []
        self.calls = 0

    def __call__(self, url, body, token):
        self.calls += 1
        if self.fail:
            raise self.fail.pop(0)
        self.requests.append({"url": url, "body": body, "token": token})

    @property
    def payloads(self) -> list[dict]:
        return [e["jsonPayload"] for r in self.requests for e in r["body"]["entries"]]

    def events(self, name: str) -> list[dict]:
        return [p for p in self.payloads if p["event"] == name]


def _events(post: _Post, **kwargs) -> RunEvents:
    kwargs.setdefault("project", "proj")
    return RunEvents(token=lambda: "tok", post=post, sleep=lambda s: None, **kwargs)


def _wait_for(condition, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out waiting for the event writer"
        time.sleep(0.01)


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://logging", code, "err", {}, io.BytesIO(b'{"error": "nope"}'))


# --- a real coordinator lifecycle: prepare, assign, run, upload -------------

def _prepare(tmp_path: Path) -> Path:
    cfg = tmp_path / "rc.json"
    cfg.write_text(json.dumps({
        "experiment_config": {"progress_stall_k": 30},
        "models": {"stub": {"provider": "claude", "model": "stub-model", "group": "stub",
                            "tasks": [str(default_maze_path("V01_empty_room.json"))]}},
    }), encoding="utf-8")
    artifacts = tmp_path / "coordinator"
    prepare_job(run_config_path=cfg, manifest_path=_MANIFEST, seeds=[0], conditions=None,
                artifacts_root=artifacts, run_set_id="dash-run", difficulty_max_static_score=1000.0)
    return artifacts


class _Agent:
    """Replays V01's plan; raises ``fail_with`` from query ``fail_at`` (1-based)."""

    def __init__(self, fail_at=None, fail_with=None):
        self._actions = iter(v01_empty_room_trajectory())
        self._fail_at, self._fail_with = fail_at, fail_with
        self.calls = 0
        self.last_usage = None

    def __call__(self, messages):
        self.calls += 1
        if self._fail_at is not None and self.calls >= self._fail_at:
            raise self._fail_with
        self.last_usage = {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}
        return f"FINAL_OUTPUT: {next(self._actions, 'DONE')}"


def _run_and_upload(store: CoordinatorStore, tmp_path: Path, agent: _Agent, worker: str = "w") -> dict:
    store.register({"worker_id": worker, "hostname": f"{worker}-vm", "capabilities": {"model_group": "stub"}})
    unit = store.assign(worker)["unit"]
    root = tmp_path / f"worker_{worker}_{id(agent)}"
    run_assigned_unit(unit, artifacts_root=root, agent_factory=lambda k, c: (agent, "stub-model"))
    store.upload(worker, unit["unit_id"], package_run_archive(unit, artifacts_root=root).read_bytes())
    return unit


def test_a_solved_episode_writes_a_start_and_a_solved_end(tmp_path):
    post = _Post()
    events = _events(post)
    artifacts = _prepare(tmp_path)
    store = CoordinatorStore(artifacts, events=events)
    unit = _run_and_upload(store, tmp_path, _Agent())
    events.close()
    verified = json.loads((artifacts / unit["run_dir"] / "episode.json").read_text(encoding="utf-8"))

    [started] = post.events("episode_started")
    assert started["run_id"] == "dash-run" and started["unit_id"] == unit["unit_id"]
    assert started["model_id"] == "stub-model" and started["task_id"] == unit["task_id"]
    assert started["worker_id"] == "w" and started["worker_host"] == "w-vm"
    assert started["attempt"] == 1 and started["resumed"] is False

    [ended] = post.events("episode_ended")
    assert ended["status"] == "verified" and ended["outcome"] == "solved" and ended["success"] is True
    # The event reports the episode the coordinator verified, as it was verified.
    steps, queries, max_steps = verified["steps_used"], verified["query_count"], verified["task_spec"]["max_steps"]
    assert verified["success"] is True and steps > 0
    assert (ended["end_reason"], ended["steps_used"], ended["query_count"], ended["max_steps"]) == \
           (verified["end_reason"], steps, queries, max_steps)
    assert ended["wall_seconds"] is not None and ended["wall_seconds"] >= 0
    assert ended["message"] == f"[dash-run] solved stub-model on {unit['task_id']}: " \
                               f"{steps}/{max_steps} steps, {queries} queries"

    severities = [e["severity"] for r in post.requests for e in r["body"]["entries"]]
    assert severities == ["INFO", "INFO"]


def test_an_infra_kill_ends_as_infra_killed_and_its_resume_is_a_new_start(tmp_path):
    post = _Post()
    events = _events(post)
    store = CoordinatorStore(_prepare(tmp_path), events=events)
    unit = _run_and_upload(store, tmp_path, _Agent(3, RuntimeError("connection reset")))
    again = store.assign("w2", {"model_group": "stub"})["unit"]
    events.close()

    assert again["unit_id"] == unit["unit_id"]
    [ended] = post.events("episode_ended")
    assert ended["status"] == "resumable" and ended["outcome"] == "infra_killed"
    assert ended["end_reason"].startswith("agent_error:RuntimeError: connection reset")
    assert ended["steps_used"] == 2 and ended["query_count"] == 2
    assert "(agent_error:RuntimeError: connection reset)" in ended["message"]
    first, resumed = post.events("episode_started")
    assert first["resumed"] is False
    assert resumed["resumed"] is True and resumed["resume_count"] == 1 and resumed["attempt"] == 1
    assert resumed["message"].endswith("(resume 1)")
    warnings = [e for r in post.requests for e in r["body"]["entries"] if e["severity"] == "WARNING"]
    assert [e["jsonPayload"]["event"] for e in warnings] == ["episode_ended"]


# --- fail, stale and re-handouts on a minimal plan ---------------------------

def _minimal_store(tmp_path: Path, monkeypatch, events: RunEvents, **kwargs) -> CoordinatorStore:
    plan = {
        "job_id": "job_test", "run_set_id": "mini",
        "units": [{"unit_id": f"u{i}", "task_id": f"t{i}", "model_id": "m", "model_group": "g"}
                  for i in range(2)],
        "scorer_config": {}, "difficulty_max_static_score": 1000.0,
    }
    plan_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    plan_path(tmp_path).write_text(json.dumps(plan))
    store = CoordinatorStore(tmp_path, events=events, **kwargs)
    monkeypatch.setattr(store, "_unit_payload", lambda plan, unit: dict(unit))
    return store


def test_a_reported_failure_and_a_stale_lease_each_end_the_attempt(tmp_path, monkeypatch):
    post = _Post()
    events = _events(post)
    store = _minimal_store(tmp_path, monkeypatch, events)
    unit_id = store.assign("A", {"model_group": "g"})["unit"]["unit_id"]
    store.fail("A", unit_id, "RuntimeError: worker crashed")
    assert store.assign("B", {"model_group": "g"})["unit"]["unit_id"] == unit_id  # retried first
    state = store.load_state()
    state["units"][unit_id]["heartbeat_at"] = 0  # B went quiet long ago
    store.save_state(state)
    store.status()
    events.close()

    assert [(p["worker_id"], p["attempt"]) for p in post.events("episode_started")] == [("A", 1), ("B", 2)]
    failed, stale = post.events("episode_ended")
    assert (failed["unit_id"], failed["outcome"], failed["worker_id"]) == (unit_id, "failed", "A")
    assert failed["reason"] == "RuntimeError: worker crashed"
    assert failed["steps_used"] is None and failed["success"] is None
    assert (stale["unit_id"], stale["outcome"], stale["worker_id"]) == (unit_id, "stale", "B")
    assert stale["reason"] == "no heartbeat for over 300s"


def test_a_unit_handed_back_to_the_worker_holding_it_is_not_a_new_start(tmp_path, monkeypatch):
    post = _Post()
    events = _events(post)
    store = _minimal_store(tmp_path, monkeypatch, events)
    first = store.assign("A", {"model_group": "g"})["unit"]["unit_id"]
    again = store.assign("A", {"model_group": "g"})["unit"]["unit_id"]  # at its concurrency limit
    events.close()
    assert first == again
    assert [p["unit_id"] for p in post.events("episode_started")] == [first]


def test_without_events_the_coordinator_writes_nothing(tmp_path, monkeypatch):
    store = _minimal_store(tmp_path, monkeypatch, events=None)
    unit_id = store.assign("A", {"model_group": "g"})["unit"]["unit_id"]
    assert store.fail("A", unit_id, "x") == {"ok": True}
    assert store.events is None


# --- monitoring problems never reach the run ---------------------------------

def test_cloud_logging_down_drops_events_but_the_upload_still_verifies(tmp_path):
    post = _Post(fail=[urllib.error.URLError("unreachable")] * 3)  # every attempt of one write
    events = _events(post)
    store = CoordinatorStore(_prepare(tmp_path), events=events)
    store.register({"worker_id": "w", "capabilities": {"model_group": "stub"}})
    unit = store.assign("w")["unit"]
    _wait_for(lambda: post.calls == 3)  # the started event's write has failed for good
    root = tmp_path / "worker"
    run_assigned_unit(unit, artifacts_root=root, agent_factory=lambda k, c: (_Agent(), "stub-model"))
    store.upload("w", unit["unit_id"], package_run_archive(unit, artifacts_root=root).read_bytes())
    events.close()

    assert store.load_state()["units"][unit["unit_id"]]["status"] == "verified"
    # The started event was lost; the next write that landed says so, first.
    [request] = post.requests
    first, rest = request["body"]["entries"][0], request["body"]["entries"][1:]
    assert first["severity"] == "WARNING"
    assert first["jsonPayload"]["event"] == "events_dropped" and first["jsonPayload"]["count"] == 1
    assert [e["jsonPayload"]["event"] for e in rest] == ["episode_ended"]


def test_a_transient_error_is_retried(tmp_path, monkeypatch):
    sleeps: list[float] = []
    post = _Post(fail=[_http_error(503), _http_error(429)])
    events = RunEvents(project="proj", token=lambda: "tok", post=post, sleep=sleeps.append)
    store = _minimal_store(tmp_path, monkeypatch, events)
    store.assign("A", {"model_group": "g"})
    events.close()
    assert sleeps == [1, 2]
    assert [p["event"] for p in post.payloads] == ["episode_started"]


def test_a_permission_error_turns_events_off_with_one_error(tmp_path, monkeypatch, caplog):
    post = _Post(fail=[_http_error(403)])
    events = _events(post)
    store = _minimal_store(tmp_path, monkeypatch, events)
    with caplog.at_level(logging.WARNING, logger="scripts.run_events"):
        unit_id = store.assign("A", {"model_group": "g"})["unit"]["unit_id"]
        events.close()
        assert events.off
        store.fail("A", unit_id, "x")  # nothing more is queued or sent
    assert post.requests == []
    [record] = [r for r in caplog.records if "OFF" in r.getMessage()]
    assert "HTTP 403" in record.getMessage() and record.levelno == logging.ERROR


def test_off_a_gce_vm_events_turn_themselves_off(tmp_path, monkeypatch, caplog):
    def no_metadata(path):
        raise urllib.error.URLError("metadata.google.internal: Name or service not known")

    monkeypatch.setattr(run_events, "_metadata", no_metadata)
    post = _Post()
    events = RunEvents(token=lambda: "tok", post=post, sleep=lambda s: None)
    store = _minimal_store(tmp_path, monkeypatch, events)
    with caplog.at_level(logging.WARNING, logger="scripts.run_events"):
        store.assign("A", {"model_group": "g"})
        events.close()
    assert events.off and post.requests == []
    assert any("no GCE metadata server" in r.getMessage() for r in caplog.records)


def test_a_metadata_blip_does_not_turn_events_off(monkeypatch):
    answers = [urllib.error.URLError("blip"), "proj-from-metadata"]

    def metadata(path):
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(run_events, "_metadata", metadata)
    post = _Post()
    events = RunEvents(token=lambda: "tok", post=post, sleep=lambda s: None)
    events.episode_ended({}, {}, {"status": "failed"}, None, "w", reason="x")
    events.close()
    assert not events.off
    assert post.requests[0]["body"]["logName"] == f"projects/proj-from-metadata/logs/{LOG_ID}"


def test_emitting_never_raises_or_blocks(caplog):
    post = _Post()
    events = _events(post, max_queue=1)
    with caplog.at_level(logging.WARNING, logger="scripts.run_events"):
        events.episode_started(None, None, None, None)  # unbuildable: skipped with a warning
    assert any("could not build" in r.getMessage() for r in caplog.records)

    events.close()  # the writer has stopped, so the queue (size 1) fills up
    events.episode_ended({}, {}, {"status": "failed"}, None, "w", reason="a")
    events.episode_ended({}, {}, {"status": "failed"}, None, "w", reason="b")  # full: dropped, no block
    assert events._take_dropped() == 1


def test_the_write_request_is_one_log_in_the_project():
    post = _Post()
    events = _events(post)
    events.episode_ended({"run_set_id": "r"}, {"unit_id": "u", "model_id": "m", "task_id": "t"},
                         {"status": "stale", "assigned_at": "not-a-time"}, None, "w", reason="quiet")
    events.close()
    [request] = post.requests
    assert request["url"] == run_events.WRITE_URL and request["token"] == "tok"
    body = request["body"]
    assert body["logName"] == f"projects/proj/logs/{LOG_ID}"
    assert body["resource"] == {"type": "global", "labels": {"project_id": "proj"}}
    assert body["partialSuccess"] is True
    [entry] = body["entries"]
    assert entry["severity"] == "WARNING" and entry["timestamp"].endswith("Z")
    assert entry["jsonPayload"]["wall_seconds"] is None
    assert entry["jsonPayload"]["message"] == "[r] stale m on t (quiet)"


# --- coordinator-serve wiring --------------------------------------------------

def test_the_cloud_logging_flag_gives_coordinator_serve_an_event_writer(tmp_path, monkeypatch):
    seen: list = []
    monkeypatch.setattr(drp, "serve_coordinator", lambda root, **kw: seen.append(kw["events"]))
    base = ["--distributed-role", "coordinator-serve", "--artifacts-root", str(tmp_path)]
    run_pipeline.main(base + ["--cloud-logging"])
    run_pipeline.main(base)
    assert isinstance(seen[0], RunEvents) and seen[1] is None
    seen[0].close()


def test_serve_announces_the_run_and_flushes_on_exit(tmp_path, monkeypatch):
    artifacts = _prepare(tmp_path)

    class _Server:
        def serve_forever(self):
            raise KeyboardInterrupt  # an operator's Ctrl-C

        def server_close(self):
            pass

    monkeypatch.setattr(drp, "make_coordinator_server", lambda root, **kw: _Server())
    post = _Post()
    with pytest.raises(KeyboardInterrupt):
        drp.serve_coordinator(artifacts, events=_events(post))
    [started] = post.events("coordinator_started")  # sent by the close in serve's finally
    assert started["run_id"] == "dash-run" and started["unit_count"] == 1
    assert started["units"] == {"pending": 1} and started["models"] == ["stub-model"]
    assert started["message"] == "[dash-run] coordinator started: 1 episodes planned " \
                                 "(0 already verified) for stub-model"


# --- CLOUD_LOGGING through the fleet launchers ----------------------------------

def _run_remote_start(tmp_path: Path, start: str, cloud_logging: str) -> list[str]:
    """Run a coordinator start with a fake ``gcloud compute ssh`` that executes the
    remote command here, in a sandbox HOME with ``python`` and ``curl`` stubbed.
    Returns the argv of every ``python`` the remote script ran."""
    home, stubs = tmp_path / "home", tmp_path / "bin"
    (home / "MultiNet-v2.0" / ".venv-multinet" / "bin").mkdir(parents=True)
    (home / "MultiNet-v2.0" / ".venv-multinet" / "bin" / "activate").write_text("")
    stubs.mkdir()
    py_log = tmp_path / "python.log"
    for name, body in {
        # Like ssh, the remote shell sees none of the caller's environment.
        "gcloud": 'while [[ $# -gt 0 ]]; do [[ "$1" == --command ]] && cmd="$2"; shift; done\n'
                  'cd "$HOME" && exec env -i HOME="$HOME" PATH="$PATH" PY_LOG="$PY_LOG" bash -c "$cmd"\n',
        "python": 'echo "$*" >> "$PY_LOG"\n[[ "${1:-}" == - ]] && cat >/dev/null\nexit 0\n',
        "curl": "exit 0\n",
    }.items():
        (stubs / name).write_text("#!/usr/bin/env bash\n" + body)
        (stubs / name).chmod(0o755)
    env = {**os.environ, "PATH": f"{stubs}:{os.environ['PATH']}", "HOME": str(home), "PY_LOG": str(py_log),
           "DIFFICULTY_MAX": "3000", "CLOUD_LOGGING": cloud_logging}
    r = subprocess.run(
        ["bash", "-c", "source ./lib/distributed_start.sh; COORD=c; ZONE=z; RUN_ID=r1; "
                       f"RUN_CONFIG=rc.json; MANIFEST=m.json; MASSIVE_BATCHES=1; {start}"],
        capture_output=True, text=True, cwd=_REPO, env=env)
    assert r.returncode == 0, r.stderr
    _wait_for(lambda: py_log.exists() and "coordinator-serve" in py_log.read_text())  # nohup'd
    return py_log.read_text().splitlines()


@pytest.mark.parametrize("start", ["start_coordinator", "start_coordinator_combined"])
@pytest.mark.parametrize("cloud_logging, expected", [("1", True), ("", False), ("0", False)])
def test_cloud_logging_1_reaches_coordinator_serve(tmp_path, start, cloud_logging, expected):
    [serve] = [line for line in _run_remote_start(tmp_path, start, cloud_logging) if "coordinator-serve" in line]
    assert serve.startswith("-m scripts.run_pipeline --distributed-role coordinator-serve")
    assert ("--cloud-logging" in serve.split()) is expected


# --- the dashboard reads what the events carry ---------------------------------

def test_the_metrics_and_dashboard_read_the_fields_the_events_carry():
    _, started = run_events.episode_started({}, {}, {}, None)
    _, ended = run_events.episode_ended({}, {}, {"status": "verified"}, None, None)
    payloads = {"episode_started": started, "episode_ended": ended}
    metrics = {}
    for name in ("multinet_episodes_started", "multinet_episodes_ended"):
        metric = json.loads((_MONITORING / f"{name}.json").read_text(encoding="utf-8"))
        assert metric["name"] == name
        assert f'log_id("{LOG_ID}")' in metric["filter"]
        event = metric["filter"].split('jsonPayload.event="')[1].split('"')[0]
        declared = {label["key"] for label in metric["metricDescriptor"]["labels"]}
        assert declared == set(metric["labelExtractors"])
        for key, extractor in metric["labelExtractors"].items():
            assert extractor == f"EXTRACT(jsonPayload.{key})" and key in payloads[event]
        metrics[name] = declared

    dashboard = json.loads((_MONITORING / "dashboard.json").read_text(encoding="utf-8"))
    for tile in dashboard["mosaicLayout"]["tiles"]:
        widget = tile["widget"]
        if "logsPanel" in widget:
            assert widget["logsPanel"]["filter"] == f'log_id("{LOG_ID}")'
        data_sets = (widget.get("xyChart") or widget.get("timeSeriesTable") or {}).get("dataSets", [])
        for data_set in data_sets:
            flt = data_set["timeSeriesQuery"]["timeSeriesFilter"]
            name = flt["filter"].split("logging.googleapis.com/user/")[1].rstrip('"')
            groups = {g.removeprefix("metric.label.") for g in flt["aggregation"]["groupByFields"]}
            assert groups <= metrics[name]
    assert {f["labelKey"] for f in dashboard["dashboardFilters"]} <= metrics["multinet_episodes_ended"]


def test_every_outcome_is_named_in_the_metric_and_dashboard_text():
    outcomes = {run_events._outcome("verified", True), run_events._outcome("verified", False),
                run_events._outcome("resumable", None), run_events._outcome("failed", None),
                run_events._outcome("stale", None)}
    assert outcomes == {"solved", "unsolved", "infra_killed", "failed", "stale"}
    metric = (_MONITORING / "multinet_episodes_ended.json").read_text(encoding="utf-8")
    about = json.loads((_MONITORING / "dashboard.json").read_text(encoding="utf-8"))[
        "mosaicLayout"]["tiles"][0]["widget"]["text"]["content"]
    for outcome in outcomes:
        assert outcome in metric and f"`{outcome}`" in about


# --- deploy/monitoring/apply.sh against a fake gcloud ---------------------------

def _apply(tmp_path: Path, *, metrics_exist: bool, dashboards: str) -> tuple[subprocess.CompletedProcess, str]:
    log = tmp_path / "gcloud.log"
    fake = tmp_path / "gcloud"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        'echo "$*" >> "$GCLOUD_LOG"\n'
        'case "$1 $2 $3" in\n'
        f'  "logging metrics describe") exit {0 if metrics_exist else 1} ;;\n'
        '  "monitoring dashboards list") printf "%b" "$DASHBOARDS" ;;\n'
        "esac\n"
    )
    fake.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "GCLOUD_LOG": str(log),
           "DASHBOARDS": dashboards}
    r = subprocess.run(["bash", str(_MONITORING / "apply.sh"), "proj"], capture_output=True, text=True, env=env)
    return r, log.read_text() if log.exists() else ""


def test_apply_creates_the_metrics_and_dashboard_in_the_named_project(tmp_path):
    r, log = _apply(tmp_path, metrics_exist=False, dashboards="")
    assert r.returncode == 0, r.stderr
    assert "logging metrics create multinet_episodes_started --project proj" in log
    assert "logging metrics create multinet_episodes_ended --project proj" in log
    assert "monitoring dashboards create --project proj --config-from-file" in log
    assert "update" not in log


def test_apply_updates_in_place_when_rerun(tmp_path):
    r, log = _apply(tmp_path, metrics_exist=True, dashboards="projects/1/dashboards/abc\\n")
    assert r.returncode == 0, r.stderr
    assert "logging metrics update multinet_episodes_ended --project proj" in log
    assert "monitoring dashboards update projects/1/dashboards/abc --project proj" in log
    assert "create" not in log


def test_apply_refuses_to_guess_between_duplicate_dashboards(tmp_path):
    r, log = _apply(tmp_path, metrics_exist=True, dashboards="projects/1/dashboards/a\\nprojects/1/dashboards/b\\n")
    assert r.returncode == 1 and "More than one dashboard" in r.stderr
    assert "dashboards update" not in log and "dashboards create" not in log


def test_apply_needs_an_explicit_project(tmp_path):
    r = subprocess.run(["bash", str(_MONITORING / "apply.sh")], capture_output=True, text=True)
    assert r.returncode != 0 and "PROJECT_ID" in r.stderr
