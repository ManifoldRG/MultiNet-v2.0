"""Episode events from the distributed coordinator to Cloud Logging.

With ``coordinator-serve --cloud-logging`` (``CLOUD_LOGGING=1`` on the fleet
launchers) the coordinator writes one structured entry to the
``multinet-run-events`` log of its own project when:

- it starts serving (``coordinator_started``: the run's planned units);
- it hands an episode to a worker (``episode_started``: fresh, retried or
  resumed);
- that attempt ends (``episode_ended``): verified (``solved``/``unsolved``),
  ``infra_killed`` (resumable), ``failed``, or ``stale`` (the worker went
  silent).

The Cloud Monitoring dashboard in ``deploy/monitoring/`` reads these entries;
``deploy/MONITORING.md`` covers setup, access and cost.

Monitoring must never get in a run's way. ``RunEvents`` methods only build a
small dict and queue it: they never block on the network and never raise. One
daemon thread writes to Cloud Logging. When a write fails the events are
dropped, a warning goes to the coordinator log, and the next write that lands
carries an ``events_dropped`` entry with the count, so a gap is visible on the
dashboard. The coordinator's ``/status`` stays the source of truth.

Auth is the VM's service account through the GCE metadata server, so this adds
no dependency. Off GCE (or without permission) it turns itself off with one
warning.
"""

from __future__ import annotations

import datetime
import json
import logging
import queue
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

LOG_ID = "multinet-run-events"
WRITE_URL = "https://logging.googleapis.com/v2/entries:write"
METADATA_URL = "http://metadata.google.internal/computeMetadata/v1"

_BATCH = 100
_ATTEMPTS = 3
_TEXT_CHARS = 300
_STOP = object()


class _Off(Exception):
    """Cloud Logging cannot work for this process (no metadata server, no permission)."""


def _metadata(path: str) -> str:
    req = urllib.request.Request(f"{METADATA_URL}/{path}", headers={"Metadata-Flavor": "Google"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.read().decode("utf-8")


class _MetadataToken:
    """The VM service account's access token, refreshed a minute before it expires."""

    def __init__(self) -> None:
        self._token: Optional[str] = None
        self._expires = 0.0

    def __call__(self) -> str:
        if self._token is None or time.time() >= self._expires:
            data = json.loads(_metadata("instance/service-accounts/default/token"))
            self._token = data["access_token"]
            self._expires = time.time() + int(data.get("expires_in", 0)) - 60
        return self._token


def _post(url: str, body: dict[str, Any], token: str) -> None:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        resp.read()


def _short(value: Any) -> Optional[str]:
    return None if value is None else str(value)[:_TEXT_CHARS]


def _outcome(status: str, success: Any) -> str:
    if status == "verified":
        return "solved" if success else "unsolved"
    if status == "resumable":
        return "infra_killed"
    return status  # failed | stale


def _seconds_since(iso: Any) -> Optional[int]:
    try:
        then = datetime.datetime.strptime(str(iso), "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    return max(0, int((now - then).total_seconds()))


def _episode_fields(plan: dict[str, Any], unit: dict[str, Any], unit_state: dict[str, Any],
                    worker: Optional[dict[str, Any]], worker_id: Optional[str]) -> dict[str, Any]:
    return {
        "run_id": plan.get("run_set_id"),
        "job_id": plan.get("job_id"),
        "unit_id": unit.get("unit_id"),
        "model_id": unit.get("model_id"),
        "model_group": unit.get("model_group"),
        "provider": unit.get("provider"),
        "task_id": unit.get("task_id"),
        "seed": unit.get("seed"),
        "prompt_variant": unit.get("prompt_variant"),
        "backend": unit.get("backend"),
        "worker_id": worker_id,
        "worker_host": (worker or {}).get("hostname"),
        "attempt": int(unit_state.get("attempts", 0)),
        "resume_count": int(unit_state.get("resume_count", 0)),
    }


def episode_started(plan: dict[str, Any], unit: dict[str, Any], unit_state: dict[str, Any],
                    worker: Optional[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    fields = _episode_fields(plan, unit, unit_state, worker, unit_state.get("worker_id"))
    resumed = bool(unit_state.get("resume_pending"))
    how = f"resume {fields['resume_count']}" if resumed else f"attempt {fields['attempt']}"
    fields.update({
        "event": "episode_started",
        "resumed": resumed,
        "message": f"[{fields['run_id']}] started {fields['model_id']} on {fields['task_id']} ({how})",
    })
    return "INFO", fields


def episode_ended(plan: dict[str, Any], unit: dict[str, Any], unit_state: dict[str, Any],
                  worker: Optional[dict[str, Any]], worker_id: Optional[str], *,
                  episode: Optional[dict[str, Any]] = None,
                  reason: Optional[str] = None) -> tuple[str, dict[str, Any]]:
    """One attempt ended. ``episode`` is the uploaded episode.json (verified or
    infra-killed); ``reason`` explains an attempt that ended without one."""
    fields = _episode_fields(plan, unit, unit_state, worker, worker_id)
    status = str(unit_state.get("status"))
    episode = episode or {}
    outcome = _outcome(status, episode.get("success"))
    fields.update({
        "event": "episode_ended",
        "status": status,
        "outcome": outcome,
        "success": episode.get("success"),
        "end_reason": _short(episode.get("end_reason")),
        "steps_used": episode.get("steps_used"),
        "max_steps": (episode.get("task_spec") or {}).get("max_steps"),
        "query_count": episode.get("query_count"),
        # Coordinator-observed: from the hand-out to the result (or the failure).
        "wall_seconds": _seconds_since(unit_state.get("assigned_at")),
        "reason": _short(reason),
    })
    detail = fields["end_reason"] or fields["reason"]
    summary = f"[{fields['run_id']}] {outcome} {fields['model_id']} on {fields['task_id']}"
    if fields["steps_used"] is not None:
        summary += f": {fields['steps_used']}/{fields['max_steps']} steps, {fields['query_count']} queries"
    if detail and outcome != "solved":
        summary += f" ({detail})"
    fields["message"] = summary
    return ("INFO" if outcome in ("solved", "unsolved") else "WARNING"), fields


def coordinator_started(plan_file: str | Path, state_file: str | Path) -> tuple[str, dict[str, Any]]:
    plan = json.loads(Path(plan_file).read_text(encoding="utf-8"))
    state = json.loads(Path(state_file).read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    for unit_state in state.get("units", {}).values():
        status = str(unit_state.get("status"))
        counts[status] = counts.get(status, 0) + 1
    unit_count = len(plan.get("units", []))
    models = sorted({str(u.get("model_id")) for u in plan.get("units", [])})
    done = counts.get("verified", 0)
    return "INFO", {
        "event": "coordinator_started",
        "run_id": plan.get("run_set_id"),
        "job_id": plan.get("job_id"),
        "unit_count": unit_count,
        "units": counts,
        "models": models,
        "message": f"[{plan.get('run_set_id')}] coordinator started: {unit_count} episodes planned "
                   f"({done} already verified) for {', '.join(models)}",
    }


class RunEvents:
    """Best-effort, non-blocking writer of run events to Cloud Logging.

    ``project``, ``token``, ``post``, ``url`` and ``sleep`` are seams for tests;
    on a coordinator VM the defaults use the metadata server."""

    def __init__(
        self,
        *,
        project: Optional[str] = None,
        token: Optional[Callable[[], str]] = None,
        post: Callable[[str, dict[str, Any], str], None] = _post,
        url: str = WRITE_URL,
        max_queue: int = 10_000,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._project = project
        self._token = token or _MetadataToken()
        self._post = post
        self._url = url
        self._sleep = sleep
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._lock = threading.Lock()
        self._dropped = 0
        self.off = False
        self._thread = threading.Thread(target=self._run, name="run-events", daemon=True)
        self._thread.start()

    # --- what the coordinator calls: build now (state changes later), send later ---

    def coordinator_started(self, plan_file: str | Path, state_file: str | Path) -> None:
        self._emit(coordinator_started, plan_file, state_file)

    def episode_started(self, plan: dict[str, Any], unit: dict[str, Any], unit_state: dict[str, Any],
                        worker: Optional[dict[str, Any]]) -> None:
        self._emit(episode_started, plan, unit, unit_state, worker)

    def episode_ended(self, plan: dict[str, Any], unit: dict[str, Any], unit_state: dict[str, Any],
                      worker: Optional[dict[str, Any]], worker_id: Optional[str], *,
                      episode: Optional[dict[str, Any]] = None, reason: Optional[str] = None) -> None:
        self._emit(episode_ended, plan, unit, unit_state, worker, worker_id, episode=episode, reason=reason)

    def close(self, timeout: float = 5.0) -> None:
        """Send what is queued and stop the writer thread, waiting at most ``timeout``
        in all (events still unsent then are lost with the process)."""
        deadline = time.monotonic() + timeout
        try:
            self._queue.put(_STOP, timeout=timeout)
        except queue.Full:
            return
        self._thread.join(max(0.0, deadline - time.monotonic()))

    # --- internals ---

    def _emit(self, build: Callable[..., tuple[str, dict[str, Any]]], *args: Any, **kwargs: Any) -> None:
        if self.off:
            return
        try:
            severity, payload = build(*args, **kwargs)
            entry = {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
                "severity": severity,
                "jsonPayload": payload,
            }
        except Exception:
            logger.warning("run events: could not build a %s event; skipped", build.__name__, exc_info=True)
            return
        try:
            self._queue.put_nowait(entry)
        except queue.Full:
            self._count_dropped(1)

    def _count_dropped(self, n: int) -> None:
        with self._lock:
            self._dropped += n

    def _take_dropped(self) -> int:
        with self._lock:
            n, self._dropped = self._dropped, 0
            return n

    def _run(self) -> None:
        while True:
            batch = [self._queue.get()]
            while len(batch) < _BATCH:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            entries = [e for e in batch if e is not _STOP]
            if entries and not self.off:
                self._write(entries)
            if len(entries) < len(batch):
                return

    def _write(self, events: list[dict[str, Any]]) -> None:
        dropped = self._take_dropped()
        entries = events
        if dropped:
            entries = [{
                "timestamp": events[0]["timestamp"],
                "severity": "WARNING",
                "jsonPayload": {"event": "events_dropped", "count": dropped,
                                "message": f"{dropped} run event(s) could not be written before this point"},
            }] + events
        error: Optional[BaseException] = None
        for attempt in range(1, _ATTEMPTS + 1):
            try:
                project = self._project or self._resolve_project()
                body = {
                    "logName": f"projects/{project}/logs/{LOG_ID}",
                    "resource": {"type": "global", "labels": {"project_id": project}},
                    "partialSuccess": True,
                    "entries": entries,
                }
                self._post(self._url, body, self._token())
                return
            except _Off as exc:  # off only if it persists through every attempt
                error = exc
                if attempt == _ATTEMPTS:
                    self._turn_off(str(exc))
            except urllib.error.HTTPError as exc:
                error = exc
                if exc.code == 403:
                    self._turn_off(f"Cloud Logging refused the write (HTTP 403: {_http_detail(exc)})")
                    break
                if exc.code != 429 and exc.code < 500:
                    break  # a malformed request does not improve on retry
            except Exception as exc:  # network, timeout, token refresh
                error = exc
            if attempt < _ATTEMPTS:
                self._sleep(2 ** (attempt - 1))
        self._count_dropped(dropped + len(events))
        if not self.off:
            logger.warning("run events: dropped %d event(s); Cloud Logging write failed: %s", len(events), error)

    def _resolve_project(self) -> str:
        try:
            self._project = _metadata("project/project-id").strip()
        except Exception as exc:
            raise _Off(f"no GCE metadata server ({exc}); run events need a coordinator VM") from exc
        return self._project

    def _turn_off(self, why: str) -> None:
        self.off = True
        logger.error("run events: OFF for this coordinator, the run dashboard will not show it: %s", why)


def _http_detail(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", "replace")[:_TEXT_CHARS]
    except Exception:
        return str(exc)
