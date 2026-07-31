"""Central-coordinator distributed execution for ``scripts.run_pipeline``.

The coordinator owns Stage 2 artifacts, the durable work plan, worker state,
uploaded run archives, and final report aggregation. Workers run one Stage 3/4
unit at a time against the same helpers used by the single-process pipeline.
"""

from __future__ import annotations

import contextlib
import json
import logging
import shutil
import socket
import tarfile
import tempfile
import threading
import time
import traceback

try:  # POSIX-only; used for cross-process locking of the shared state file.
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None  # type: ignore[assignment]
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Optional

from pipeline import episode_metrics
from scorer import compute_runtime_score, load_scorer_config
from scorer.config import ScorerConfig
from scorer.io import stable_hash

from scripts import run_pipeline as pipeline
from scripts.gcs_storage import StorageConfig, gcs_destination, mirror_to_bucket

logger = logging.getLogger(__name__)


DISTRIBUTED_DIR = "distributed"
PLAN_FILENAME = "job_plan.json"
STATE_FILENAME = "job_state.json"
EXPECTED_RUN_FILES = ("episode.json", "run_inputs.json", "run_score.json")
ACTIVE_STATUSES = {"assigned", "running"}

AgentFactory = Callable[[str, dict[str, Any]], tuple[pipeline.Agent, str]]


def _now() -> float:
    return time.time()


def _iso(ts: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(_now() if ts is None else ts))


def _distributed_root(artifacts_root: str | Path) -> Path:
    return Path(artifacts_root) / DISTRIBUTED_DIR


def plan_path(artifacts_root: str | Path) -> Path:
    return _distributed_root(artifacts_root) / PLAN_FILENAME


def state_path(artifacts_root: str | Path) -> Path:
    return _distributed_root(artifacts_root) / STATE_FILENAME


def _uploads_root(artifacts_root: str | Path) -> Path:
    return _distributed_root(artifacts_root) / "uploads"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        tmp_path.replace(path)
        tmp_path = None
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def _model_label(model_key: str, model_cfg: dict[str, Any]) -> str:
    return pipeline._sanitize(str(model_cfg.get("model") or model_key))


def _model_group(model_key: str, model_cfg: dict[str, Any]) -> str:
    return str(model_cfg.get("group") or _model_label(model_key, model_cfg))


def _unit_id(job_id: str, payload: dict[str, Any]) -> str:
    """Stable identity for one fully resolved distributed unit.

    ``job_id`` supplies the durable job namespace; ``inputs_hash`` carries the
    complete output-affecting identity (episode production plus scoring inputs),
    while ``model_key`` keeps two explicitly declared plan slots distinct even
    if they resolve to the same runtime model recipe. A caller may therefore
    reuse an explicit durable job ID without causing completed work from an
    older resolved config to be mistaken for current work.
    """
    digest = stable_hash(
        {
            "job_id": job_id,
            "model_key": payload["model_key"],
            "inputs_hash": payload["inputs_hash"],
        }
    )
    return f"unit_{digest[:16]}"


def _unit_inputs_hash(
    *,
    episode_inputs_hash: str,
    task_row: dict[str, Any],
    canonical_paths: dict[str, Any],
    scored_static: dict[str, Any],
    scorer_config: ScorerConfig,
    difficulty_max_static_score: float,
) -> str:
    """Hash every output-affecting input of a distributed Stage-3/4 unit.

    ``episode_inputs_hash`` is the canonical local-pipeline recipe used by
    ``run_inputs.json``. The remaining fields cover Stage 4 and aggregate-row
    production, which are also part of a verified unit archive and must not be
    silently reused after their inputs change.
    """
    return stable_hash(
        {
            "schema_version": 1,
            "episode_inputs_hash": episode_inputs_hash,
            "task_row": task_row,
            "canonical_paths_hash": stable_hash(canonical_paths),
            "scored_static_hash": stable_hash(scored_static),
            "scorer_config": scorer_config.to_dict(),
            "difficulty_max_static_score": difficulty_max_static_score,
        }
    )


# Unit statuses that represent completed, paid-for work: preserved across a
# same-job re-prepare so we never re-run them. Everything else is reset.
_TERMINAL_OK_STATUSES = frozenset({"verified", "uploaded"})


def _reset_state_for_rerun(
    existing_state: dict[str, Any], fresh_state: dict[str, Any]
) -> dict[str, Any]:
    """Merge an all-pending ``fresh_state`` onto an existing SAME-job state.

    Keep only terminal-OK units (verified/uploaded) so completed work isn't
    re-run, and reset every other unit (failed/stale/assigned/running/pending)
    to its fresh pending/attempts=0 entry so a re-prepared batch can retry units
    a prior run failed at the attempt cap. New units absent from the old state
    stay pending.
    """
    existing_units = (existing_state or {}).get("units") or {}
    merged_units: dict[str, Any] = {}
    for uid, fresh_unit in fresh_state["units"].items():
        prev = existing_units.get(uid)
        if prev is not None and prev.get("status") in _TERMINAL_OK_STATUSES:
            merged_units[uid] = prev
        else:
            merged_units[uid] = fresh_unit
    merged = dict(existing_state)
    merged["job_id"] = fresh_state["job_id"]
    merged["units"] = merged_units
    merged["updated_at"] = _iso()
    return merged


def _order_units_lpt(units: list[dict[str, Any]],
                     static_by_task: dict[str, Any]) -> list[dict[str, Any]]:
    """Longest-processing-time-first ordering of units. Keyed on the maze's ``max_steps``
    (the 3x-BFS step cap = worst-case episode length) with a static ``optimal_steps``
    fallback; descending, stable. The coordinator assigns in plan order, so the longest
    mazes claim concurrency slots first and the fast mazes fill the tail — minimizing
    batch makespan when episodes are long sequential chains."""
    def _length(unit: dict[str, Any]) -> int:
        payload = unit.get("task_payload") or {}
        ms = payload.get("max_steps")
        if ms is not None:
            return int(ms)
        return int((static_by_task.get(unit["task_id"]) or {}).get("optimal_steps", 0) or 0)

    return sorted(units, key=_length, reverse=True)


def prepare_job(
    *,
    run_config_path: str | Path,
    manifest_path: str | Path,
    seeds: list[int],
    conditions: Optional[str],
    artifacts_root: str | Path,
    run_set_id: str,
    prompt_variant: Optional[str] = None,
    difficulty_max_static_score: Optional[float] = None,
    force: bool = False,
    job_id: Optional[str] = None,
    scorer_config: Optional[ScorerConfig] = None,
) -> dict[str, Any]:
    """Resolve a run-config, run/reuse Stage 2, and write plan/state files."""
    run_config_path = Path(run_config_path)
    manifest_path = Path(manifest_path)
    artifacts_root = Path(artifacts_root)
    config = scorer_config or load_scorer_config()
    run_config = pipeline.load_run_config(run_config_path)
    pipeline.check_run_config_expectations(run_config, manifest_path, conditions)
    catalog = pipeline.load_manifest(manifest_path)
    prompt_variants = pipeline.condition_variant_names(conditions)
    if prompt_variant is not None:
        if prompt_variant not in prompt_variants:
            raise ValueError(
                f"Unknown prompt variant {prompt_variant!r} for --conditions {conditions!r}; "
                f"available: {prompt_variants}."
            )
        prompt_variants = [prompt_variant]

    # Resolve the fully-composed ExperimentConfig once per (conditions, variant)
    # so every unit for that variant embeds the exact same resolved config,
    # including any top-level experiment_config overlay from the run-config.
    exp_overlay = run_config.get("experiment_config") or {}
    variant_configs: dict[str, Any] = {
        name: cfg.to_dict()
        for name, cfg in pipeline._condition_configs(conditions, base_overrides=exp_overlay)
    }

    model_plans: list[tuple[str, str, str, dict[str, Any], list[dict[str, Any]]]] = []
    models: dict[str, dict[str, Any]] = {}
    union: dict[str, dict[str, Any]] = {}
    for model_key, model_cfg in run_config["models"].items():
        entries = model_cfg.get("tasks") or model_cfg.get("runs") or []
        if not entries:
            raise ValueError(f"Model {model_key!r} lists no tasks/runs.")
        rows = pipeline.resolve_task_rows(entries, catalog, manifest_path)
        model_id = _model_label(model_key, model_cfg)
        group = _model_group(model_key, model_cfg)
        model_plans.append((model_key, model_id, group, model_cfg, rows))
        models[model_key] = {
            "model_key": model_key,
            "model_id": model_id,
            "model_group": group,
            "provider": model_cfg.get("provider"),
            "model": model_cfg.get("model"),
            "worker_count": model_cfg.get("worker_count"),
            "hardware_profile": model_cfg.get("hardware_profile"),
            "worker_tags": _as_list(model_cfg.get("worker_tags")),
            "max_in_flight": model_cfg.get("max_in_flight"),
            "task_ids": [r["task_id"] for r in rows],
        }
        for row in rows:
            union.setdefault(row["task_id"], row)

    union_rows = list(union.values())
    static_by_task, difficulty_max = pipeline._score_suite(
        union_rows,
        manifest_path,
        artifacts_root,
        config,
        force,
        difficulty_max_static_score=difficulty_max_static_score,
    )

    if job_id is None:
        job_digest = stable_hash(
            {
                # Strip the top-level ``phase`` provenance block (Task B3) before
                # hashing: it labels the two-tier pass but is NOT a generation
                # input, so a re-labeled phase must not churn job_id/unit_id and
                # orphan already-paid units. The cap change (max_tokens) already
                # differentiates the phases via each unit's episode_inputs_hash.
                "run_config": {k: v for k, v in run_config.items() if k != "phase"},
                "manifest_path": str(manifest_path),
                "seeds": [int(s) for s in seeds],
                "conditions": conditions,
                "run_set_id": run_set_id,
                "pipeline_version": pipeline.PIPELINE_VERSION,
            }
        )
        job_id = f"job_{job_digest[:12]}"

    units: list[dict[str, Any]] = []
    for model_key, model_id, group, model_cfg, rows in model_plans:
        for row in rows:
            task_id = row["task_id"]
            if not static_by_task[task_id].get("is_beatable", True):
                continue
            source = pipeline._resolve_source(row, manifest_path)
            source_payload = json.loads(source.read_text(encoding="utf-8"))
            canonical_paths = _read_json(
                artifacts_root / "tasks" / task_id / "canonical_paths.json"
            )
            runtime_spec = pipeline._runtime_capped_spec(
                pipeline.task_spec_from_payload(source_payload), canonical_paths
            )
            for seed in seeds:
                for variant in prompt_variants:
                    run_rel = (
                        Path("runs")
                        / task_id
                        / "minigrid"
                        / model_id
                        / f"seed_{int(seed)}"
                        / variant
                    )
                    unit = {
                        "job_id": job_id,
                        "model_key": model_key,
                        "model_id": model_id,
                        "model_group": group,
                        "provider": model_cfg.get("provider"),
                        "model_config": dict(model_cfg),
                        "task_id": task_id,
                        "task_row": dict(row),
                        "task_source_metadata": {
                            "source": row.get("source"),
                            "resolved_source": str(source),
                        },
                        "task_payload": source_payload,
                        "seed": int(seed),
                        "prompt_variant": variant,
                        "condition_set": conditions,
                        "experiment_config": variant_configs[variant],
                        "backend": "minigrid",
                        "run_dir": run_rel.as_posix(),
                        "expected_files": list(EXPECTED_RUN_FILES),
                        "hardware_profile": model_cfg.get("hardware_profile"),
                        "worker_tags": _as_list(model_cfg.get("worker_tags")),
                        "max_in_flight": model_cfg.get("max_in_flight"),
                    }
                    episode_inputs_hash = pipeline._expected_run_hash(
                        runtime_spec,
                        model_id,
                        int(seed),
                        "minigrid",
                        condition_set=conditions,
                        prompt_variant=variant,
                        experiment_config=variant_configs[variant],
                        model_config=model_cfg,
                    )
                    unit["episode_inputs_hash"] = episode_inputs_hash
                    unit["inputs_hash"] = _unit_inputs_hash(
                        episode_inputs_hash=episode_inputs_hash,
                        task_row=dict(row),
                        canonical_paths=canonical_paths,
                        scored_static=static_by_task[task_id],
                        scorer_config=config,
                        difficulty_max_static_score=difficulty_max,
                    )
                    unit["unit_id"] = _unit_id(job_id, unit)
                    units.append(unit)

    # Longest-processing-time-first: the coordinator assigns in plan order, so ordering
    # the long mazes first makes them claim concurrency slots immediately and the fast
    # mazes (empty_room) fill the tail — minimizing batch makespan when episodes are long
    # sequential chains (a hard corridor can grind hundreds of steps).
    units = _order_units_lpt(units, static_by_task)

    plan = {
        "schema_version": "0.1.0",
        "job_id": job_id,
        "created_at": _iso(),
        "pipeline_version": pipeline.PIPELINE_VERSION,
        "run_config_path": str(run_config_path),
        "manifest_path": str(manifest_path),
        "artifacts_root": str(artifacts_root),
        "run_set_id": run_set_id,
        "seeds": [int(s) for s in seeds],
        "conditions": conditions,
        "prompt_variants": prompt_variants,
        "difficulty_max_static_score": difficulty_max,
        "scorer_config": config.to_dict(),
        "models": models,
        "tasks": union_rows,
        "static_by_task": static_by_task,
        "units": units,
    }
    state = {
        "schema_version": "0.1.0",
        "job_id": job_id,
        "created_at": _iso(),
        "updated_at": _iso(),
        "workers": {},
        "units": {
            unit["unit_id"]: {
                "status": "pending",
                "attempts": 0,
                "worker_id": None,
                "updated_at": _iso(),
            }
            for unit in units
        },
    }
    _write_json_atomic(plan_path(artifacts_root), plan)
    # Preserve progress on re-run: only (re)initialise the all-pending state when
    # there is no (readable) state yet, or the plan changed (different job_id).
    # Overwriting an existing same-job state would revert verified units to
    # "pending" and re-run (re-pay for) work that already completed. A corrupt/
    # unreadable state is treated as absent so prepare still self-heals.
    existing_state_path = state_path(artifacts_root)
    existing_state: Optional[dict[str, Any]] = None
    if existing_state_path.exists():
        try:
            existing_state = _read_json(existing_state_path)
        except (json.JSONDecodeError, OSError):
            existing_state = None
    if existing_state is None or existing_state.get("job_id") != job_id:
        _write_json_atomic(existing_state_path, state)
    else:
        # Same job re-prepared (e.g. next-batch re-running a batch on a reused
        # fleet): keep completed units so paid work isn't re-run, but reset every
        # non-terminal unit to fresh pending. Without this, a prior run that
        # failed all units at the attempt cap (e.g. the since-fixed vLLM GPU-OOM)
        # leaves them failed-at-cap forever, so the coordinator dispatches nothing
        # and every worker idles.
        _write_json_atomic(existing_state_path, _reset_state_for_rerun(existing_state, state))
    _uploads_root(artifacts_root).mkdir(parents=True, exist_ok=True)
    return plan


class CoordinatorStore:
    """Durable coordinator state backed by JSON files under artifacts_root."""

    # Statuses a unit can be (re)handed out from. "failed"/"stale" units are
    # retried so a transient worker/upload failure does not strand a paid unit;
    # the attempt cap stops a poison unit from re-spending tokens forever.
    REASSIGNABLE_STATUSES = frozenset({"pending", "stale", "failed"})

    def __init__(
        self,
        artifacts_root: str | Path,
        stale_after_seconds: float = 300.0,
        storage: Optional[StorageConfig] = None,
        max_unit_attempts: int = 3,
    ):
        self.artifacts_root = Path(artifacts_root)
        self.stale_after_seconds = float(stale_after_seconds)
        self.storage = storage
        self.max_unit_attempts = int(max_unit_attempts)
        self._lock = threading.RLock()
        self._lock_path = state_path(self.artifacts_root).with_name(STATE_FILENAME + ".lock")

    @contextlib.contextmanager
    def _guard(self):
        """Serialise a load-modify-save against both other threads (RLock) and
        other processes (an exclusive ``flock`` on a sidecar lock file).

        ``coordinator-serve`` and ``coordinator-run-api-client`` are separate
        processes that both read-modify-write ``job_state.json``; the in-process
        RLock alone cannot stop one from clobbering the other's just-written
        assignment (a lost update that re-runs a paid unit). The methods never
        nest ``_guard``, so a single exclusive lock is deadlock-free.
        """
        with self._lock:
            if fcntl is None:
                yield
                return
            self._lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_file = open(self._lock_path, "w")
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                yield
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
                lock_file.close()

    def load_plan(self) -> dict[str, Any]:
        path = plan_path(self.artifacts_root)
        if not path.exists():
            raise FileNotFoundError(f"Distributed job plan not found: {path}")
        return _read_json(path)

    def load_state(self) -> dict[str, Any]:
        path = state_path(self.artifacts_root)
        if path.exists():
            return _read_json(path)
        plan = self.load_plan()
        state = {
            "schema_version": "0.1.0",
            "job_id": plan["job_id"],
            "created_at": _iso(),
            "updated_at": _iso(),
            "workers": {},
            "units": {
                unit["unit_id"]: {
                    "status": "pending",
                    "attempts": 0,
                    "worker_id": None,
                    "updated_at": _iso(),
                }
                for unit in plan["units"]
            },
        }
        self.save_state(state)
        return state

    def save_state(self, state: dict[str, Any]) -> None:
        state["updated_at"] = _iso()
        _write_json_atomic(state_path(self.artifacts_root), state)

    def register(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._guard():
            state = self.load_state()
            worker_id = payload.get("worker_id") or f"worker_{uuid.uuid4().hex[:12]}"
            worker = state["workers"].get(worker_id, {})
            worker.update(
                {
                    "worker_id": worker_id,
                    "capabilities": dict(payload.get("capabilities") or {}),
                    "hostname": payload.get("hostname") or socket.gethostname(),
                    "registered_at": worker.get("registered_at") or _iso(),
                    "heartbeat_at": _now(),
                    "status": "idle",
                }
            )
            state["workers"][worker_id] = worker
            self.save_state(state)
            return {"worker_id": worker_id, "job_id": state["job_id"]}

    def assign(self, worker_id: str, capabilities: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        with self._guard():
            plan = self.load_plan()
            state = self.load_state()
            if worker_id not in state["workers"]:
                state["workers"][worker_id] = {
                    "worker_id": worker_id,
                    "registered_at": _iso(),
                    "hostname": socket.gethostname(),
                    "capabilities": {},
                }
            worker = state["workers"][worker_id]
            if capabilities:
                worker["capabilities"] = dict(capabilities)
            worker["heartbeat_at"] = _now()
            worker["status"] = "polling"
            self._mark_stale(plan, state)

            # A worker may hold up to its declared concurrency in active units at once
            # (served-vLLM workers run many episodes in parallel so the server batches
            # their prefills). Re-return an already-active unit ONLY when the worker is
            # at its limit — a fresh assign call below its limit falls through and hands
            # out a new *distinct* unit. worker_concurrency defaults to 1, preserving the
            # original serial one-unit-per-worker behaviour exactly.
            active_units = [
                unit for unit in plan["units"]
                if state["units"][unit["unit_id"]].get("worker_id") == worker_id
                and state["units"][unit["unit_id"]].get("status") in ACTIVE_STATUSES
            ]
            worker_concurrency = max(
                int((worker.get("capabilities") or {}).get("worker_concurrency") or 1), 1
            )
            if active_units and len(active_units) >= worker_concurrency:
                self.save_state(state)
                return {"unit": self._unit_payload(plan, active_units[0]), "job_id": plan["job_id"]}

            for unit in plan["units"]:
                unit_state = state["units"][unit["unit_id"]]
                if unit_state.get("status") not in self.REASSIGNABLE_STATUSES:
                    continue
                if int(unit_state.get("attempts", 0)) >= self.max_unit_attempts:
                    continue  # attempt cap reached -> leave terminal, report at finalize
                if not self._unit_matches_worker(unit, worker):
                    continue
                if not self._below_max_in_flight(unit, state, plan):
                    continue
                unit_state.update(
                    {
                        "status": "assigned",
                        "worker_id": worker_id,
                        "assigned_at": _iso(),
                        "heartbeat_at": _now(),
                        "attempts": int(unit_state.get("attempts", 0)) + 1,
                        "updated_at": _iso(),
                    }
                )
                worker["status"] = "assigned"
                worker["current_unit_id"] = unit["unit_id"]
                self.save_state(state)
                return {"unit": self._unit_payload(plan, unit), "job_id": plan["job_id"]}

            worker["status"] = "idle"
            worker.pop("current_unit_id", None)
            self.save_state(state)
            return {"unit": None, "job_id": plan["job_id"]}

    def heartbeat(
        self, worker_id: str, unit_id: Optional[str] = None, progress: Optional[int] = None
    ) -> dict[str, Any]:
        with self._guard():
            state = self.load_state()
            worker = state["workers"].setdefault(worker_id, {"worker_id": worker_id})
            worker["heartbeat_at"] = _now()
            worker["status"] = "running" if unit_id else "idle"
            if unit_id and unit_id in state["units"]:
                unit_state = state["units"][unit_id]
                if unit_state.get("worker_id") == worker_id and unit_state.get("status") in ACTIVE_STATUSES:
                    unit_state["status"] = "running"
                    unit_state["heartbeat_at"] = _now()
                    unit_state["updated_at"] = _iso()
                    if progress is not None:
                        # Monotonic: a re-attempt that resets to 0 must not move it backwards.
                        unit_state["progress"] = max(int(unit_state.get("progress", 0)), int(progress))
                    worker["current_unit_id"] = unit_id
            self.save_state(state)
            return {"ok": True}

    def fail(self, worker_id: str, unit_id: str, reason: str) -> dict[str, Any]:
        with self._guard():
            state = self.load_state()
            unit_state = state["units"].get(unit_id)
            if unit_state is None:
                raise KeyError(f"Unknown unit_id: {unit_id}")
            unit_state.update(
                {
                    "status": "failed",
                    "worker_id": worker_id,
                    "failed_at": _iso(),
                    "failure_reason": reason,
                    "updated_at": _iso(),
                }
            )
            worker = state["workers"].setdefault(worker_id, {"worker_id": worker_id})
            worker["status"] = "failed"
            worker.pop("current_unit_id", None)
            self.save_state(state)
            return {"ok": True}

    def upload(self, worker_id: str, unit_id: str, archive_bytes: bytes) -> dict[str, Any]:
        with self._guard():
            plan = self.load_plan()
            state = self.load_state()
            unit = self._find_unit(plan, unit_id)
            unit_state = state["units"].get(unit_id)
            if unit_state is None:
                raise KeyError(f"Unknown unit_id: {unit_id}")
            if unit_state.get("worker_id") not in (None, worker_id):
                raise ValueError(f"Unit {unit_id} is assigned to {unit_state.get('worker_id')}, not {worker_id}.")

            archive_path = _uploads_root(self.artifacts_root) / f"{unit_id}.tar.gz"
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            archive_path.write_bytes(archive_bytes)
            unit_state.update(
                {
                    "status": "uploaded",
                    "worker_id": worker_id,
                    "uploaded_at": _iso(),
                    "archive_path": str(archive_path),
                    "updated_at": _iso(),
                }
            )
            try:
                self._verify_and_extract(unit, archive_path)
            except Exception as exc:
                unit_state.update(
                    {
                        "status": "failed",
                        "failure_reason": str(exc),
                        "updated_at": _iso(),
                    }
                )
                self.save_state(state)
                raise

            unit_state.update(
                {
                    "status": "verified",
                    "verified_at": _iso(),
                    "updated_at": _iso(),
                }
            )
            worker = state["workers"].setdefault(worker_id, {"worker_id": worker_id})
            worker["status"] = "idle"
            worker.pop("current_unit_id", None)
            self.save_state(state)
            run_dir = unit["run_dir"]
            run_set_id = plan.get("run_set_id", "")

        # Durable off-box copy happens outside the lock so a slow gsutil call does
        # not block other workers' heartbeats/uploads.
        self._mirror_unit(unit_id, run_dir, run_set_id)
        return {"ok": True, "status": "verified"}

    def _mirror_unit(self, unit_id: str, run_dir: str, run_set_id: str) -> None:
        """Best-effort mirror of a verified run dir to the configured bucket.

        A mirror failure must never lose the (paid) run: the local copy stays and
        the unit is flagged ``gcs_pending`` for a re-push at finalize."""
        if not (self.storage and self.storage.active):
            return
        src = self.artifacts_root / run_dir
        dest = gcs_destination(self.storage.bucket, run_set_id, run_dir)
        try:
            mirror_to_bucket(src, dest, gsutil=self.storage.gsutil)
            result = {"gcs_uri": dest, "gcs_pending": False}
        except Exception as exc:
            logger.error("GCS mirror failed for unit %s -> %s: %s", unit_id, dest, exc)
            result = {"gcs_pending": True, "gcs_error": str(exc)}
        with self._guard():
            state = self.load_state()
            unit_state = state["units"].get(unit_id)
            if unit_state is not None:
                unit_state.update(result)
                unit_state["updated_at"] = _iso()
                self.save_state(state)

    def status(self) -> dict[str, Any]:
        with self._guard():
            plan = self.load_plan()
            state = self.load_state()
            self._mark_stale(plan, state)
            counts: dict[str, int] = {}
            progress_total = 0
            for unit_state in state["units"].values():
                status = str(unit_state.get("status"))
                counts[status] = counts.get(status, 0) + 1
                progress_total += int(unit_state.get("progress", 0))
            self.save_state(state)
            return {
                "job_id": plan["job_id"],
                "unit_count": len(plan["units"]),
                "units": counts,
                "worker_count": len(state["workers"]),
                "progress_total": progress_total,
            }

    def _mark_stale(self, plan: dict[str, Any], state: dict[str, Any]) -> None:
        now = _now()
        for unit in plan["units"]:
            unit_state = state["units"][unit["unit_id"]]
            if unit_state.get("status") not in ACTIVE_STATUSES:
                continue
            heartbeat_at = float(unit_state.get("heartbeat_at") or 0)
            if now - heartbeat_at <= self.stale_after_seconds:
                continue
            unit_state.update(
                {
                    "status": "stale",
                    "previous_worker_id": unit_state.get("worker_id"),
                    "worker_id": None,
                    "stale_at": _iso(),
                    "updated_at": _iso(),
                }
            )

    def _active_unit_for_worker(
        self,
        plan: dict[str, Any],
        state: dict[str, Any],
        worker_id: str,
    ) -> Optional[dict[str, Any]]:
        for unit in plan["units"]:
            unit_state = state["units"][unit["unit_id"]]
            if unit_state.get("worker_id") == worker_id and unit_state.get("status") in ACTIVE_STATUSES:
                return unit
        return None

    def _unit_matches_worker(self, unit: dict[str, Any], worker: dict[str, Any]) -> bool:
        caps = worker.get("capabilities") or {}
        worker_groups = _as_list(caps.get("model_groups") or caps.get("model_group") or caps.get("groups"))
        if unit.get("model_group") and unit.get("model_group") not in worker_groups:
            return False
        required_profile = unit.get("hardware_profile")
        if required_profile and caps.get("hardware_profile") != required_profile:
            return False
        required_tags = set(_as_list(unit.get("worker_tags")))
        worker_tags = set(_as_list(caps.get("worker_tags") or caps.get("tags")))
        if required_tags and not required_tags.issubset(worker_tags):
            return False
        return True

    def _below_max_in_flight(
        self,
        unit: dict[str, Any],
        state: dict[str, Any],
        plan: dict[str, Any],
    ) -> bool:
        raw_limit = unit.get("max_in_flight")
        if raw_limit in (None, ""):
            return True
        limit = int(raw_limit)
        if limit <= 0:
            return False
        unit_by_id = {u["unit_id"]: u for u in plan["units"]}
        active = 0
        for unit_id, unit_state in state["units"].items():
            if unit_state.get("status") not in ACTIVE_STATUSES:
                continue
            other = unit_by_id.get(unit_id)
            if other and other.get("model_group") == unit.get("model_group"):
                active += 1
        return active < limit

    def _unit_payload(self, plan: dict[str, Any], unit: dict[str, Any]) -> dict[str, Any]:
        task_id = unit["task_id"]
        task_dir = self.artifacts_root / "tasks" / task_id
        payload = dict(unit)
        payload["scorer_config"] = plan["scorer_config"]
        payload["difficulty_max_static_score"] = plan["difficulty_max_static_score"]
        payload["task_artifacts"] = {
            "canonical_paths": _read_json(task_dir / "canonical_paths.json"),
            "scored_static": _read_json(task_dir / "scored_static.json"),
            "task_payload": unit["task_payload"],
        }
        return payload

    def _find_unit(self, plan: dict[str, Any], unit_id: str) -> dict[str, Any]:
        for unit in plan["units"]:
            if unit["unit_id"] == unit_id:
                return unit
        raise KeyError(f"Unknown unit_id: {unit_id}")

    def _verify_and_extract(self, unit: dict[str, Any], archive_path: Path) -> None:
        expected = set(unit.get("expected_files") or EXPECTED_RUN_FILES)
        with tarfile.open(archive_path, "r:gz") as tar:
            members = tar.getmembers()
            names = {PurePosixPath(member.name) for member in members if member.isfile()}
            root_names = {path.name for path in names if len(path.parts) == 1}
            missing = expected - root_names
            if missing:
                raise ValueError(f"Archive for {unit['unit_id']} is missing files: {sorted(missing)}")
            for member in members:
                path = PurePosixPath(member.name)
                if member.issym() or member.islnk():
                    raise ValueError(f"Archive contains link member: {member.name}")
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"Unsafe archive path: {member.name}")

            dest = self.artifacts_root / unit["run_dir"]
            tmp = dest.parent / f".{dest.name}.{unit['unit_id']}.tmp"
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir(parents=True, exist_ok=True)
            try:
                tar.extractall(tmp)
                json_payloads: dict[str, Any] = {}
                for name in expected:
                    fpath = tmp / name
                    if not fpath.exists():
                        raise ValueError(f"Extracted archive missing {name}")
                    # A present-but-truncated JSON file would otherwise pass
                    # verification and crash finalize; reject it here so the unit
                    # is retried instead.
                    if name.endswith(".json"):
                        try:
                            json_payloads[name] = json.loads(
                                fpath.read_text(encoding="utf-8")
                            )
                        except (ValueError, OSError) as exc:
                            raise ValueError(f"Extracted {name} is not valid JSON: {exc}") from exc
                expected_episode_hash = unit.get("episode_inputs_hash")
                if expected_episode_hash is not None:
                    sidecar = json_payloads.get("run_inputs.json")
                    actual_episode_hash = (
                        sidecar.get("inputs_hash")
                        if isinstance(sidecar, dict)
                        else None
                    )
                    if actual_episode_hash != expected_episode_hash:
                        raise ValueError(
                            f"Archive for {unit['unit_id']} has run_inputs.json "
                            "inputs_hash mismatch: "
                            f"expected {expected_episode_hash}, got {actual_episode_hash}"
                        )
                if dest.exists():
                    shutil.rmtree(dest)
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp.replace(dest)
            finally:
                if tmp.exists():
                    shutil.rmtree(tmp)


class CoordinatorClient:
    def __init__(self, base_url: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)

    def register(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._json("POST", "/register", payload)

    def assign(self, worker_id: str, capabilities: dict[str, Any]) -> dict[str, Any]:
        return self._json("POST", "/assign", {"worker_id": worker_id, "capabilities": capabilities})

    def heartbeat(
        self, worker_id: str, unit_id: Optional[str] = None, progress: Optional[int] = None
    ) -> dict[str, Any]:
        return self._json(
            "POST", "/heartbeat", {"worker_id": worker_id, "unit_id": unit_id, "progress": progress}
        )

    def fail(self, worker_id: str, unit_id: str, reason: str) -> dict[str, Any]:
        return self._json("POST", "/fail", {"worker_id": worker_id, "unit_id": unit_id, "reason": reason})

    def upload(self, worker_id: str, unit_id: str, archive_path: Path) -> dict[str, Any]:
        query = urllib.parse.urlencode({"worker_id": worker_id, "unit_id": unit_id})
        req = urllib.request.Request(
            f"{self.base_url}/upload?{query}",
            data=archive_path.read_bytes(),
            headers={"Content-Type": "application/gzip"},
            method="POST",
        )
        return self._open_json(req)

    def status(self) -> dict[str, Any]:
        return self._json("GET", "/status", None)

    def _json(self, method: str, path: str, payload: Optional[dict[str, Any]]) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"} if data is not None else {}
        req = urllib.request.Request(f"{self.base_url}{path}", data=data, headers=headers, method=method)
        return self._open_json(req)

    def _open_json(self, req: urllib.request.Request) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Coordinator request failed ({exc.code}): {detail}") from exc
        return json.loads(raw.decode("utf-8"))


def make_coordinator_server(
    artifacts_root: str | Path,
    *,
    host: str = "0.0.0.0",
    port: int = 8765,
    stale_after_seconds: float = 300.0,
    storage: Optional[StorageConfig] = None,
) -> ThreadingHTTPServer:
    store = CoordinatorStore(artifacts_root, stale_after_seconds=stale_after_seconds, storage=storage)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802
            try:
                if urllib.parse.urlparse(self.path).path == "/status":
                    self._send_json(200, store.status())
                    return
                self._send_json(404, {"error": "not found"})
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})

        def do_POST(self) -> None:  # noqa: N802
            try:
                parsed = urllib.parse.urlparse(self.path)
                if parsed.path == "/upload":
                    params = urllib.parse.parse_qs(parsed.query)
                    worker_id = (params.get("worker_id") or [""])[0]
                    unit_id = (params.get("unit_id") or [""])[0]
                    length = int(self.headers.get("Content-Length") or 0)
                    body = self.rfile.read(length)  # always drain so HTTP/1.1 keep-alive stays in sync
                    if not worker_id or not unit_id:
                        self._send_json(400, {"error": "worker_id and unit_id are required"})
                        return
                    self._send_json(200, store.upload(worker_id, unit_id, body))
                    return

                payload = self._read_json_body()
                if parsed.path == "/register":
                    self._send_json(200, store.register(payload))
                elif parsed.path == "/assign":
                    self._send_json(
                        200,
                        store.assign(str(payload["worker_id"]), payload.get("capabilities")),
                    )
                elif parsed.path == "/heartbeat":
                    self._send_json(
                        200,
                        store.heartbeat(
                            str(payload["worker_id"]),
                            payload.get("unit_id"),
                            payload.get("progress"),
                        ),
                    )
                elif parsed.path == "/fail":
                    self._send_json(
                        200,
                        store.fail(str(payload["worker_id"]), str(payload["unit_id"]), str(payload.get("reason", ""))),
                    )
                else:
                    self._send_json(404, {"error": "not found"})
            except Exception as exc:
                self._send_json(500, {"error": str(exc), "traceback": traceback.format_exc()})

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length == 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    return ThreadingHTTPServer((host, int(port)), Handler)


def serve_coordinator(
    artifacts_root: str | Path,
    *,
    host: str = "0.0.0.0",
    port: int = 8765,
    stale_after_seconds: float = 300.0,
    storage: Optional[StorageConfig] = None,
) -> None:
    server = make_coordinator_server(
        artifacts_root,
        host=host,
        port=port,
        stale_after_seconds=stale_after_seconds,
        storage=storage,
    )
    print(f"Coordinator serving {artifacts_root} on http://{host}:{port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()


def materialize_worker_inputs(unit: dict[str, Any], artifacts_root: str | Path) -> dict[str, Any]:
    artifacts_root = Path(artifacts_root)
    task_id = unit["task_id"]
    task_dir = artifacts_root / "tasks" / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    artifacts = unit["task_artifacts"]
    _write_json_atomic(task_dir / "canonical_paths.json", artifacts["canonical_paths"])
    _write_json_atomic(task_dir / "scored_static.json", artifacts["scored_static"])

    source_dir = artifacts_root / DISTRIBUTED_DIR / "task_sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_path = source_dir / f"{task_id}.json"
    _write_json_atomic(source_path, artifacts["task_payload"])
    row = dict(unit["task_row"])
    row["source"] = str(source_path)
    return row


class ProgressCounter:
    """Mutable, shared between the worker's main thread (writer, via _CountingAgent)
    and its heartbeat thread (reader). A single int; a missed/stale read only delays
    a stall-clock reset by one heartbeat and cannot cause a false stall."""

    __slots__ = ("count",)

    def __init__(self) -> None:
        self.count = 0


class _CountingAgent:
    """Wraps a live agent to count completed generation calls (one per model
    generation, including parse-retries) for progress heartbeats. Increments
    AFTER the call returns so a hung generation does not advance progress.

    Delegates attribute reads *and writes* to the inner agent so the runner sees
    the original API. The write delegation matters: the runner resets
    ``last_usage`` on the agent before each call and reads it back after, so a
    write that stayed on the wrapper would shadow the inner's real value and
    silently drop usage telemetry for agents without a ``reset_usage()`` method
    (e.g. the Kimi/Claude agents)."""

    def __init__(self, inner: Any, counter: ProgressCounter) -> None:
        # Bypass our own __setattr__ for the two private fields; everything else
        # must reach the inner agent.
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_counter", counter)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        result = self._inner(*args, **kwargs)
        self._counter.count += 1
        return result

    def __getattr__(self, name: str) -> Any:
        # Only fires for names not found normally; guard the private fields so a
        # lookup before __init__ finishes cannot recurse forever.
        if name in ("_inner", "_counter"):
            raise AttributeError(name)
        attr = getattr(self._inner, name)
        if name == "generate" and callable(attr):
            # ExperimentRunner.run now prefers ``agent.generate(messages)`` over
            # ``__call__`` (interface/runner.py). Delegating generate straight to
            # the inner agent would bypass the counter, freezing per-unit progress
            # heartbeats at 0 and tripping supervise_run.sh's stall detector on a
            # healthy long run. Count generate the same as __call__. Returning the
            # wrapper only when the inner actually HAS generate preserves the
            # runner's ``hasattr(agent, "generate")`` legacy-double check.
            def _counting_generate(*args: Any, **kwargs: Any) -> Any:
                result = attr(*args, **kwargs)
                self._counter.count += 1
                return result

            return _counting_generate
        return attr

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_inner", "_counter"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._inner, name, value)


def run_assigned_unit(
    unit: dict[str, Any],
    *,
    artifacts_root: str | Path,
    agent_factory: Optional[AgentFactory] = None,
    force: bool = False,
    progress: Optional[ProgressCounter] = None,
) -> tuple[dict[str, Any], Optional[float]]:
    artifacts_root = Path(artifacts_root)
    row = materialize_worker_inputs(unit, artifacts_root)
    factory = agent_factory or pipeline._build_agent_from_spec
    agent, _ = factory(unit["model_key"], unit["model_config"])
    if progress is not None:
        agent = _CountingAgent(agent, progress)
    from interface.config import ExperimentConfig

    if "experiment_config" not in unit:
        raise RuntimeError(
            f"Unit {unit.get('unit_id')} has no 'experiment_config' — this job_plan.json "
            "predates the resolved-config change; regenerate it with the coordinator-prepare role."
        )

    result = pipeline._run_one_unit(
        row,
        agent,
        unit["model_id"],
        model_config=unit.get("model_config"),
        manifest_path=artifacts_root / DISTRIBUTED_DIR / "worker_manifest.json",
        artifacts_root=artifacts_root,
        static_by_task={unit["task_id"]: unit["task_artifacts"]["scored_static"]},
        difficulty_max=float(unit["difficulty_max_static_score"]),
        config=ScorerConfig.from_dict(unit["scorer_config"]),
        seed=int(unit["seed"]),
        prompt_variant=str(unit["prompt_variant"]),
        experiment_config=ExperimentConfig.from_dict(unit["experiment_config"]),
        conditions=unit.get("condition_set"),
        force=force,
    )
    if result is None:
        raise RuntimeError(f"Assigned unit {unit['unit_id']} is not runnable.")
    return result


def package_run_archive(
    unit: dict[str, Any],
    *,
    artifacts_root: str | Path,
    archive_path: Optional[str | Path] = None,
) -> Path:
    artifacts_root = Path(artifacts_root)
    run_dir = artifacts_root / unit["run_dir"]
    if archive_path is None:
        archive_path = artifacts_root / DISTRIBUTED_DIR / f"{unit['unit_id']}.tar.gz"
    archive_path = Path(archive_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    for name in unit.get("expected_files") or EXPECTED_RUN_FILES:
        if not (run_dir / name).exists():
            raise FileNotFoundError(f"Run directory is missing expected file: {run_dir / name}")
    with tarfile.open(archive_path, "w:gz") as tar:
        for path in sorted(run_dir.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=str(path.relative_to(run_dir)))
    return archive_path


def _load_worker_state(path: Path) -> dict[str, Any]:
    if path.exists():
        return _read_json(path)
    return {}


def _safe_heartbeat(client: Any, worker_id: str, unit_id: str, progress: Optional[int] = None) -> bool:
    """Send a heartbeat, swallowing transient errors so a brief coordinator
    outage cannot kill the heartbeat thread (which would let the unit go stale
    and be double-assigned while this worker is still running it)."""
    try:
        client.heartbeat(worker_id, unit_id, progress)
        return True
    except Exception as exc:
        logger.warning("Heartbeat failed for unit %s (will retry next interval): %s", unit_id, exc)
        return False


def _process_assigned_unit(
    client: Any,
    worker_id: str,
    unit: dict[str, Any],
    *,
    artifacts_root: str | Path,
    agent_factory: Optional[AgentFactory],
    heartbeat_interval_seconds: float,
) -> bool:
    """Run one assigned unit end-to-end (heartbeat thread + episode + upload) or report
    it failed. Returns True on success, False on a *reported* failure. A unit failure
    never raises, so a long-running worker keeps going; the coordinator's attempt cap
    stops poison units. Safe to call from many threads at once: all per-unit state
    (heartbeat event, progress counter) is local to the call."""
    unit_id = unit["unit_id"]
    stop_heartbeat = threading.Event()
    progress = ProgressCounter()

    def heartbeat_loop() -> None:
        while not stop_heartbeat.wait(heartbeat_interval_seconds):
            _safe_heartbeat(client, worker_id, unit_id, progress.count)

    hb = threading.Thread(target=heartbeat_loop, daemon=True)
    try:
        client.heartbeat(worker_id, unit_id)
        hb.start()
        run_assigned_unit(
            unit, artifacts_root=artifacts_root, agent_factory=agent_factory, progress=progress
        )
        archive_path = package_run_archive(unit, artifacts_root=artifacts_root)
        client.upload(worker_id, unit_id, archive_path)
        return True
    except Exception as exc:
        client.fail(worker_id, unit_id, "".join(traceback.format_exception_only(type(exc), exc)).strip())
        logger.warning("worker %s: unit %s failed, continuing: %s", worker_id, unit_id, exc)
        return False
    finally:
        stop_heartbeat.set()
        hb.join(timeout=1.0)


def run_worker_loop(
    *,
    coordinator_url: str,
    artifacts_root: str | Path,
    capabilities: dict[str, Any],
    worker_state_path: str | Path,
    poll_interval_seconds: float = 10.0,
    heartbeat_interval_seconds: float = 30.0,
    once: bool = False,
    agent_factory: Optional[AgentFactory] = None,
    concurrency: int = 1,
    drain: bool = False,
    client: Optional[Any] = None,
    process_fn: Optional[Callable[[Any, str, dict[str, Any]], bool]] = None,
) -> bool:
    """Poll the coordinator and run assigned units.

    With ``concurrency > 1`` up to that many units run in parallel, each in its own
    thread. For a *served*-vLLM worker (each unit's agent hits one shared vLLM server)
    this lets the server continuously batch the concurrent prefills — the whole point,
    since the multimodal prompts are prefill-bound at batch size 1. The in-process
    offline ``LLM`` is NOT safe to drive from multiple threads, so concurrency>1 is
    only correct with the served (``qwen_vllm_api``) agent.

    ``client`` and ``process_fn`` are injection points for tests.
    """
    client = client or CoordinatorClient(coordinator_url)
    state_file = Path(worker_state_path)
    local_state = _load_worker_state(state_file)
    registration = client.register(
        {
            "worker_id": local_state.get("worker_id"),
            "hostname": socket.gethostname(),
            "capabilities": capabilities,
        }
    )
    worker_id = registration["worker_id"]
    local_state["worker_id"] = worker_id
    _write_json_atomic(state_file, local_state)

    def _process(unit: dict[str, Any]) -> bool:
        if process_fn is not None:
            return process_fn(client, worker_id, unit)
        return _process_assigned_unit(
            client, worker_id, unit, artifacts_root=artifacts_root,
            agent_factory=agent_factory, heartbeat_interval_seconds=heartbeat_interval_seconds,
        )

    def _next_unit() -> Optional[dict[str, Any]]:
        try:
            return client.assign(worker_id, capabilities).get("unit")
        except Exception as exc:
            # A transient coordinator outage must not kill a long-running worker.
            logger.warning(
                "worker %s: assign failed, retrying in %ss: %s",
                worker_id, poll_interval_seconds, exc,
            )
            return None

    if concurrency <= 1:
        while True:
            unit = _next_unit()
            if not unit:
                if once or drain:
                    return False
                time.sleep(poll_interval_seconds)
                continue
            ok = _process(unit)
            if once:
                return ok

    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    executor = ThreadPoolExecutor(max_workers=concurrency)
    in_flight: set = set()
    any_done = False
    try:
        while True:
            while len(in_flight) < concurrency:
                unit = _next_unit()
                if not unit:
                    break
                in_flight.add(executor.submit(_process, unit))
            if not in_flight:
                if once or drain:
                    return any_done
                time.sleep(poll_interval_seconds)
                continue
            done, pending = wait(in_flight, timeout=poll_interval_seconds, return_when=FIRST_COMPLETED)
            in_flight = set(pending)
            if done:
                any_done = True
                if once:
                    return True
    finally:
        executor.shutdown(wait=True)


def _api_client_groups(
    plan: dict[str, Any],
    state: dict[str, Any],
    model_group: Optional[str] = None,
) -> list[str]:
    groups = []
    for unit in plan["units"]:
        if str(unit.get("provider") or "").lower() not in {"claude", "kimi"}:
            continue
        if model_group is not None and unit.get("model_group") != model_group:
            continue
        unit_state = state["units"].get(unit["unit_id"], {})
        if unit_state.get("status") not in {"pending", "stale", "assigned", "running"}:
            continue
        groups.append(str(unit["model_group"]))
    return sorted(set(groups))


def run_coordinator_api_client(
    *,
    artifacts_root: str | Path,
    model_group: Optional[str] = None,
    max_units: int = 1,
    client_artifacts_root: Optional[str | Path] = None,
    agent_factory: Optional[AgentFactory] = None,
) -> dict[str, Any]:
    """Run pending Claude/Kimi units locally from the coordinator VM.

    This is for API-backed models that do not need a GPU worker. It still uses
    the same assignment, local-run, archive, upload, and verification lifecycle
    as an external worker client.
    """
    artifacts_root = Path(artifacts_root)
    store = CoordinatorStore(artifacts_root)
    plan = store.load_plan()
    state = store.load_state()
    groups = _api_client_groups(plan, state, model_group=model_group)
    if not groups:
        return {"completed": 0, "worker_id": None, "groups": []}

    worker_id = f"coordinator_api_{uuid.uuid4().hex[:12]}"
    capabilities = {
        "model_groups": groups,
        "hardware_profile": "coordinator-api",
        "worker_tags": ["api-client", "coordinator-local"],
    }
    store.register(
        {
            "worker_id": worker_id,
            "hostname": socket.gethostname(),
            "capabilities": capabilities,
        }
    )

    completed = 0
    failures: list[dict[str, Any]] = []
    # max_units <= 0 means "drain all pending units" (the value the launch-command
    # printer emits); a positive value caps how many this invocation runs.
    unlimited = int(max_units) <= 0
    limit = int(max_units)
    worker_root = Path(client_artifacts_root) if client_artifacts_root else (
        artifacts_root / DISTRIBUTED_DIR / "local_api_clients" / worker_id
    )

    while unlimited or completed < limit:
        assignment = store.assign(worker_id, capabilities)
        unit = assignment.get("unit")
        if not unit:
            break
        unit_id = unit["unit_id"]
        try:
            store.heartbeat(worker_id, unit_id)
            run_assigned_unit(unit, artifacts_root=worker_root, agent_factory=agent_factory)
            archive_path = package_run_archive(unit, artifacts_root=worker_root)
            store.upload(worker_id, unit_id, archive_path.read_bytes())
            completed += 1
        except Exception as exc:
            # Record the failure and move on: one transient API error must not abort
            # the remaining units. The coordinator's attempt cap stops poison units
            # from being re-handed out forever.
            reason = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            store.fail(worker_id, unit_id, reason)
            failures.append({"unit_id": unit_id, "reason": reason})

    return {
        "completed": completed,
        "worker_id": worker_id,
        "groups": groups,
        "failures": failures,
    }


def _upload_archive(client: Any, worker_id: str, unit_id: str, archive_path: Path) -> dict[str, Any]:
    """Upload a run archive, bridging the two transports.

    ``CoordinatorStore.upload`` (in-process / on the coordinator VM) takes archive
    *bytes*; ``CoordinatorClient.upload`` (HTTP) takes the archive *path* and reads
    the bytes itself. Mirror ``run_coordinator_api_client`` vs ``_process_assigned_unit``,
    which already split this way, so one worker function serves both."""
    if isinstance(client, CoordinatorStore):
        return client.upload(worker_id, unit_id, Path(archive_path).read_bytes())
    return client.upload(worker_id, unit_id, archive_path)


def run_lockstep_worker(
    *,
    artifacts_root: str | Path,
    capabilities: dict[str, Any],
    max_batches: int,
    coordinator_url: Optional[str] = None,
    worker_state_path: Optional[str | Path] = None,
    agent_factory: Optional[AgentFactory] = None,
    client: Optional[Any] = None,
    round_deadline_s: float = 7200.0,
) -> dict[str, Any]:
    """Drive the batch-API lockstep runner as a distributed worker role.

    One lockstep worker serves exactly ONE model group (its ``model_key`` is
    pinned to the first assigned unit's; a mismatching assignment is failed back).
    It builds a single ``LockstepBatchRunner`` over that group's agent and keeps
    the working set (<= ``max_batches``) topped up by ``client.assign`` refills.

    Per round the ``on_round`` hook heartbeats every still-active unit
    (progress = the unit's stepper ``query_count``) AND finalizes any unit that
    completed that round: a normal result is written (episode.json /
    run_inputs.json /run_score.json) through the same scoring path as
    ``run_assigned_unit`` and uploaded; an ``{"error": ...}`` result is
    ``client.fail``ed. Finalizing mid-run (not after ``run()``) frees the
    coordinator's per-worker concurrency slot so refill hands out fresh work,
    keeping the batch saturated.

    ``worker_concurrency`` in ``capabilities`` MUST equal ``max_batches`` so the
    coordinator's assign never re-hands an already-held unit as new work.
    """
    from interface.batch_runner import LockstepBatchRunner, LockstepUnit
    from interface.config import ExperimentConfig
    from interface.episode_checkpoint import resume_stepper
    from interface.episode_log import flush_episode_log
    from interface.episode_step import EpisodeStepper
    from pipeline.run_stage3 import build_episode_runner

    artifacts_root = Path(artifacts_root)
    client = client or CoordinatorClient(coordinator_url)
    max_batches = max(int(max_batches), 1)
    capabilities = {**capabilities, "worker_concurrency": max_batches}
    factory = agent_factory or pipeline._build_agent_from_spec

    # Restore/register a stable worker id (so re-assigned units stay ours).
    state_file = Path(worker_state_path) if worker_state_path else None
    local_state = _load_worker_state(state_file) if state_file else {}
    registration = client.register(
        {
            "worker_id": local_state.get("worker_id"),
            "hostname": socket.gethostname(),
            "capabilities": capabilities,
        }
    )
    worker_id = registration["worker_id"]
    if state_file is not None:
        local_state["worker_id"] = worker_id
        _write_json_atomic(state_file, local_state)

    handed_out: dict[str, dict[str, Any]] = {}   # unit_id -> assign payload
    lockstep_units: dict[str, Any] = {}          # unit_id -> LockstepUnit
    preps: dict[str, Any] = {}                    # unit_id -> _PreparedUnitRun
    finalized: set[str] = set()
    completed: list[str] = []
    failed: list[str] = []
    group_model_key: dict[str, Optional[str]] = {"key": None}
    agent_box: dict[str, Any] = {}
    runner_box: dict[str, Any] = {}

    def _build_unit(unit: dict[str, Any]):
        unit_id = unit["unit_id"]
        model_key = unit["model_key"]
        if group_model_key["key"] is None:
            group_model_key["key"] = model_key
            agent_box["agent"], _ = factory(model_key, unit["model_config"])
        elif model_key != group_model_key["key"]:
            # One lockstep worker serves exactly one model group/key. A unit of a
            # different key would need a different agent, so fail it back to the
            # coordinator (a group filter should make this unreachable).
            _safe_fail(
                client, worker_id, unit_id,
                f"lockstep worker pinned to model_key {group_model_key['key']!r}; "
                f"refused {model_key!r}",
            )
            return None

        row = materialize_worker_inputs(unit, artifacts_root)
        prep = pipeline._prepare_unit_run(
            row,
            unit["model_id"],
            model_config=unit["model_config"],
            manifest_path=artifacts_root / DISTRIBUTED_DIR / "worker_manifest.json",
            artifacts_root=artifacts_root,
            scored_static=unit["task_artifacts"]["scored_static"],
            difficulty_max=float(unit["difficulty_max_static_score"]),
            config=ScorerConfig.from_dict(unit["scorer_config"]),
            seed=int(unit["seed"]),
            prompt_variant=str(unit["prompt_variant"]),
            experiment_config=ExperimentConfig.from_dict(unit["experiment_config"]),
            conditions=unit.get("condition_set"),
        )
        # The run dir must exist before the runner writes its per-round checkpoint.
        prep.run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = prep.run_dir / "checkpoint.json"
        runner = build_episode_runner(
            prep.source, prep.experiment_config, int(unit["seed"]),
            max_steps=prep.runtime_spec.max_steps,
        )
        if checkpoint_path.exists():
            # Crash resume: rebuild the stepper at the query boundary the previous
            # attempt checkpointed instead of re-running from scratch.
            stepper = resume_stepper(checkpoint_path, runner=runner)
            stepper.maze_path = str(prep.source)
        else:
            stepper = EpisodeStepper(runner, maze_path=str(prep.source))
        lsu = LockstepUnit(unit_id, stepper, checkpoint_path=checkpoint_path)
        handed_out[unit_id] = unit
        lockstep_units[unit_id] = lsu
        preps[unit_id] = prep
        finalized.discard(unit_id)  # allow a re-assigned (retried) unit to re-run
        return lsu

    def _finalize_unit(unit_id: str, result: dict[str, Any]) -> None:
        unit = handed_out[unit_id]
        try:
            if isinstance(result, dict) and "error" in result:
                _safe_fail(client, worker_id, unit_id, str(result["error"]))
                failed.append(unit_id)
                return
            prep = preps[unit_id]
            flush_episode_log(result, prep.run_dir)          # episode.json (+frames)
            prep.write_run_inputs(extras={"pricing_tier": "batch"})
            episode = json.loads(prep.episode_path.read_text(encoding="utf-8"))
            prep.score_episode(episode)                      # run_score.json
            archive_path = package_run_archive(unit, artifacts_root=artifacts_root)
            _upload_archive(client, worker_id, unit_id, archive_path)
            completed.append(unit_id)
        except Exception as exc:  # noqa: BLE001 - one bad unit must not kill the batch
            logger.warning("lockstep worker %s: finalize unit %s failed: %s", worker_id, unit_id, exc)
            _safe_fail(
                client, worker_id, unit_id,
                "".join(traceback.format_exception_only(type(exc), exc)).strip(),
            )
            failed.append(unit_id)

    def refill():
        try:
            unit = client.assign(worker_id, capabilities).get("unit")
        except Exception as exc:
            logger.warning("lockstep worker %s: assign failed: %s", worker_id, exc)
            return None
        if not unit:
            return None
        unit_id = unit["unit_id"]
        # A unit id we still hold active (not yet finalized) is the coordinator's
        # concurrency-cap re-return, not new work — don't add it twice.
        if unit_id in handed_out and unit_id not in finalized:
            return None
        return _build_unit(unit)

    def on_round(active) -> None:
        active_ids = {u.unit_id for u in active}
        for u in active:
            _safe_heartbeat(client, worker_id, u.unit_id, getattr(u.stepper, "query_count", 0))
        # Finalize units that dropped out of the working set this round. Their
        # results are already in the runner's results map (populated before this
        # hook fires), including the ``{"error": ...}`` shape for failed steppers.
        results = getattr(runner_box.get("runner"), "_results", {})
        for unit_id in list(handed_out):
            if unit_id in active_ids or unit_id in finalized:
                continue
            result = results.get(unit_id)
            if result is None:
                continue
            finalized.add(unit_id)
            _finalize_unit(unit_id, result)

    first_unit = None
    try:
        first_unit = client.assign(worker_id, capabilities).get("unit")
    except Exception as exc:
        logger.warning("lockstep worker %s: initial assign failed: %s", worker_id, exc)
    if not first_unit:
        return {"worker_id": worker_id, "model_key": None, "completed": 0, "failed": []}

    first_lsu = _build_unit(first_unit)
    if first_lsu is None:
        return {"worker_id": worker_id, "model_key": group_model_key["key"], "completed": 0, "failed": failed}

    runner = LockstepBatchRunner(
        agent_box["agent"],
        max_batches=max_batches,
        round_deadline_s=round_deadline_s,
        refill=refill,
        on_round=on_round,
    )
    runner_box["runner"] = runner
    runner.add(first_lsu)
    runner.run()

    return {
        "worker_id": worker_id,
        "model_key": group_model_key["key"],
        "completed": len(completed),
        "failed": failed,
    }


def _safe_fail(client: Any, worker_id: str, unit_id: str, reason: str) -> bool:
    """``client.fail`` swallowing transport errors so a coordinator blip cannot
    abort the in-flight batch (the unit will go stale and be re-handed anyway)."""
    try:
        client.fail(worker_id, unit_id, reason)
        return True
    except Exception as exc:
        logger.warning("Fail report failed for unit %s: %s", unit_id, exc)
        return False


def _validate_final_run_files(unit: dict[str, Any], artifacts_root: Path) -> list[str]:
    run_dir = artifacts_root / unit["run_dir"]
    return [
        name
        for name in unit.get("expected_files") or EXPECTED_RUN_FILES
        if not (run_dir / name).exists()
    ]


def finalize_job(
    *,
    artifacts_root: str | Path,
    run_set_id: Optional[str] = None,
    allow_partial: bool = False,
    storage: Optional[StorageConfig] = None,
) -> dict[str, Any]:
    artifacts_root = Path(artifacts_root)
    store = CoordinatorStore(artifacts_root, storage=storage)
    plan = store.load_plan()
    state = store.load_state()
    config = ScorerConfig.from_dict(plan["scorer_config"])
    difficulty_max = float(plan["difficulty_max_static_score"])

    missing: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    composites: dict[tuple, Optional[float]] = {}

    for unit in plan["units"]:
        missing_files = _validate_final_run_files(unit, artifacts_root)
        unit_state = state["units"].setdefault(unit["unit_id"], {})
        if missing_files:
            missing.append({"unit_id": unit["unit_id"], "missing_files": missing_files})
            continue

        task_id = unit["task_id"]
        run_dir = artifacts_root / unit["run_dir"]
        episode = _read_json(run_dir / "episode.json")
        canonical = _read_json(artifacts_root / "tasks" / task_id / "canonical_paths.json")
        static_score = _read_json(artifacts_root / "tasks" / task_id / "scored_static.json")
        manifest_row = dict(unit["task_row"])
        metrics = episode_metrics.build_metrics(episode, canonical, manifest_row)
        enriched = episode_metrics.enrich_run_for_scoring(
            episode,
            manifest_row,
            agent_or_model=unit["model_id"],
            seed=int(unit["seed"]),
            metrics=metrics,
        )
        run_score = compute_runtime_score(
            enriched,
            static_score=static_score,
            canonical_paths=canonical,
            config=config,
            difficulty_max_static_score=difficulty_max,
        ).to_dict()
        (run_dir / "run_score.json").write_text(json.dumps(run_score, indent=2), encoding="utf-8")
        run_row = episode_metrics.build_run_row(
            episode,
            canonical,
            manifest_row,
            agent_or_model=unit["model_id"],
            seed=int(unit["seed"]),
            raw_output_ref=str((run_dir / "episode.json").relative_to(artifacts_root)),
            metrics=metrics,
            prompt_variant=str(unit["prompt_variant"]),
        )
        run_rows.append(run_row)
        composites[
            (
                task_id,
                unit["model_id"],
                int(unit["seed"]),
                manifest_row.get("condition"),
                unit["prompt_variant"],
            )
        ] = run_score.get("composite")
        unit_state.update({"status": "verified", "finalized_at": _iso(), "updated_at": _iso()})

    if missing and not allow_partial:
        raise RuntimeError(f"Missing distributed work units: {missing}")

    run_set = run_set_id or plan["run_set_id"]
    payloads = pipeline._write_aggregate(
        run_rows,
        composites,
        plan["static_by_task"],
        plan["tasks"],
        artifacts_root,
        run_set,
    )
    store.save_state(state)

    if storage and storage.active:
        # Re-push any per-unit runs whose live mirror failed, then mirror the
        # aggregate outputs, so the bucket holds a complete, durable copy.
        for unit in plan["units"]:
            unit_state = state["units"].get(unit["unit_id"], {})
            if unit_state.get("status") == "verified" and unit_state.get("gcs_pending"):
                store._mirror_unit(unit["unit_id"], unit["run_dir"], run_set)
        jsonl = artifacts_root / "episode_runs.jsonl"
        if jsonl.exists():
            mirror_to_bucket(
                jsonl, gcs_destination(storage.bucket, run_set, "episode_runs.jsonl"),
                gsutil=storage.gsutil,
            )
        report_dir = artifacts_root / "reports" / run_set
        if report_dir.exists():
            mirror_to_bucket(
                report_dir, gcs_destination(storage.bucket, run_set, "reports", run_set),
                gsutil=storage.gsutil,
            )

    return {
        "job_id": plan["job_id"],
        "run_count": len(run_rows),
        "missing_units": missing,
        "payloads": payloads,
    }


def _load_storage(args: Any) -> Optional[StorageConfig]:
    path = getattr(args, "storage_config", None)
    if not path:
        return None
    storage = StorageConfig.from_file(path)
    if not storage.active:
        logger.warning("Storage config %s has no bucket / is disabled; bucket mirroring is OFF.", path)
    return storage


def dispatch_distributed_role(args: Any) -> None:
    role = args.distributed_role
    if role == "coordinator-prepare":
        if not args.run_config:
            raise SystemExit("--run-config is required for coordinator-prepare.")
        plan = prepare_job(
            run_config_path=args.run_config,
            manifest_path=args.manifest,
            seeds=[int(s) for s in args.seeds],
            conditions=args.conditions,
            prompt_variant=args.prompt_variant,
            artifacts_root=args.artifacts_root,
            run_set_id=args.run_set_id,
            difficulty_max_static_score=args.difficulty_max_static_score,
            force=args.force,
            job_id=args.job_id,
        )
        print(
            f"Prepared distributed job {plan['job_id']}: {len(plan['units'])} units "
            f"-> {plan_path(args.artifacts_root)}"
        )
        return
    if role == "coordinator-serve":
        serve_coordinator(
            args.artifacts_root,
            host=args.host,
            port=args.port,
            stale_after_seconds=args.stale_after_seconds,
            storage=_load_storage(args),
        )
        return
    if role == "worker":
        if not args.coordinator_url:
            raise SystemExit("--coordinator-url is required for worker.")
        state_file = args.worker_state or str(Path(args.artifacts_root) / DISTRIBUTED_DIR / "worker_state.json")
        capabilities = {
            "model_group": args.model_group,
            "hardware_profile": args.hardware_profile,
            "worker_tags": args.worker_tag or [],
            "local_model_cache": args.local_model_cache or [],
            "worker_concurrency": int(getattr(args, "worker_concurrency", 1) or 1),
        }
        completed = run_worker_loop(
            coordinator_url=args.coordinator_url,
            artifacts_root=args.artifacts_root,
            capabilities=capabilities,
            worker_state_path=state_file,
            poll_interval_seconds=args.poll_interval_seconds,
            heartbeat_interval_seconds=args.heartbeat_interval_seconds,
            once=args.once,
            concurrency=int(getattr(args, "worker_concurrency", 1) or 1),
        )
        print(f"Worker {'completed one unit' if completed else 'found no unit'}.")
        return
    if role == "lockstep-worker":
        if not args.coordinator_url:
            raise SystemExit("--coordinator-url is required for lockstep-worker.")
        state_file = args.worker_state or str(
            Path(args.artifacts_root) / DISTRIBUTED_DIR / "lockstep_worker_state.json"
        )
        # worker_concurrency == MAX_BATCHES for the lockstep runner's working set.
        max_batches = int(getattr(args, "worker_concurrency", 1) or 1)
        capabilities = {
            "model_group": args.model_group,
            "hardware_profile": args.hardware_profile,
            "worker_tags": args.worker_tag or [],
            "worker_concurrency": max_batches,
        }
        result = run_lockstep_worker(
            coordinator_url=args.coordinator_url,
            artifacts_root=args.artifacts_root,
            capabilities=capabilities,
            max_batches=max_batches,
            worker_state_path=state_file,
        )
        print(
            f"Lockstep worker completed {result['completed']} unit(s) "
            f"({len(result['failed'])} failed) for group={args.model_group}."
        )
        return
    if role == "coordinator-run-api-client":
        result = run_coordinator_api_client(
            artifacts_root=args.artifacts_root,
            model_group=args.model_group,
            max_units=args.max_units,
            client_artifacts_root=args.client_artifacts_root,
        )
        print(
            f"Coordinator API client completed {result['completed']} unit(s) "
            f"for groups={result['groups']}."
        )
        return
    if role == "coordinator-finalize":
        result = finalize_job(
            artifacts_root=args.artifacts_root,
            run_set_id=args.run_set_id,
            allow_partial=args.allow_partial_finalize,
            storage=_load_storage(args),
        )
        print(
            f"Finalized distributed job {result['job_id']}: {result['run_count']} runs, "
            f"{len(result['missing_units'])} missing units."
        )
        return
    raise SystemExit(f"Unknown distributed role: {role}")
