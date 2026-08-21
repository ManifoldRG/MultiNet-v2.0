"""R1 task allowlist helpers shared by desktop UI and web API."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from demo.compare import R1ResultCatalog, r1_task_id
from demo.r1_config import DEFAULT_EXPERIMENT, DEFAULT_MANIFEST, R1_CONFIG
from demo.session import MiniGridPlaySession, load_manifest_tasks

_REPO_ROOT = Path(__file__).resolve().parent.parent


def canonical_task_id(session: MiniGridPlaySession) -> str:
    row = session.manifest_row_by_path.get(session.task_path)
    if row is not None:
        return row["task_id"]
    return r1_task_id(session.task_path)


def restrict_to_r1_tasks(
    session: MiniGridPlaySession,
    catalog: R1ResultCatalog | None = None,
) -> None:
    """Narrow browsing to mazes present in the R1 results table, when any of
    the current selection is. If none are -- a non-R1 experiment/directory
    was requested, or no results table is available at all -- leave the
    task list alone; R1-comparison is just unavailable for this session
    rather than a reason to refuse to launch."""
    catalog = catalog or R1ResultCatalog()
    r1_tasks = [p for p in session.task_list if r1_task_id(p) in catalog]
    if not r1_tasks:
        return
    session.task_list = r1_tasks
    session.task_list_locked = True
    if session.task_path not in r1_tasks:
        session._load_task(str(r1_tasks[0]))
    else:
        session.task_index = r1_tasks.index(session.task_path)


def resolve_r1_task_path(
    task_id: str,
    *,
    manifest: str = DEFAULT_MANIFEST,
    experiment: str = DEFAULT_EXPERIMENT,
    catalog: R1ResultCatalog | None = None,
) -> Path:
    catalog = catalog or R1ResultCatalog()
    manifest_path = _REPO_ROOT / manifest
    for path, row in load_manifest_tasks(manifest_path, experiment):
        if row["task_id"] == task_id:
            if task_id not in catalog:
                raise KeyError(f"Task {task_id!r} is not in the R1 results table")
            return path
    raise KeyError(f"Unknown R1 taskId {task_id!r}")


def create_r1_play_session(
    task_id: str,
    *,
    catalog: R1ResultCatalog | None = None,
) -> MiniGridPlaySession:
    catalog = catalog or R1ResultCatalog()
    path = resolve_r1_task_path(task_id, catalog=catalog)
    session = MiniGridPlaySession(
        task_path=str(path),
        config=dataclasses.replace(R1_CONFIG),
        manifest=DEFAULT_MANIFEST,
        experiment=DEFAULT_EXPERIMENT,
    )
    restrict_to_r1_tasks(session, catalog)
    return session


def list_r1_tasks(catalog: R1ResultCatalog | None = None) -> list[dict]:
    catalog = catalog or R1ResultCatalog()
    out = []
    for path, row in load_manifest_tasks(_REPO_ROOT / DEFAULT_MANIFEST, DEFAULT_EXPERIMENT):
        tid = row["task_id"]
        if tid not in catalog:
            continue
        comparison = catalog.lookup(tid)
        out.append(
            {
                "taskId": tid,
                "optimalSteps": comparison.optimal_steps,
                "gridSize": comparison.grid_size,
                "mazeCategory": row.get("maze_category"),
            }
        )
    return out
