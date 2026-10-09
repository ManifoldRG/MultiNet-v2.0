"""Load gridworld tasks for the NLU interface."""

from __future__ import annotations

from pathlib import Path

from gridworld.backends import get_backend
from gridworld.backends.base import AbstractGridBackend
from gridworld.task_spec import TaskSpecification


def load_task(
    path: str | Path, backend: str = "minigrid", **backend_kwargs
) -> tuple[AbstractGridBackend, TaskSpecification]:
    spec = TaskSpecification.from_json(str(path))
    grid_backend = get_backend(backend, **backend_kwargs)
    grid_backend.configure(spec)
    return grid_backend, spec


def default_maze_path(name: str = "V01_empty_room.json") -> Path:
    return Path(__file__).resolve().parents[1] / "mazes" / "validation_10" / name
