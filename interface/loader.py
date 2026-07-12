"""Load gridworld tasks for the NLU interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gridworld.backends import get_backend
from gridworld.backends.base import AbstractGridBackend
from gridworld.task_spec import TaskSpecification


def load_task(
    path: str | Path,
    backend: str = "minigrid",
    **backend_kwargs: Any,
) -> tuple[AbstractGridBackend, TaskSpecification]:
    spec = TaskSpecification.from_json(str(path))
    backend_obj = get_backend(backend, render_mode="rgb_array", **backend_kwargs)
    backend_obj.configure(spec)
    return backend_obj, spec


def default_maze_path(name: str = "V01_empty_room.json") -> Path:
    return Path(__file__).resolve().parents[1] / "mazes" / "validation_10" / name
