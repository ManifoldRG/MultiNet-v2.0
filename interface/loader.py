"""Load gridworld tasks for the NLU interface."""

from __future__ import annotations

from pathlib import Path

from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.task_spec import TaskSpecification


def load_task(path: str | Path) -> tuple[MiniGridBackend, TaskSpecification]:
    spec = TaskSpecification.from_json(str(path))
    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(spec)
    return backend, spec


def default_maze_path(name: str = "V01_empty_room.json") -> Path:
    return Path(__file__).resolve().parents[1] / "mazes" / "validation_10" / name
