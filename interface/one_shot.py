"""One-shot in-context learning example loader."""

from __future__ import annotations

import base64
import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from interface.observation import ObservationMode

_EXAMPLE_DIR = Path(__file__).parent.parent / "mazes" / "one_shot_example"
_SOLUTION_PATH = _EXAMPLE_DIR / "one_shot_example_solution.json"
_PNG_PATH = _EXAMPLE_DIR / "one_shot_example_14x14_dense_kr_sg_kb_2.png"
_JSON_PATH = _EXAMPLE_DIR / "one_shot_example_14x14_dense_kr_sg_kb_2.json"

_INTRO = (
    "Example maze and solution "
    "(14x14 maze with a red key-door, switch-gate, and blue key-door chain):\n"
)


@lru_cache(maxsize=1)
def _solution_str() -> str:
    data = json.loads(_SOLUTION_PATH.read_text())
    return ", ".join(data["actions"])


@lru_cache(maxsize=1)
def _png_b64() -> str:
    return base64.b64encode(_PNG_PATH.read_bytes()).decode("utf-8")


@lru_cache(maxsize=1)
def _maze_text() -> str:
    from gridworld.task_spec import TaskSpecification
    from interface.renderer import render_initial_maze_text

    spec = TaskSpecification.from_json(str(_JSON_PATH))
    return render_initial_maze_text(spec)


def one_shot_content_blocks(observation: "ObservationMode") -> list[dict]:
    """Return content blocks for the one-shot example to prepend to the user message."""
    solution_line = f"Actions to solve: {_solution_str()}"

    if observation == "text_only":
        text = f"{_INTRO}{_maze_text()}\n{solution_line}\n\n"
        return [{"type": "text", "text": text}]

    return [
        {"type": "text", "text": _INTRO},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_png_b64()}"}},
        {"type": "text", "text": f"\n{solution_line}\n\n"},
    ]
