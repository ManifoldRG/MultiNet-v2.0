"""Shared fixtures for the 3D render backend tests (imported as a top-level
module, like maze_test_utils)."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Prepended to subprocess code: any import of minigrid raises ImportError, so a
# module that still depends on minigrid fails loudly.
_MINIGRID_BLOCKER = '''
import importlib.abc, sys
class _BlockMinigrid(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "minigrid" or name.startswith("minigrid."):
            raise ImportError("minigrid blocked for import-boundary test: " + name)
sys.meta_path.insert(0, _BlockMinigrid())
'''


def run_with_minigrid_blocked(code: str) -> subprocess.CompletedProcess:
    """Run ``code`` in a fresh interpreter (repo root as cwd) with minigrid blocked."""
    return subprocess.run(
        [sys.executable, "-c", _MINIGRID_BLOCKER + textwrap.dedent(code)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )


# 7x3 corridor: start (1,1) facing EAST, red key (2,1), locked red door (3,1),
# goal (5,1). BFS plan: MOVE_FORWARD, PICKUP, TOGGLE, MOVE_FORWARD x3.
CORRIDOR = {
    "task_id": "render3d_corridor",
    "seed": 0,
    "difficulty_tier": 2,
    "maze": {"dimensions": [7, 3], "walls": [], "start": [1, 1], "goal": [5, 1]},
    "mechanisms": {
        "keys": [{"id": "k1", "position": [2, 1], "color": "red"}],
        "doors": [
            {"id": "d1", "position": [3, 1], "requires_key": "red", "initial_state": "locked"}
        ],
    },
    "goal": {"type": "reach_position", "target": [5, 1]},
    "max_steps": 40,
}
