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


def corpus_sample(step: int = 27) -> list[Path]:
    """Every ``step``-th maze of the submodule corpus (sorted; ~8 of 214 by default)."""
    from maze_test_utils import MAZE_JSON_DIR

    return sorted(MAZE_JSON_DIR.rglob("*.json"))[::step]


# 9x5 room with every supported mechanism: red key (2,1), red door (4,2)
# between spec walls (4,1)/(4,3), yellow switch s1 (2,3) -> grey gate g1 (6,2),
# goal (7,2). Agent starts at (1,2) facing EAST.
MECHANISMS = {
    "task_id": "render3d_mechanisms",
    "seed": 0,
    "difficulty_tier": 3,
    "maze": {"dimensions": [9, 5], "walls": [[4, 1], [4, 3]], "start": [1, 2], "goal": [7, 2]},
    "mechanisms": {
        "keys": [{"id": "k1", "position": [2, 1], "color": "red"}],
        "doors": [{"id": "d1", "position": [4, 2], "requires_key": "red"}],
        "switches": [{"id": "s1", "position": [2, 3], "controls": ["g1"], "color": "yellow"}],
        "gates": [{"id": "g1", "position": [6, 2]}],
    },
    "goal": {"type": "reach_position", "target": [7, 2]},
    "max_steps": 80,
}


def make_play_session(tmp_path, backend: str, camera: str | None = None, spec: dict = CORRIDOR):
    """Headless human-play demo session on ``spec`` (no pygame window)."""
    import dataclasses
    import json

    from demo.r1_config import R1_CONFIG
    from demo.session import MiniGridPlaySession

    path = tmp_path / f"{spec['task_id']}.json"
    path.write_text(json.dumps(spec))
    return MiniGridPlaySession(
        task_path=str(path),
        config=dataclasses.replace(R1_CONFIG),
        backend=backend,
        camera=camera,
    )


# 11x5 room with every PR #57 tile: purple portal (3,1)<->(7,3), cyan portal
# (3,3)<->(7,1), kill cell (5,1), frozen tile (5,3), rotating tiles (4,2)
# facing EAST and (6,2) facing NORTH (validate() rejects tiles on walls).
# Agent starts (1,2) facing EAST; goal (9,2).
TILES = {
    "task_id": "render3d_tiles",
    "seed": 0,
    "difficulty_tier": 3,
    "maze": {"dimensions": [11, 5], "walls": [], "start": [1, 2], "goal": [9, 2]},
    "mechanisms": {
        "teleporters": [
            {"id": "tp", "position_a": [3, 1], "position_b": [7, 3], "color": "purple"},
            {"id": "tc", "position_a": [3, 3], "position_b": [7, 1], "color": "cyan"},
        ],
        "death_portals": [[5, 1]],
        "frozen_tiles": [[5, 3]],
        "rotating_tiles": [[4, 2], [6, 2]],
        "rotating_initial_directions": [0, 3],
    },
    "goal": {"type": "reach_position", "target": [9, 2]},
    "max_steps": 80,
}


def new_mechanism_mazes() -> list[Path]:
    """Helen's M7-M10 mazes (ogbench PR #2) when the submodule has them."""
    from maze_test_utils import MAZE_JSON_DIR

    return sorted(p for tier in ("M7", "M8", "M9", "M10") for p in (MAZE_JSON_DIR / tier).glob("*.json"))
