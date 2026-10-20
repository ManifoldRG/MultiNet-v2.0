"""The mechanics-15 panel: one maze per non-empty combination of the four mechanics."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_MANIFEST = json.loads((_REPO / "gridworld" / "fixtures" / "manifest.mechanics_15.json").read_text())
_MECHANICS = ("portals", "death", "rotating", "frozen")


def test_one_maze_per_mechanic_combination():
    combos = {frozenset(t["new_mechanisms"]) for t in _MANIFEST["tasks"]}
    expected = {
        frozenset(c) for n in range(1, 5) for c in itertools.combinations(_MECHANICS, n)
    }
    assert len(_MANIFEST["tasks"]) == 15
    assert combos == expected


def test_rows_are_unique_and_point_into_the_ogbench_submodule():
    ids = [t["task_id"] for t in _MANIFEST["tasks"]]
    assert len(set(ids)) == len(ids)
    for t in _MANIFEST["tasks"]:
        assert t["source"].startswith("ogbench/ogbench/procgen/maze_jsons/M")
        assert (_REPO / t["source"]).is_file(), t["source"]
        assert t["rank"] == ids.index(t["task_id"]) + 1


def test_no_render_or_backend_in_rows():
    """Domain (2D / NL / 3D) is the run config's choice, so one manifest serves all three."""
    for t in _MANIFEST["tasks"]:
        assert not {"render", "backend", "observation"} & set(t)
