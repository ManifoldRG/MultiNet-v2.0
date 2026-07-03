"""Held-out blind (B-family) probe maze invariants.

The blind probe is the conditional-eval's held-out B maze: all keys, doors, and
switches render white, so the agent must reason from state changes rather than
color. Distractors are positional/consumption decoys (the engine matches keys by
color, so all-white keys are interchangeable). We lock that the maze is beatable
and that the BFS-optimal solution ignores the decoy key and decoy switch.
"""

from __future__ import annotations

import json
from pathlib import Path

from gridworld.baselines import plan_bfs_path
from gridworld.task_spec import TaskSpecification
from scorer.scoring import compute_canonical_paths

_REPO = Path(__file__).resolve().parents[1]
_MAZE = _REPO / "mazes" / "conditional" / "blind_probe_B_holdout.json"


def _spec() -> TaskSpecification:
    return TaskSpecification.from_dict(json.loads(_MAZE.read_text()))


def test_blind_probe_mechanism_counts_and_uniform_color():
    payload = json.loads(_MAZE.read_text())
    mech = payload["mechanisms"]
    assert len(mech["keys"]) == 2
    assert len(mech["doors"]) == 4
    assert len(mech["switches"]) == 2
    # Blind: uniform color within each mechanism type so nothing is
    # distinguishable. Keys/doors are grey (the MiniGrid palette has no white
    # key/door); switches are white. The point is intra-type uniformity.
    assert len({k["color"] for k in mech["keys"]}) == 1
    assert len({d["requires_key"] for d in mech["doors"]}) == 1
    assert len({s["color"] for s in mech["switches"]}) == 1
    assert all(k["color"] == "grey" for k in mech["keys"])
    assert all(d["requires_key"] == "grey" for d in mech["doors"])
    assert all(s["color"] == "white" for s in mech["switches"])


def test_blind_probe_is_beatable():
    report = compute_canonical_paths(_spec())
    assert report.success is True
    assert report.optimal_steps > 0


def test_blind_probe_optimal_path_ignores_the_decoys():
    bfs = plan_bfs_path(_spec())
    assert bfs.success is True
    labels = bfs.action_labels
    # The productive mechanisms are required...
    assert "toggle:s1" in labels
    assert "pickup:k1" in labels
    assert "open_door:D_goal" in labels
    # ...and the decoy key + decoy switch are NOT on the optimal solution.
    assert "pickup:k2" not in labels
    assert "toggle:s2" not in labels
