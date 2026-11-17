"""Neutral tile labels on the R3 mazes (M11-M14, PR #88): portal chains.

An R3 portal chain is one portal with 3+ pads. Under neutral labels every pad
is listed under the maze's portal label, which pad leads where stays hidden,
and the episode records the chain for grading.
"""

from __future__ import annotations

import json
from pathlib import Path

from gridworld.baselines import plan_bfs_path
from gridworld.task_spec import TaskSpecification
from interface.config import ExperimentConfig
from interface.coords import to_row_col
from interface.parser import ACTION_ORDER
from interface.renderer import render_initial_maze_text
from interface.tile_labels import TileLabeling
from tests.test_tile_labels import _leaks, _run, _texts

REPO = Path(__file__).resolve().parents[1]
R3_RUN_CONFIG = json.loads((REPO / "gridworld" / "fixtures" / "run_config.r3_nl.json").read_text())
R3_TASKS = json.loads((REPO / "gridworld" / "fixtures" / "manifest.r3.json").read_text())["tasks"]
# The R3 NL cell, as named (R3) and as neutral.
NAMED_R3 = ExperimentConfig(**R3_RUN_CONFIG["experiment_config"])
NEUTRAL_R3 = ExperimentConfig(**R3_RUN_CONFIG["experiment_config"], tile_labels="neutral")
CHAIN_MAZE = next(t for t in R3_TASKS if t["task_id"].endswith("planning_portal_chain_0"))


def _spec(task) -> TaskSpecification:
    return TaskSpecification.from_json(str(REPO / task["source"]))


def test_r3_portal_chain_lists_every_pad_under_one_label_and_records_the_chain():
    spec = _spec(CHAIN_MAZE)
    chain = next(t for t in spec.mechanisms.teleporters if len(t.ends()) > 2)
    pads = [to_row_col(p) for p in chain.ends()]
    labeling = TileLabeling.for_task(spec, 0)
    label = labeling.label_by_kind["portal"]
    text = render_initial_maze_text(spec, labeling=labeling)
    for row, col in pads:
        assert f"There is a {label} tile at ({row},{col})." in text
    assert set(pads) <= set(labeling.cells_by_kind["portal"])

    links = labeling.to_dict()["portal_links"]
    assert len(links) == len(spec.mechanisms.teleporters)
    chained = [link for link in links if "pads" in link]
    assert chained == [
        {
            "label": label,
            "spec_color": chain.color,
            "pads": [list(p) for p in pads],
            "bidirectional": chain.bidirectional,
        }
    ]
    # Two-pad portals are recorded exactly as before.
    for link, teleporter in zip(links, spec.mechanisms.teleporters):
        if len(teleporter.ends()) == 2:
            assert set(link) == {"label", "spec_color", "a", "b", "bidirectional"}


def test_r3_portal_chain_route_under_neutral_labels():
    """The planner's route rides the chain; replayed through the NL pipeline,
    neutral and named walk the same states, a chain ride reads MOVED (r,c),
    and no request names a tile's effect."""
    spec = _spec(CHAIN_MAZE)
    plan = plan_bfs_path(spec)
    assert plan.success
    route = [ACTION_ORDER[a] for a in plan.actions]
    chain = next(t for t in spec.mechanisms.teleporters if len(t.ends()) > 2)
    middle = chain.ends()[1].to_tuple()
    assert middle in [tuple(p) for p in plan.positions]  # the route uses the middle pad

    named, named_requests = _run(spec, NAMED_R3, route)
    neutral, requests = _run(spec, NEUTRAL_R3, route)
    for result in (named, neutral):
        assert (result["end_reason"], result["steps_used"]) == ("success", len(route))
    n_steps = [r for r in named["transcript"] if r.get("kind") == "step"]
    x_steps = [r for r in neutral["transcript"] if r.get("kind") == "step"]
    assert [s["state_after"] for s in x_steps] == [s["state_after"] for s in n_steps]
    assert [s["event_type"] for s in x_steps] == [s["event_type"] for s in n_steps]

    teleports = [
        (named_step["prompt_feedback"], step["prompt_feedback"])
        for named_step, step in zip(n_steps, x_steps)
        if step["event_type"] == "TELEPORTED"
    ]
    assert teleports
    for named_text, neutral_text in teleports:
        assert named_text.startswith("TELEPORTED (")
        assert neutral_text == "MOVED " + named_text[len("TELEPORTED "):]

    assert _leaks(_texts(named_requests)) >= {"teleport", "cyan"}
    assert not _leaks(_texts(requests)), _leaks(_texts(requests))
    # Every pad of the chain is in the first prompt under the portal label.
    label = neutral["tile_labels"]["portal_links"][0]["label"]
    first = _texts(requests)[0] + _texts(requests)[1]
    for row, col in (to_row_col(p) for p in chain.ends()):
        assert f"There is a {label} tile at ({row},{col})." in first
