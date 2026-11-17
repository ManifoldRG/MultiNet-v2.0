"""Neutral tile labels on the R3 mazes (M11-M14, PR #88).

R3 adds portal chains (3+ pads), phased cells, disappearing floor, doors and
switches that shut, a pink key and gates that start open. Under neutral labels
none of them may vanish from what the model reads: on every R3 maze, the
neutral observation must show every cell the named observation shows, and must
not name what a neutral-labelled tile does.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pytest

from gridworld.actions import MiniGridActions
from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.baselines import plan_bfs_path
from gridworld.task_spec import TaskSpecification
from interface.config import ExperimentConfig
from interface.coords import to_row_col
from interface.parser import ACTION_ORDER
from interface.renderer import render_initial_maze_text, render_user_observation_text
from interface.tile_labels import TileLabeling
from tests.test_tile_labels import _leaks, _run, _texts

REPO = Path(__file__).resolve().parents[1]
R3_RUN_CONFIG = json.loads((REPO / "gridworld" / "fixtures" / "run_config.r3_nl.json").read_text())
R3_TASKS = json.loads((REPO / "gridworld" / "fixtures" / "manifest.r3.json").read_text())["tasks"]
# The R3 NL cell, as named (R3) and as neutral.
NAMED_R3 = ExperimentConfig(**R3_RUN_CONFIG["experiment_config"])
NEUTRAL_R3 = ExperimentConfig(**R3_RUN_CONFIG["experiment_config"], tile_labels="neutral")
CHAIN_MAZE = next(t for t in R3_TASKS if t["task_id"].endswith("planning_portal_chain_0"))
COORD = re.compile(r"\((\d+),(\d+)\)")


def _spec(task) -> TaskSpecification:
    return TaskSpecification.from_json(str(REPO / task["source"]))


def _coords(text: str) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in COORD.findall(text)}


def _walk(spec: TaskSpecification, n: int = 30, seed: int = 0):
    """The reset state and a seeded random walk from it (real runtime states:
    floor that disappeared, phase clocks ticking)."""
    backend = MiniGridBackend(render_mode=None)
    backend.configure(spec)
    _, state, _ = backend.reset(seed=spec.seed)
    states = [state]
    rng = random.Random(seed)
    moves = [int(MiniGridActions.MOVE_FORWARD)] * 2 + [
        int(MiniGridActions.TURN_LEFT),
        int(MiniGridActions.TURN_RIGHT),
    ]
    for _ in range(n):
        _, _, terminated, truncated, state, _ = backend.step(rng.choice(moves))
        states.append(state)
        if terminated or truncated:
            break
    return states


def test_r3_cell_is_the_r2_nl_cell_neutral_mode_allows():
    assert NEUTRAL_R3.tile_labels == "neutral" and NAMED_R3.tile_labels == "named"
    assert len(R3_TASKS) == 18


@pytest.mark.parametrize("task", R3_TASKS, ids=[t["task_id"] for t in R3_TASKS])
def test_r3_neutral_text_shows_every_cell_the_named_text_shows(task):
    spec = _spec(task)
    labeling = TileLabeling.for_task(spec, 0)

    named = render_initial_maze_text(spec)
    neutral = render_initial_maze_text(spec, labeling=labeling)
    assert _coords(neutral) == _coords(named), sorted(_coords(named) ^ _coords(neutral))
    named_texts, neutral_texts = [named], [neutral]

    for i, state in enumerate(_walk(spec)):
        named = render_user_observation_text(spec, state)
        neutral = render_user_observation_text(spec, state, labeling=labeling)
        missing = _coords(named) - _coords(neutral)
        assert not missing, f"state {i}: neutral text drops {sorted(missing)}"
        assert _coords(neutral) == _coords(named), f"state {i}"
        named_texts.append(named)
        neutral_texts.append(neutral)

    # The detector does see the R3 names (else "no leaks" proves nothing)...
    assert _leaks(named_texts) >= {"death", "frozen", "rotat", "teleport", "purple", "cyan"}
    # ...and the neutral text has none of them.
    assert not _leaks(neutral_texts), _leaks(neutral_texts)

    # The walk reached the R3 state the text has to carry.
    if spec.mechanisms.phased_patches:
        joined = "\n".join(neutral_texts)
        assert re.search(r"Cell \(\d+,\d+\) is open\.", joined)
        assert re.search(r"Cell \(\d+,\d+\) is closed\.", joined)
    if spec.rules.disappear_after != float("inf"):
        assert any("The following cells are walls" in t for t in neutral_texts[1:])


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
