"""Neutral tile labels (``tile_labels="neutral"``): the NL rule-learning arm.

Only names and feedback change, never the maze or the world model, so every
mechanics-15 maze must stay exactly as solvable as in R2. The planner's BFS
route is replayed through the NL pipeline on each maze: it must succeed in the
BFS step count, take the same states as the planner, record the label mapping,
and show the model no word that names a tile's effect or pairs its portal pads.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pytest

from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.baselines import plan_bfs_path
from gridworld.task_spec import TaskSpecification
from interface.config import ExperimentConfig
from interface.coords import to_row_col
from interface.feedback import format_step_feedback
from interface.parser import ACTION_ORDER
from interface.renderer import render_initial_maze_text, render_user_observation_text
from interface.runner import build_runner
from interface.tile_labels import KINDS, LABEL_POOL, TileLabeling
from scripts.run_pipeline import _expected_run_hash, _experiment_config_hash_payload
from tests.test_nl_default_prompts_golden import CELLS, MAZE, death_detour_route

REPO = Path(__file__).resolve().parents[1]
TASKS = json.loads((REPO / "gridworld" / "fixtures" / "manifest.mechanics_15.json").read_text())["tasks"]
NAMED = ExperimentConfig(**CELLS["r2_nl"])
NEUTRAL = ExperimentConfig(**CELLS["r2_nl"], tile_labels="neutral")

# Anything that says what a tile does, or that two pads belong together.
# Word-bounded where a short stem is part of ordinary words ("notice", "diet").
LEAK = re.compile(
    r"portal|death|\bdead|\bdie|frozen|freez|\bice\b|\bicy\b|frost|rotat|turntable|\bspin"
    r"|teleport|\bwarp|lava|\bkill|carr(?:y|ied)|\bride\b|\brode\b|riding|skull|hazard"
    r"|\btrap|purple|cyan|between|\bpair|linked|arrow",
    re.IGNORECASE,
)

# seed 0: maze task_id -> kind -> label. Changing LABEL_POOL or the draw changes
# what every maze is called; re-pin deliberately, never to make a test pass.
PINNED_SEED0 = {
    "18x18_dense_kr_sg_kb_portals_0": {"portal": "fep"},
    "18x18_dense_kr_sg_kb_death_0": {"death": "pilk"},
    "18x18_dense_kr_sg_kb_rotating_0": {"rotating": "blicket"},
    "18x18_dense_kr_sg_kb_frozen_0": {"frozen": "toma"},
    "18x18_dense_kr_sg_kb_portals_rotating_0": {"rotating": "blicket", "portal": "dax"},
    "18x18_dense_kr_sg_kb_portals_death_0": {"death": "pilk", "portal": "blicket"},
    "18x18_dense_kr_sg_kb_death_rotating_0": {"death": "pilk", "rotating": "wug"},
    "18x18_dense_kr_sg_kb_portals_frozen_0": {"frozen": "pilk", "portal": "modi"},
    "18x18_dense_kr_sg_kb_frozen_rotating_0": {"frozen": "blicket", "rotating": "zav"},
    "18x18_dense_kr_sg_kb_death_frozen_0": {"death": "blicket", "frozen": "fep"},
    "18x18_dense_kr_sg_kb_portals_death_rotating_0": {"death": "wug", "rotating": "fep", "portal": "dax"},
    "18x18_dense_kr_sg_kb_portals_rotating_frozen_0": {"frozen": "modi", "rotating": "blicket", "portal": "dax"},
    "18x18_dense_kr_sg_kb_portals_death_frozen_0": {"death": "wug", "frozen": "zav", "portal": "blicket"},
    "18x18_dense_kr_sg_kb_death_rotating_frozen_0": {"death": "pilk", "frozen": "zav", "rotating": "blicket"},
    "18x18_dense_kr_sg_kb_portals_death_rotating_frozen_0": {
        "death": "toma", "frozen": "pilk", "rotating": "dax", "portal": "modi",
    },
}


def _spec(task) -> TaskSpecification:
    return TaskSpecification.from_json(str(REPO / task["source"]))


def _kind_cells(spec) -> dict[str, set[tuple[int, int]]]:
    m = spec.mechanisms
    cells = {
        "death": {to_row_col(c) for c in m.kill_cells},
        "frozen": {to_row_col(c) for c in m.frozen_tiles},
        "rotating": {to_row_col(c) for c in m.rotating_tiles},
        "portal": {to_row_col(p) for t in m.teleporters for p in (t.position_a, t.position_b)},
    }
    return {k: v for k, v in cells.items() if v}


class _Replay:
    def __init__(self, actions):
        self._actions = list(actions)
        self.requests: list[list[dict]] = []
        self.last_usage = None

    def __call__(self, messages):
        self.requests.append(json.loads(json.dumps(messages)))
        i = len(self.requests) - 1
        return f"FINAL_OUTPUT: {self._actions[i] if i < len(self._actions) else 'DONE'}"


def _run(spec, config, actions):
    backend = MiniGridBackend(render_mode=None)  # text_only: no frames needed
    backend.configure(spec)
    agent = _Replay(actions)
    return build_runner(config, backend, spec).run(agent, verbose=False), agent.requests


def _texts(requests) -> list[str]:
    out = []
    for messages in requests:
        for message in messages:
            content = message["content"]
            out.append(content if isinstance(content, str) else "".join(b["text"] for b in content))
    return out


def _leaks(texts) -> set[str]:
    return {m.group(0).lower() for text in texts for m in LEAK.finditer(text)}


# --------------------------------------------------------------------------- config


def test_named_is_the_default_and_serializes_as_before():
    assert ExperimentConfig().tile_labels == "named"
    for cfg in (ExperimentConfig(), NAMED):
        assert "tile_labels" not in cfg.to_dict() and "tile_label_seed" not in cfg.to_dict()
        assert "tile_labels" not in _experiment_config_hash_payload(cfg)


def test_neutral_round_trips_and_moves_the_run_hash():
    seeded = dataclasses.replace(NEUTRAL, tile_label_seed=3)
    for cfg in (NEUTRAL, seeded):
        d = cfg.to_dict()
        assert (d["tile_labels"], d["tile_label_seed"]) == ("neutral", cfg.tile_label_seed)
        assert ExperimentConfig.from_dict(d) == cfg
    spec = TaskSpecification.from_json(str(MAZE))
    hashes = {
        _expected_run_hash(spec, "m", 0, "minigrid", experiment_config=cfg)
        for cfg in (NAMED, NEUTRAL, seeded)
    }
    assert len(hashes) == 3


@pytest.mark.parametrize(
    "field,value",
    [
        ("observation", "image_text"),
        ("observation", "image_only"),
        ("observation_text_format", "json"),
        ("observation_text_format", "ascii"),
        ("prompting", "standard"),
        ("prompting", "verbose"),
        ("prompting", "text_initial_maze"),
        ("in_context_learning", "one_shot"),
        ("feedback", "standard"),
        ("feedback", "causal"),
        ("context_window", "text_summary"),
        ("context_window", "text_summary_and_last_n"),
    ],
)
def test_neutral_rejects_cells_that_name_the_tiles(field, value):
    with pytest.raises(ValueError, match="tile_labels='neutral'"):
        ExperimentConfig(**{**CELLS["r2_nl"], field: value}, tile_labels="neutral")


@pytest.mark.parametrize("seed", [-1, True, 1.0])
def test_tile_label_seed_must_be_a_non_negative_int(seed):
    with pytest.raises(ValueError):
        ExperimentConfig(tile_labels="neutral", tile_label_seed=seed, **CELLS["r2_nl"])


def test_a_seed_without_neutral_labels_is_an_error():
    with pytest.raises(ValueError, match="tile_label_seed only applies"):
        ExperimentConfig(tile_label_seed=1)


# --------------------------------------------------------------------------- labels


def test_label_draw_is_pinned_and_reshuffles_across_mazes():
    effects_by_label: dict[str, set[str]] = {}
    for task in TASKS:
        spec = _spec(task)
        labeling = TileLabeling.for_task(spec, 0)
        assert labeling.label_by_kind == PINNED_SEED0[spec.task_id], spec.task_id
        assert list(labeling.label_by_kind) == [k for k in KINDS if k in _kind_cells(spec)]
        assert len(set(labeling.label_by_kind.values())) == len(labeling.label_by_kind)
        assert set(labeling.label_by_kind.values()) <= set(LABEL_POOL)
        assert TileLabeling.for_task(spec, 0) == labeling  # deterministic
        for kind, label in labeling.label_by_kind.items():
            effects_by_label.setdefault(label, set()).add(kind)
    # The same word means different things on different mazes.
    assert sum(len(kinds) > 1 for kinds in effects_by_label.values()) >= 4
    # Another seed is another draw.
    assert any(
        TileLabeling.for_task(_spec(t), 1).label_by_kind != TileLabeling.for_task(_spec(t), 0).label_by_kind
        for t in TASKS
    )


def test_m10_maze_text_lists_every_tile_once_under_its_label():
    spec = TaskSpecification.from_json(str(MAZE))
    labeling = TileLabeling.for_task(spec, 0)
    text = render_initial_maze_text(spec, labeling=labeling)
    assert not _leaks([text])
    tile_lines = [line for line in text.splitlines() if " tile at " in line]
    cells = _kind_cells(spec)
    assert len(tile_lines) == sum(len(c) for c in cells.values())
    listed = [re.match(r"There is a (\w+) tile at \((\d+),(\d+)\)", line).groups() for line in tile_lines]
    # Grouped by label in alphabetical order, cells row-major within a label.
    keys = [(label, int(r), int(c)) for label, r, c in listed]
    assert keys == sorted(keys)
    for kind, label in labeling.label_by_kind.items():
        assert {(r, c) for lab, r, c in keys if lab == label} == cells[kind]
    pointing = [line for line in tile_lines if "pointing" in line]
    assert {re.match(r"There is a (\w+) tile", line).group(1) for line in pointing} == {
        labeling.label_by_kind["rotating"]
    }
    for fmt in ("json", "ascii"):
        with pytest.raises(ValueError, match="coords"):
            render_initial_maze_text(spec, observation_text_format=fmt, labeling=labeling)


def test_neutral_observation_has_no_frozen_status_and_follows_the_turntables():
    spec = TaskSpecification.from_json(str(MAZE))
    labeling = TileLabeling.for_task(spec, 0)
    backend = MiniGridBackend(render_mode=None)
    backend.configure(spec)
    _, state, _ = backend.reset(seed=spec.seed)
    frozen = dataclasses.replace(state, freeze_remaining=3)
    named = render_user_observation_text(spec, frozen, include_facing=True)
    neutral = render_user_observation_text(spec, frozen, include_facing=True, labeling=labeling)
    assert "You are frozen" in named and not _leaks([neutral])
    spun = dataclasses.replace(state, rotator_dirs=tuple((d + 1) % 4 for d in state.rotator_dirs))
    rot = labeling.label_by_kind["rotating"]
    before = [l for l in neutral.splitlines() if l.startswith(f"There is a {rot} tile")]
    after = [
        l for l in render_user_observation_text(spec, spun, labeling=labeling).splitlines()
        if l.startswith(f"There is a {rot} tile")
    ]
    assert len(before) == len(after) == 6 and before != after


# --------------------------------------------------------------------------- feedback


# M10's death-detour route (tests/test_mechanics_m10_trace.py): every step whose
# R2 event is not MOVED/TURNED, as (step, R2 event type, neutral feedback).
M10_NEUTRAL_FEEDBACK = [
    (7, "ROTATING", "MOVED"),            # onto a turntable, the cell ahead
    (8, "NOTHING", "NOTHING"),
    (9, "FROZEN", "MOVED (3,4)"),        # a ride east onto ice while facing south
    (10, "FROZEN", "NOTHING"),
    (11, "FROZEN", "NOTHING"),
    (12, "FROZEN", "NOTHING"),
    (13, "FROZEN", "NOTHING"),
    (14, "FROZEN", "NOTHING"),
    (15, "DEATH RESET", "RESET"),
    (20, "ROTATING", "MOVED"),
    (21, "CARRIED", "MOVED"),            # a ride that went the way the agent faced
    (22, "CARRIED", "MOVED"),
    (26, "TELEPORTED", "MOVED (11,2)"),
    (37, "TELEPORTED", "MOVED (15,14)"),
    (53, "PICKUP", "PICKUP"),
    (60, "FROZEN", "MOVED"),             # onto ice, the cell ahead
    (61, "FROZEN", "NOTHING"),
    (62, "FROZEN", "NOTHING"),
    (63, "FROZEN", "NOTHING"),
    (64, "FROZEN", "NOTHING"),
    (65, "FROZEN", "NOTHING"),
    (66, "ROTATING", "MOVED"),
    (68, "OPENED", "OPENED"),
    (69, "NOTHING", "NOTHING"),
    (70, "CARRIED", "MOVED"),
    (90, "DONE", "DONE"),
]


def test_m10_route_same_dynamics_consequence_only_feedback_no_leaks():
    spec = TaskSpecification.from_json(str(MAZE))
    route = death_detour_route(spec)
    named, named_requests = _run(spec, NAMED, route)
    neutral, requests = _run(spec, NEUTRAL, route)
    # The detector does see R2's names (else "no leaks" below proves nothing).
    assert _leaks(_texts(named_requests)) >= {
        "death", "frozen", "rotat", "teleport", "purple", "cyan", "between",
    }
    for result in (named, neutral):
        assert (result["end_reason"], result["steps_used"]) == ("success", 90)
    n_steps = [r for r in named["transcript"] if r.get("kind") == "step"]
    x_steps = [r for r in neutral["transcript"] if r.get("kind") == "step"]
    # Same world, same world-model labels: only the text the model reads differs.
    assert [s["state_after"] for s in x_steps] == [s["state_after"] for s in n_steps]
    assert [s["event_type"] for s in x_steps] == [s["event_type"] for s in n_steps]
    assert [s.get("mechanic") for s in x_steps] == [s.get("mechanic") for s in n_steps]
    got = [
        (i, s["event_type"], s["prompt_feedback"])
        for i, s in enumerate(x_steps, 1)
        if s["event_type"] not in ("MOVED", "TURNED")
    ]
    assert got == M10_NEUTRAL_FEEDBACK
    assert all(s["prompt_feedback"] in ("MOVED", "TURNED") for s in x_steps if s["event_type"] in ("MOVED", "TURNED"))
    assert not _leaks(_texts(requests))
    assert neutral["tile_labels"]["labels"]["toma"]["effect"] == "death"
    assert "tile_labels" not in named


def test_neutral_feedback_is_minimal_only():
    spec = TaskSpecification.from_json(str(MAZE))
    backend = MiniGridBackend(render_mode=None)
    backend.configure(spec)
    _, state, _ = backend.reset(seed=spec.seed)
    with pytest.raises(ValueError, match="minimal"):
        format_step_feedback("TURN_LEFT", state, state, 0.0, False, spec, level="standard", neutral_tiles=True)


# --------------------------------------------------------------------------- planner replay


@pytest.mark.parametrize("task", TASKS, ids=[t["task_id"] for t in TASKS])
def test_planner_route_solves_every_mechanics_15_maze_under_neutral_labels(task):
    spec = _spec(task)
    plan = plan_bfs_path(spec)  # the BFS behind the manifest's optimal_steps and R2's 3x cap
    optimal = task["optimal_steps"]
    assert plan.success and len(plan.actions) == optimal
    spec = dataclasses.replace(spec, max_steps=min(spec.max_steps, 3 * optimal))

    result, requests = _run(spec, NEUTRAL, [ACTION_ORDER[a] for a in plan.actions])
    assert (result["end_reason"], result["steps_used"]) == ("success", optimal)
    steps = [r for r in result["transcript"] if r.get("kind") == "step"]
    # The NL episode walks the planner's own states.
    assert [tuple(s["state_after"]["agent_position"]) for s in steps] == [tuple(p) for p in plan.positions[1:]]

    recorded = result["tile_labels"]
    assert recorded["seed"] == 0 and recorded["task_id"] == spec.task_id
    cells = _kind_cells(spec)
    assert {v["effect"]: {tuple(c) for c in v["cells"]} for v in recorded["labels"].values()} == cells
    assert {v["effect"]: k for k, v in recorded["labels"].items()} == PINNED_SEED0[spec.task_id]
    assert len(recorded["portal_links"]) == len(spec.mechanisms.teleporters)
    assert result["config"]["tile_labels"] == "neutral"

    texts = _texts(requests)
    assert not _leaks(texts), _leaks(texts)
    for label in recorded["labels"]:
        assert f"There is a {label} tile at" in texts[0]


def test_episode_json_records_the_mapping(tmp_path):
    from pipeline.run_stage3 import run_episode

    spec = TaskSpecification.from_json(str(MAZE))
    route = death_detour_route(spec)
    episode = run_episode(MAZE, NEUTRAL, _Replay(route), 0, tmp_path / "neutral")
    assert episode["success"] is True
    assert episode["tile_labels"] == TileLabeling.for_task(spec, 0).to_dict()
    assert (episode["config"]["tile_labels"], episode["config"]["tile_label_seed"]) == ("neutral", 0)
    assert {v["effect"] for v in episode["tile_labels"]["labels"].values()} == set(KINDS)
    named = run_episode(MAZE, NAMED, _Replay(route), 0, tmp_path / "named")
    assert "tile_labels" not in named and "tile_labels" not in named["config"]
