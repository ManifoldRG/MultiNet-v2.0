"""The planner's DROP edge exists only when the harness exposed DROP.

R1 ran without DROP in the model-facing vocabulary, so a decoy-key pickup was
mechanically unrecoverable ("doomed"). The 2026-07-30 rerun exposed DROP, so the
same state is recoverable and must not be scored as doomed.
"""
from __future__ import annotations

import json
from pathlib import Path

from gridworld.baselines import TaskPlanningContext, _successors
from gridworld.task_spec import TaskSpecification

D2 = Path("ogbench/ogbench/procgen/maze_jsons/D2/8x8_corridor_wrong_ky_inactive_sb_sg_kr_1.json")


def _spec():
    return TaskSpecification.from_dict(json.loads(D2.read_text()))


def _carrying_decoy(ctx):
    """A planner state holding the yellow decoy key."""
    decoy = next(k for k, v in ctx.keys_by_id.items() if v["color"] == "yellow")
    base = ctx.initial_state()
    return base.__class__(
        agent_pos=ctx.keys_by_id[decoy]["position"],
        agent_dir=base.agent_dir,
        carrying_key=decoy,
        collected_keys=frozenset({decoy}),
        active_switches=base.active_switches,
        used_switches=base.used_switches,
        open_gates=base.open_gates,
        open_doors=base.open_doors,
    )


def test_no_drop_edge_by_default():
    ctx = TaskPlanningContext(_spec())
    labels = [t.label for t in _successors(ctx, _carrying_decoy(ctx))]
    assert not any(l.startswith("drop:") for l in labels)


def test_drop_edge_when_enabled():
    ctx = TaskPlanningContext(_spec(), drop_available=True)
    state = _carrying_decoy(ctx)
    drops = [t for t in _successors(ctx, state) if t.label.startswith("drop:")]
    assert len(drops) == 1
    after = drops[0].next_state
    assert after.carrying_key is None
    # The key stays "collected" — it is gone from the world, not re-acquirable.
    assert after.collected_keys == state.collected_keys
    assert after.agent_pos == state.agent_pos


def test_no_drop_edge_with_empty_hands():
    ctx = TaskPlanningContext(_spec(), drop_available=True)
    labels = [t.label for t in _successors(ctx, ctx.initial_state())]
    assert not any(l.startswith("drop:") for l in labels)


def test_optimal_cost_is_unchanged_by_the_drop_edge():
    """The regression gate: DROP must never shorten the optimal solution, or
    every difficulty number in the corpus shifts."""
    from gridworld.baselines import plan_bfs_path

    spec = _spec()
    without = plan_bfs_path(spec)
    with_drop = plan_bfs_path(spec, drop_available=True)
    # The brief's test compares path length via len(); PlannedPath is a plain
    # dataclass with no __len__, so we compare the actual action-count field
    # it's meant to express (see task-1-report.md for why).
    assert len(without.actions) == len(with_drop.actions)
