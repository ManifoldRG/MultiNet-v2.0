"""Unit tests for Stage-3 instrumentation (pipeline.episode_metrics)."""

from __future__ import annotations

from pipeline import episode_metrics as em


def _state(pos, *, keys=(), switches=(), doors=(), gates=(), reward=0.0):
    return {
        "agent_position": list(pos),
        "collected_keys": list(keys),
        "active_switches": list(switches),
        "open_doors": list(doors),
        "open_gates": list(gates),
        "reward": reward,
    }


def _step(pos, event_type="MOVED", **state_kwargs):
    return {"kind": "step", "event_type": event_type, "state_after": _state(pos, **state_kwargs)}


def _episode(steps, *, success, end_reason, initial_pos=(1, 1)):
    initial = _state(initial_pos)
    final = steps[-1]["state_after"] if steps else initial
    return {
        "success": success,
        "end_reason": end_reason,
        "steps_used": len(steps),
        "initial_state": initial,
        "final_state": final,
        "transcript": [{"kind": "reset", "state": initial}, *steps],
    }


# --------------------------------------------------------------------------- #
# visited_cells: uses state_after.agent_position (x, y), collapses duplicates
# --------------------------------------------------------------------------- #
def test_visited_cells_uses_agent_position_and_dedupes():
    ep = _episode(
        [_step((1, 1)), _step((2, 1)), _step((2, 1)), _step((3, 1))],
        success=True,
        end_reason="success",
    )
    assert em.visited_cells(ep) == [(1, 1), (2, 1), (3, 1)]


# --------------------------------------------------------------------------- #
# mechanism_interaction_order
# --------------------------------------------------------------------------- #
def test_mechanism_order_key_then_switch():
    ep = _episode(
        [
            _step((2, 1), "PICKUP", keys=("kB",)),
            _step((2, 1), "TOGGLED", keys=("kB",), switches=("s1",), gates=("g1",)),
        ],
        success=True,
        end_reason="success",
    )
    # switch (active_switches) ranks before its downstream gate (open_gates).
    assert em.mechanism_interaction_order(ep) == ["kB", "s1", "g1"]


def test_mechanism_order_switch_then_key():
    ep = _episode(
        [
            _step((2, 1), "TOGGLED", switches=("s1",), gates=("g1",)),
            _step((6, 1), "PICKUP", switches=("s1",), gates=("g1",), keys=("kB",)),
        ],
        success=True,
        end_reason="success",
    )
    assert em.mechanism_interaction_order(ep) == ["s1", "g1", "kB"]


def test_mechanism_order_navigation_only_is_empty():
    ep = _episode([_step((2, 1)), _step((3, 1))], success=True, end_reason="success")
    assert em.mechanism_interaction_order(ep) == []


# --------------------------------------------------------------------------- #
# failure_point
# --------------------------------------------------------------------------- #
def test_failure_point_reports_first_missing_expected_mechanism():
    ep = _episode(
        [_step((2, 1), "PICKUP", keys=("kB",))],
        success=False,
        end_reason="max_steps",
    )
    order = em.mechanism_interaction_order(ep)
    fp = em.failure_point(ep, ["kB", "s1"], order)
    assert fp["mechanism"] == "s1"
    assert fp["end_reason"] == "max_steps"
    assert fp["final_cell"] == [2, 1]


def test_failure_point_none_on_success():
    ep = _episode([_step((2, 1))], success=True, end_reason="success")
    assert em.failure_point(ep, ["kB"], []) is None


# --------------------------------------------------------------------------- #
# path_choice
# --------------------------------------------------------------------------- #
def test_path_choice_short_long_mixed_none():
    short = [[5, 1], [6, 1]]
    long = [[2, 5], [3, 5]]
    short_ep = _episode([_step((5, 1)), _step((6, 1))], success=True, end_reason="success")
    long_ep = _episode([_step((2, 5)), _step((3, 5))], success=False, end_reason="max_steps")
    mixed_ep = _episode([_step((5, 1)), _step((2, 5))], success=False, end_reason="max_steps")
    none_ep = _episode([_step((9, 9))], success=False, end_reason="max_steps")

    assert em.path_choice(short_ep, short, long) == "short_mech"
    assert em.path_choice(long_ep, short, long) == "long_open"
    assert em.path_choice(mixed_ep, short, long) == "mixed"
    assert em.path_choice(none_ep, short, long) == "none"
    assert em.path_choice(short_ep, None, None) is None


# --------------------------------------------------------------------------- #
# token accounting + run row
# --------------------------------------------------------------------------- #
def test_episode_token_count_sums_query_usage():
    ep = {
        "transcript": [
            {"kind": "query", "usage": {"total_tokens": 10}},
            {"kind": "step"},
            {"kind": "query", "usage": {"input_tokens": 5, "output_tokens": 3}},
        ]
    }
    assert em.episode_token_count(ep) == 18


def test_build_run_row_fields_and_optimality():
    ep = _episode(
        [_step((2, 1), "PICKUP", keys=("kB",)), _step((3, 1))],
        success=True,
        end_reason="success",
    )
    ep["transcript"].append({"kind": "query", "usage": {"total_tokens": 12}})
    canonical = {"bfs": {"optimal_steps": 2}}
    manifest_row = {
        "task_id": "T_demo",
        "experiment": "test3",
        "condition": "key_first",
        "expected_mechanisms": ["kB", "s1"],
    }
    row = em.build_run_row(
        ep, canonical, manifest_row, agent_or_model="stub", seed=0, raw_output_ref="x/episode.json"
    )
    assert row["task_id"] == "T_demo"
    assert row["experiment"] == "test3"
    assert row["backend"] == "minigrid"
    assert row["agent_or_model"] == "stub"
    assert row["success"] is True
    assert row["terminated"] is True
    assert row["truncated"] is False
    assert row["optimal_steps"] == 2
    assert row["optimality_ratio"] == 1.0  # steps_used (2) == optimal (2)
    assert row["mechanism_interaction_order"] == ["kB"]
    assert row["failure_point"] is None
    assert row["tokens"] == 12
    assert row["raw_output_ref"] == "x/episode.json"
