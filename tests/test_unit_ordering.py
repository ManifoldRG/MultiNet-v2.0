"""Longest-processing-time-first unit ordering: the coordinator assigns in plan order,
so long mazes (big step cap) must come first and fast mazes (empty_room) last, to
minimize batch makespan under concurrency."""
from scripts.distributed_run_pipeline import _order_units_lpt


def test_orders_by_max_steps_descending():
    units = [
        {"unit_id": "a", "task_id": "empty_room", "task_payload": {"max_steps": 33}},
        {"unit_id": "b", "task_id": "s5_corridor", "task_payload": {"max_steps": 267}},
        {"unit_id": "c", "task_id": "mid", "task_payload": {"max_steps": 120}},
    ]
    assert [u["task_id"] for u in _order_units_lpt(units, {})] == \
        ["s5_corridor", "mid", "empty_room"]


def test_falls_back_to_optimal_steps():
    units = [
        {"unit_id": "a", "task_id": "x", "task_payload": {}},
        {"unit_id": "b", "task_id": "y", "task_payload": {}},
    ]
    static = {"x": {"optimal_steps": 5}, "y": {"optimal_steps": 50}}
    assert [u["task_id"] for u in _order_units_lpt(units, static)] == ["y", "x"]


def test_stable_and_total_preserved():
    units = [{"unit_id": f"u{i}", "task_id": "same", "task_payload": {"max_steps": 10}}
             for i in range(5)]
    out = _order_units_lpt(units, {})
    assert [u["unit_id"] for u in out] == [f"u{i}" for i in range(5)]  # stable
    assert len(out) == 5
