"""compute_fragility: the irreversible-step search gives the same report as the
original algorithm, which solved from every explored state, and solves only
after irreversible transitions."""

from __future__ import annotations

from collections import deque

import pytest

from gridworld.task_spec import TaskSpecification
from gridworld.task_validator import FragilityReport, PlannerState, TaskValidator
from maze_test_utils import MAZE_JSON_DIR

# One red key, two red doors, keys consumed on use: opening the decoy door D1
# first strands the key, so the goal behind D2 becomes unreachable.
DECOY_DOOR = {
    "task_id": "fragility_decoy_door",
    "seed": 0,
    "difficulty_tier": 2,
    "maze": {
        "dimensions": [9, 5],
        "walls": [[1, 1], [2, 1], [4, 1], [5, 1], [6, 1], [7, 1],
                  [1, 3], [2, 3], [3, 3], [4, 3], [5, 3], [6, 3], [7, 3]],
        "start": [1, 2],
        "goal": [7, 2],
    },
    "mechanisms": {
        "keys": [{"id": "kR", "position": [2, 2], "color": "red"}],
        "doors": [
            {"id": "D1", "position": [3, 1], "requires_key": "red", "initial_state": "locked"},
            {"id": "D2", "position": [6, 2], "requires_key": "red", "initial_state": "locked"},
        ],
    },
    "rules": {"key_consumption": True},
    "goal": {"type": "reach_position", "target": [7, 2]},
    "max_steps": 60,
}


def _reference_fragility(validator: TaskValidator, depth_limit: int = 5) -> FragilityReport:
    """The original algorithm, verbatim: a full solve from every newly reached state."""
    from gridworld.task_validator import successors

    initial_state = validator.ctx.initial_state()
    queue = deque([(initial_state, [])])
    visited: dict[PlannerState, int] = {initial_state: 0}
    breaking_sequences: list[list[str]] = []
    min_steps_to_break = None
    while queue:
        state, sequence = queue.popleft()
        if min_steps_to_break is not None and len(sequence) >= min_steps_to_break:
            continue
        if len(sequence) >= depth_limit:
            continue
        for transition in successors(validator.ctx, state):
            next_sequence = list(sequence)
            if validator._is_irreversible_transition(state, transition):
                next_sequence = sequence + [transition.label]
            next_irrev = len(next_sequence)
            if next_irrev > depth_limit:
                continue
            if transition.next_state in visited and visited[transition.next_state] <= next_irrev:
                continue
            visited[transition.next_state] = next_irrev
            beatable, _, _ = validator._find_solution(transition.next_state)
            if not beatable and validator._is_irreversible_transition(state, transition):
                min_steps_to_break = (
                    len(next_sequence)
                    if min_steps_to_break is None
                    else min(min_steps_to_break, len(next_sequence))
                )
                if len(next_sequence) == min_steps_to_break:
                    breaking_sequences.append(next_sequence)
                continue
            queue.append((transition.next_state, next_sequence))
    if min_steps_to_break is None:
        return FragilityReport(min_steps_to_break=-1, breaking_sequences=[], is_fragile=False)
    return FragilityReport(
        min_steps_to_break=min_steps_to_break,
        breaking_sequences=breaking_sequences[:depth_limit],
        is_fragile=min_steps_to_break <= 3,
    )


def _corpus_sample() -> list:
    # Mazes across families, including keys, switches and the new tiles, so the
    # comparison covers every irreversible label. The reference algorithm takes
    # minutes on the larger ones, so those run in the slow tier.
    fast = ["D3/10x10_dense_deadend_ky_dy_1.json"]
    slow = [
        "M6/14x14_dense_kr_sg_kb_1.json",
        "M8/18x18_dense_kr_sg_kb_portals_death_0.json",
        "M8/18x18_dense_kr_sg_kb_portals_frozen_0.json",
    ]
    return [
        pytest.param(MAZE_JSON_DIR / n, marks=[pytest.mark.slow] if n in slow else [])
        for n in fast + slow
        if (MAZE_JSON_DIR / n).exists()
    ]


def test_decoy_door_breaks_in_one_irreversible_step():
    report = TaskValidator(TaskSpecification.from_dict(DECOY_DOOR)).compute_fragility()
    assert report.min_steps_to_break == 1
    assert report.is_fragile
    assert report.breaking_sequences == [["open_door:D1"]]


def test_matches_reference_on_the_decoy_door():
    spec = TaskSpecification.from_dict(DECOY_DOOR)
    assert TaskValidator(spec).compute_fragility() == _reference_fragility(TaskValidator(spec))


@pytest.mark.parametrize("path", _corpus_sample(), ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_matches_reference_on_corpus_mazes(path):
    spec = TaskSpecification.from_json(str(path))
    assert TaskValidator(spec).compute_fragility() == _reference_fragility(TaskValidator(spec))


def test_solves_only_after_irreversible_transitions(monkeypatch):
    validator = TaskValidator(TaskSpecification.from_dict(DECOY_DOOR))
    calls = []
    real = validator._find_solution

    def counting(state, *args, **kwargs):
        calls.append(state)
        return real(state, *args, **kwargs)

    monkeypatch.setattr(validator, "_find_solution", counting)
    validator.compute_fragility()
    # Only door openings (keys are consumed) and drops are irreversible here; a
    # full solve after every move would be one call per explored state.
    assert 0 < len(calls) <= 4
    assert len(calls) == len(set(calls)), "each state is solved at most once"
