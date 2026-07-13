"""Goal-source consistency between spec, solvers, and the runtime env.

The env scores ``reach_position`` against ``goal.target`` while the solvers,
prompt, and rendered goal tile historically read ``maze.goal``. Two generated
mazes shipped with the fields disagreeing (d3_dense_deadend: maze.goal (8,1)
vs goal.target (8,8)), so "BFS optimal" was computed for a goal the models
were never asked to reach — models legitimately beat it by 8 steps.

These tests pin the fix: one resolved goal everywhere, validation that refuses
mismatched specs, and difficulty step-counts that match executable actions
(a locked door costs TOGGLE + MOVE_FORWARD, not one fused transition).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gridworld.baselines import plan_bfs_path
from gridworld.task_spec import TaskSpecification
from gridworld.task_validator import TaskValidator, compute_difficulty


def make_spec(**overrides):
    data = {
        "task_id": "goal_resolution_case",
        "seed": 1,
        "difficulty_tier": 1,
        "maze": {
            "dimensions": [5, 5],
            "walls": [],
            "start": [1, 1],
            "goal": [3, 3],
        },
        "mechanisms": {},
        "rules": {"observability": "full", "view_size": 7},
        "goal": {"type": "reach_position", "target": [3, 3]},
        "max_steps": 20,
    }
    data.update(overrides)
    return TaskSpecification.from_dict(data)


def test_resolved_goal_prefers_goal_target():
    spec = make_spec(goal={"type": "reach_position", "target": [1, 3]})
    assert spec.resolved_goal() == (1, 3)


def test_resolved_goal_falls_back_to_maze_goal():
    spec = make_spec(goal={"type": "reach_position"})
    assert spec.resolved_goal() == (3, 3)
    spec = make_spec(goal={"type": "collect_all", "target_ids": ["k1"]})
    assert spec.resolved_goal() == (3, 3)


def test_validate_rejects_goal_target_maze_goal_mismatch():
    spec = make_spec(goal={"type": "reach_position", "target": [1, 3]})
    is_valid, errors = spec.validate()
    assert is_valid is False
    assert any("maze.goal" in e and "goal.target" in e for e in errors)


def test_solvers_plan_to_the_goal_the_env_scores():
    # maze.goal (3,3) is 5 executable steps away; the env goal (1,3) only 3
    # (TURN_RIGHT, MOVE_FORWARD, MOVE_FORWARD). Both solvers must plan to the
    # env goal, or "optimal" is computed for a cell the episode never targets.
    spec = make_spec(goal={"type": "reach_position", "target": [1, 3]})

    bfs = plan_bfs_path(spec)
    assert bfs.success
    assert bfs.positions[-1] == (1, 3)
    assert len(bfs.action_labels) == 3

    ok, solution, _ = TaskValidator(spec).validate()
    assert ok
    assert solution[-1] == (1, 3)

    assert compute_difficulty(spec).optimal_steps == 3


def test_difficulty_counts_door_passage_as_toggle_plus_move():
    # Corridor: start (1,2) -> key (2,2) -> locked door (3,2) -> goal (5,2).
    # Executable plan: MOVE, PICKUP, TOGGLE (opens door), MOVE, MOVE, MOVE = 6.
    # The validator's fused open-door-and-move transition under-counted this
    # as 5, drifting difficulty -1 per door vs. what an agent must execute.
    spec = make_spec(
        maze={
            "dimensions": [7, 5],
            "walls": [[x, y] for x in range(1, 6) for y in (1, 3)],
            "start": [1, 2],
            "goal": [5, 2],
        },
        mechanisms={
            "keys": [{"id": "kY", "position": [2, 2], "color": "yellow"}],
            "doors": [
                {
                    "id": "DY",
                    "position": [3, 2],
                    "requires_key": "yellow",
                    "initial_state": "locked",
                }
            ],
        },
        goal={"type": "reach_position", "target": [5, 2]},
    )

    bfs = plan_bfs_path(spec)
    assert bfs.success
    assert len(bfs.action_labels) == 6

    assert compute_difficulty(spec).optimal_steps == 6
