import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gridworld.task_spec import TaskSpecification
from gridworld.task_validator import TaskValidator, compute_difficulty


def make_spec(**overrides):
    data = {
        "task_id": "validation_case",
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


def assert_invalid(spec, *expected_fragments):
    is_valid, errors = spec.validate()
    assert is_valid is False
    joined = "\n".join(errors)
    for fragment in expected_fragments:
        assert fragment in joined


def test_validate_rejects_basic_schema_errors():
    assert_invalid(
        make_spec(maze={"dimensions": [2, 5], "walls": [], "start": [1, 1], "goal": [1, 3]}),
        "Maze dimensions too small",
    )
    assert_invalid(
        make_spec(
            mechanisms={"keys": [{"id": "k1", "position": [5, 1], "color": "red"}]},
        ),
        "Key k1 position (5, 1) out of bounds",
    )
    assert_invalid(
        make_spec(
            mechanisms={
                "keys": [{"id": "k1", "position": [1, 2], "color": "red"}],
                "doors": [{"id": "d1", "position": [2, 2], "requires_key": "blue"}],
            },
        ),
        "requires color 'blue'",
    )
    assert_invalid(
        make_spec(
            mechanisms={
                "switches": [{"id": "s1", "position": [1, 2], "controls": ["g-missing"]}],
            },
        ),
        "controls non-existent gate 'g-missing'",
    )


def test_validate_rejects_id_and_position_integrity_errors():
    assert_invalid(
        make_spec(
            mechanisms={
                "doors": [{"id": "shared", "position": [2, 1], "requires_key": "red"}],
                "gates": [{"id": "shared", "position": [3, 1]}],
            },
        ),
        "Duplicate mechanism id 'shared'",
    )
    assert_invalid(
        make_spec(
            mechanisms={
                "keys": [
                    {"id": "k1", "position": [1, 2], "color": "red"},
                    {"id": "k2", "position": [1, 2], "color": "blue"},
                ],
            },
        ),
        "Key k2 position (1, 2) overlaps Key k1",
    )
    assert_invalid(
        make_spec(rules={"hidden_mechanisms": ["missing"], "observability": "fog_of_war", "view_size": 7}),
        "rules.hidden_mechanisms references unknown id 'missing'",
    )


def test_validate_rejects_dependency_and_distractor_reference_errors():
    assert_invalid(
        make_spec(
            mechanisms={"keys": [{"id": "k1", "position": [1, 2], "color": "red"}]},
            dependency_chain={
                "depth": 1,
                "sequence": [
                    {"step": 1, "type": "key-door", "element": "k1", "unlocks": "missing-door"},
                ],
            },
            distractors=[
                {"type": "wrong_color_key", "element_id": "missing-key"},
                {"type": "distractor_chain", "element_id": "virtual-chain"},
            ],
        ),
        "references unknown unlock target 'missing-door'",
        "references unknown element_id 'missing-key'",
    )


def test_validate_rejects_border_cells_and_invalid_view_size():
    assert_invalid(
        make_spec(
            maze={"dimensions": [5, 5], "walls": [], "start": [1, 1], "goal": [0, 0]},
            goal={"type": "reach_position", "target": [0, 0]},
        ),
        "Goal position (0, 0) is a wall",
        "Goal target position (0, 0) is a wall",
    )
    assert_invalid(
        make_spec(rules={"observability": "view_cone", "view_size": 4}),
        "Invalid view_size: 4",
    )
    assert_invalid(
        make_spec(rules={"observability": "view_cone", "view_size": 1}),
        "Invalid view_size: 1",
    )


def test_validate_rejects_goal_type_field_errors():
    assert_invalid(
        make_spec(goal={"type": "reach_position"}),
        "Goal type 'reach_position' requires target",
    )
    assert_invalid(
        make_spec(goal={"type": "collect_all"}),
        "Goal type 'collect_all' requires target_ids",
    )
    assert_invalid(
        make_spec(
            mechanisms={"keys": [{"id": "k1", "position": [1, 2], "color": "red"}]},
            goal={"type": "collect_all", "target_ids": ["missing"]},
        ),
        "references unknown target_id 'missing'",
    )
    assert_invalid(
        make_spec(
            mechanisms={"blocks": [{"id": "b1", "position": [1, 2]}]},
            goal={"type": "push_block_to", "target_ids": ["b1"], "target_positions": [[3, 3], [2, 2]]},
        ),
        "requires one target_position per target_id",
    )
    assert_invalid(
        make_spec(
            mechanisms={"keys": [{"id": "k1", "position": [1, 2], "color": "red"}]},
            goal={"type": "push_block_to", "target_ids": ["k1"], "target_positions": [[3, 3]]},
        ),
        "references unknown block id 'k1'",
    )


def test_bfs_switch_and_key_interactions_use_current_cell():
    spec = make_spec(
        maze={"dimensions": [5, 5], "walls": [], "start": [1, 1], "goal": [3, 1]},
        mechanisms={
            "switches": [{"id": "s1", "position": [2, 1], "controls": ["g1"]}],
            "gates": [{"id": "g1", "position": [3, 1], "initial_state": "closed"}],
        },
        goal={"type": "reach_position", "target": [3, 1]},
    )

    is_beatable, path, message = TaskValidator(spec).validate()

    assert is_beatable is True
    assert path == [(1, 1), (2, 1), (2, 1), (3, 1)]
    assert "3 steps" in message


def test_bfs_counts_turn_actions_as_steps():
    spec = make_spec(
        maze={"dimensions": [5, 5], "walls": [], "start": [1, 1], "goal": [1, 2]},
        goal={"type": "reach_position", "target": [1, 2]},
    )

    report = compute_difficulty(spec)

    assert report.is_beatable is True
    assert report.optimal_steps == 2
    assert report.optimal_path == [(1, 1), (1, 1), (1, 2)]
    assert report.backtrack_count == 0
