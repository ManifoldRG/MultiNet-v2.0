from pathlib import Path

import pandas as pd

from analysis.propose_maze_candidates import (
    DEFAULT_MAZE_ROOT,
    PROFILES,
    _feature_row,
    candidate_table,
    generate_proposals,
)


def _row(relative: str) -> dict:
    path = DEFAULT_MAZE_ROOT / relative
    return _feature_row(path, DEFAULT_MAZE_ROOT)


def test_path_depth_and_option_count_distinguish_objects_from_challenges():
    m6 = _row("M6/8x8_corridor_kr_sg_kb_0.json")
    d2_m6 = _row("D2/8x8_corridor_wrong_ky_kr_inactive_sb_sg_kb_0.json")

    assert m6["path_signature"] == "K->S->K"
    assert m6["path_depth"] == 3
    assert m6["option_count"] == 3
    assert m6["distractor_count"] == 0

    assert d2_m6["path_signature"] == "K->S->K"
    assert d2_m6["path_depth"] == 3
    assert d2_m6["option_count"] == 5
    assert d2_m6["distractor_count"] == 2
    assert d2_m6["five_option"]


def test_feature_row_matches_difficulty_authority():
    # The panel selector must source beatability/lengths/object counts from
    # compute_difficulty so candidate lists can never drift from the repo's
    # difficulty numbers (analysis.mazes.maze_features, validator).
    from analysis.mazes import load_maze_spec
    from gridworld.task_validator import compute_difficulty

    relative = "M6/8x8_corridor_kr_sg_kb_0.json"
    row = _row(relative)
    report = compute_difficulty(load_maze_spec(str(DEFAULT_MAZE_ROOT / relative)))

    assert row["is_beatable"] == report.is_beatable
    assert row["optimal_steps"] == report.optimal_steps
    assert row["states_explored"] == report.states_explored
    assert row["object_count"] == report.mechanism_count


def test_compute_difficulty_accepts_precomputed_bfs_path():
    from analysis.mazes import load_maze_spec
    from gridworld.baselines import plan_bfs_path
    from gridworld.task_validator import compute_difficulty

    spec = load_maze_spec(str(DEFAULT_MAZE_ROOT / "M1/10x10_corridor_kr_0.json"))
    fresh = compute_difficulty(spec)
    reused = compute_difficulty(spec, bfs_path=plan_bfs_path(spec))

    assert reused.to_dict() == fresh.to_dict()


def test_d3_deadend_pair_is_one_off_path_option():
    d3 = _row("D3/10x10_dense_kr_kb_deadend_ky_dy_0.json")

    assert d3["path_signature"] == "K->K"
    assert d3["path_depth"] == 2
    assert d3["option_count"] == 3
    assert d3["distractor_count"] == 1
    assert d3["known_length_risk"]


def test_generated_panel_is_unique_and_respects_b_floor(tmp_path: Path):
    # A compact synthetic feature frame exercises selection constraints without
    # running all 214 BFS plans in the unit test.
    rows = []
    signatures = ["none", "K", "S", "K->K", "K->S", "S->K", "K->S->K"]
    families = ["S1", "M1", "M2", "M5", "M3", "M4", "M6", "D1", "D2", "D3", "B1"]
    for index in range(80):
        family = "B1" if index < 10 else families[index % len(families)]
        signature = "S" if family == "B1" else signatures[index % len(signatures)]
        depth = signature.count("->") + (signature != "none")
        rows.append({
            "maze_id": f"{family}/maze_{index}",
            "task_id": f"task_{index}",
            "family": family,
            "is_b": family == "B1",
            "is_beatable": True,
            "optimal_steps": 9 + index,
            "path_signature": signature,
            "path_depth": depth,
            "distractor_count": int(family.startswith("D")),
            "five_option": family == "D2" and depth == 3,
            "comparison_cell": f"cell_{index % 10}",
        })
    features = pd.DataFrame(rows)
    proposals = generate_proposals(features, PROFILES["balanced"], seed=7,
                                   proposals=2, iterations=5, keep=1)
    table = candidate_table(features, proposals[0][1])

    assert len(table) == 50
    assert table.maze_id.nunique() == 50
    assert table.task_id.nunique() == 50
    assert int(table.is_b.sum()) >= PROFILES["balanced"].b_min
