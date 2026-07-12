from analysis.mazes import maze_features


def test_empty_room_features(tmp_path):
    df = maze_features(maze_globs=("mazes/validation_10/V01_empty_room.json",),
                       cache=tmp_path / "m.parquet", force=True)
    r = df.iloc[0]
    assert r.task_id == "validation_10_v01_empty_room"
    assert r.is_beatable and r.optimal_steps > 0
    assert r.n_keys == 0 and r.n_switches == 0 and r.mechanism_count == 0
