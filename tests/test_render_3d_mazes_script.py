from __future__ import annotations

import json

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("minigrid")

from PIL import Image  # noqa: E402

from render3d_test_utils import CORRIDOR  # noqa: E402
from scripts.render_3d_mazes import main  # noqa: E402

BFS_TOKENS = [None, "MOVE_FORWARD", "PICKUP", "TOGGLE", "MOVE_FORWARD", "MOVE_FORWARD", "MOVE_FORWARD"]


def test_bfs_replay_writes_frames_index_and_contact_sheet(tmp_path):
    maze = tmp_path / "corridor.json"
    maze.write_text(json.dumps(CORRIDOR))
    out = tmp_path / "out"
    code = main([
        "--mazes", str(maze), "--camera", "top_down", "--camera", "chase",
        "--resolution", "64", "--replay", "bfs", "--contact-sheet", "--out", str(out),
    ])
    assert code == 0
    index = json.loads((out / "index.json").read_text())
    assert len(index) == 2 * len(BFS_TOKENS)
    assert {r["camera"] for r in index} == {"top_down", "chase"}
    assert [r["action"] for r in index if r["camera"] == "chase"] == BFS_TOKENS
    assert all((out / r["png"]).exists() for r in index)
    assert Image.open(out / "contact_sheet.png").size == (2 * 256, 256 + 18)


def test_manifest_selection_renders_start_frames(tmp_path):
    out = tmp_path / "out"
    code = main([
        "--manifest", "gridworld/fixtures/manifest.json", "--experiment", "test1",
        "--resolution", "32", "--out", str(out),
    ])
    assert code == 0
    index = json.loads((out / "index.json").read_text())
    assert index and all(r["step"] == 0 and r["camera"] == "top_down" for r in index)


def test_no_mazes_is_a_usage_error(tmp_path):
    with pytest.raises(SystemExit):
        main(["--out", str(tmp_path)])


def test_manifest_row_with_missing_source_is_skipped_with_warning(tmp_path, capsys):
    # Create a manifest with one valid row and one missing row
    maze = tmp_path / "corridor.json"
    maze.write_text(json.dumps(CORRIDOR))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "tasks": [
            {
                "task_id": "good_maze",
                "experiment": "test1",
                "condition": "default",
                "variant": "corridor",
                "source": "corridor.json",
            },
            {
                "task_id": "missing_maze",
                "experiment": "test1",
                "condition": "default",
                "variant": "missing",
                "source": "nonexistent.json",
            },
        ]
    }))
    out = tmp_path / "out"
    code = main([
        "--manifest", str(manifest), "--experiment", "test1",
        "--resolution", "32", "--out", str(out),
    ])
    assert code == 0
    index = json.loads((out / "index.json").read_text())
    # Should have only one maze rendered (the good one)
    assert len(index) == 1
    # The task_id comes from the TaskSpecification loaded from the maze file, which is from CORRIDOR
    assert index[0]["task_id"] == "render3d_corridor"
    assert index[0]["step"] == 0
    # Should have warning in stderr about skipping missing row
    captured = capsys.readouterr()
    assert "skipping manifest row" in captured.err
