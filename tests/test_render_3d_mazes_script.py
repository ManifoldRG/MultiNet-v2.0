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
