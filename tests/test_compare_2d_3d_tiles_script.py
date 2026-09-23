from __future__ import annotations

import json

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("minigrid")

from PIL import Image  # noqa: E402

from gridworld.task_spec import TaskSpecification  # noqa: E402
from render3d_test_utils import TILES  # noqa: E402
from scripts.compare_2d_3d_tiles import facing_pose, main, tile_catalog  # noqa: E402

DIRECTION_VECTORS = ((1, 0), (0, 1), (-1, 0), (0, -1))


def test_tile_catalog_lists_every_new_tile_once():
    catalog = tile_catalog(TaskSpecification.from_dict(TILES))
    assert [c.cell for c in catalog] == [(3, 1), (7, 3), (3, 3), (7, 1), (5, 1), (5, 3), (4, 2), (6, 2)]
    assert [c.label for c in catalog][:2] == ["portal tp (purple) A", "portal tp (purple) B"]
    assert catalog[4].label == "kill cell 0"
    assert catalog[5].label == "frozen tile 0"
    assert catalog[6].label == "rotating tile 0 (E)"
    assert catalog[7].label == "rotating tile 1 (N)"


def test_facing_pose_stands_on_a_free_neighbour_looking_at_the_tile():
    spec = TaskSpecification.from_dict(TILES)
    for entry in tile_catalog(spec):
        cell, direction = facing_pose(spec, entry.cell)
        dx, dy = DIRECTION_VECTORS[direction]
        assert (cell[0] + dx, cell[1] + dy) == entry.cell
        assert 0 < cell[0] < 10 and 0 < cell[1] < 4  # inside the border


def test_facing_pose_tries_west_north_east_south():
    assert facing_pose(TaskSpecification.from_dict(TILES), (5, 1)) == ((4, 1), 0)
    d = json.loads(json.dumps(TILES))
    d["maze"]["walls"] = [[4, 1]]  # west of the kill cell (5,1) is a wall; north (5,0) is border
    assert facing_pose(TaskSpecification.from_dict(d), (5, 1)) == ((6, 1), 2)


def test_facing_pose_rejects_a_boxed_in_tile():
    d = json.loads(json.dumps(TILES))
    d["maze"]["walls"] = [[4, 1], [6, 1], [5, 2]]  # (5,0) is the border
    with pytest.raises(ValueError, match="no free neighbour"):
        facing_pose(TaskSpecification.from_dict(d), (5, 1))


def test_main_writes_a_sheet_per_maze_and_an_index(tmp_path):
    maze = tmp_path / "tiles.json"
    maze.write_text(json.dumps(TILES))
    out = tmp_path / "out"
    assert main(["--mazes", str(maze), "--resolution", "96", "--tile-size", "8", "--out", str(out)]) == 0
    index = json.loads((out / "index.json").read_text())
    assert [m["task_id"] for m in index] == ["render3d_tiles"]
    maze_dir = out / "render3d_tiles"
    assert Image.open(maze_dir / "topdown_2d.png").size == (11 * 8, 5 * 8)
    assert Image.open(maze_dir / "topdown_3d.png").size == (96, 96)
    tiles = index[0]["tiles"]
    assert len(tiles) == 8
    for t in tiles:
        assert Image.open(out / t["first_person_png"]).size == (96, 96)
        assert Image.open(out / t["glyph_2d_png"]).size == (8, 8)
        assert t["viewer_cell"] != t["cell"]
    assert (maze_dir / "sheet.png").exists()
