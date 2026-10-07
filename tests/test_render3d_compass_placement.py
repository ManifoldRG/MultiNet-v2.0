"""The top_down compass never covers a maze's interior (non-border) cells.

Measured on the old fixed top-right box (16% of the frame): the disc covered
the goal tile in 70 corpus mazes. The frame comes from the real 3D backend and
SceneRenderer composition, with only MuJoCo's GL rasterizer replaced by a
black frame, so every non-black pixel is the compass.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("minigrid")

from gridworld.backends import get_backend  # noqa: E402
from gridworld.render3d.cameras import pose_for, project, to_pixel  # noqa: E402
from gridworld.render3d.hud import compass_box, draw_compass  # noqa: E402
from gridworld.render_settings import GRID_PIXELS_PER_CELL  # noqa: E402
from gridworld.task_spec import TaskSpecification  # noqa: E402
from maze_test_utils import MAZE_JSON_DIR  # noqa: E402
from render3d_test_utils import CORRIDOR, REPO_ROOT  # noqa: E402

CORPUS = sorted(MAZE_JSON_DIR.rglob("*.json")) + sorted(
    (REPO_ROOT / "gridworld" / "fixtures").glob("test[0-9]/*.json")
)
EPS = 1e-6


class _BlackRenderer:
    """Stands in for mujoco.Renderer: the scene draws as black."""

    def __init__(self, model, height, width):
        self._shape = (height, width, 3)

    def update_scene(self, *args, **kwargs):
        pass

    def render(self):
        return np.zeros(self._shape, np.uint8)

    def close(self):
        pass


def _compass_mask(spec: TaskSpecification, camera: str, resolution: int) -> np.ndarray:
    backend = get_backend("mujoco3d", camera=camera, resolution=resolution)
    try:
        backend.configure(spec)
        frame, _state, _info = backend.reset(seed=spec.seed)
    finally:
        backend.close()
    return frame.any(axis=-1)


def _interior_hits(spec: TaskSpecification, mask: np.ndarray) -> int:
    """Compass pixels whose area overlaps any interior cell (x in 1..w-2,
    y in 1..h-2), using the top_down camera's own projection."""
    width, height = spec.maze.dimensions
    resolution = mask.shape[0]
    pose = pose_for("top_down", agent_cell=(1, 1), direction=0, maze_dims=(width, height), wall_height=0.4)
    top, left = to_pixel(project(pose, (1.0, -1.0, 0.0)), resolution)
    bottom, right = to_pixel(project(pose, (width - 1.0, -(height - 1.0), 0.0)), resolution)
    rows, cols = np.nonzero(mask)
    inside = (cols + 1 > left + EPS) & (cols < right - EPS) & (rows + 1 > top + EPS) & (rows < bottom - EPS)
    return int(inside.sum())


@pytest.fixture
def black_frames(monkeypatch):
    monkeypatch.setattr(mujoco, "Renderer", _BlackRenderer)


def test_top_down_compass_covers_no_interior_cell_anywhere_in_the_corpus(black_frames):
    assert len(CORPUS) >= 214
    offenders = {}
    for path in CORPUS:
        spec = TaskSpecification.from_json(str(path))
        mask = _compass_mask(spec, "top_down", GRID_PIXELS_PER_CELL * max(spec.maze.dimensions))
        assert mask.any(), f"{path.name}: top_down lost its compass"
        hits = _interior_hits(spec, mask)
        if hits:
            offenders[str(path.relative_to(REPO_ROOT))] = hits
    assert not offenders, f"{len(offenders)} mazes: compass over interior cells {offenders}"


# 8x8 room whose goal sits in the top-right interior cell, under the old box.
ROOM_8X8 = {
    "task_id": "compass_room_8x8", "seed": 0, "difficulty_tier": 1,
    "maze": {"dimensions": [8, 8], "walls": [], "start": [1, 6], "goal": [6, 1]},
    "mechanisms": {}, "goal": {"type": "reach_position", "target": [6, 1]}, "max_steps": 60,
}


@pytest.mark.parametrize("resolution", [64, 100, 512])
def test_top_down_compass_clears_the_interior_at_other_frame_sizes(black_frames, resolution):
    spec = TaskSpecification.from_dict(ROOM_8X8)
    mask = _compass_mask(spec, "top_down", resolution)
    assert mask.any() and _interior_hits(spec, mask) == 0


@pytest.mark.parametrize("camera", ["first_person", "first_person_narrow"])
def test_other_views_keep_the_fixed_top_right_compass(black_frames, camera):
    """Only top_down moves: the perspective views draw exactly as before."""
    spec = TaskSpecification.from_dict(CORRIDOR)
    resolution = 224
    mask = _compass_mask(spec, camera, resolution)
    blank = np.zeros((resolution, resolution, 3), np.uint8)
    expected = draw_compass(blank, 0).any(axis=-1)  # the eye views turn with the agent (E heading)
    np.testing.assert_array_equal(mask, expected)
    top, left, bottom, right = compass_box(resolution)
    assert mask[top:bottom, left:right].any() and not mask.sum() > (bottom - top) * (right - left)
