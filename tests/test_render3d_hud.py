"""The compass drawn on the 3D views that turn with the agent (no mujoco needed)."""

from __future__ import annotations

import numpy as np
import pytest

from gridworld.render3d.hud import (
    COMPASS_CAMERAS,
    DISC,
    HIGHLIGHT,
    NEEDLE,
    compass_box,
    draw_compass,
    turns_with_agent,
)

RES = 256


def _drawn(direction: int) -> np.ndarray:
    return draw_compass(np.zeros((RES, RES, 3), np.uint8), direction)


def _centre() -> tuple[float, float]:
    top, left, bottom, right = compass_box(RES)
    return (top + bottom) / 2, (left + right) / 2


def _pixels(frame, colour) -> tuple[np.ndarray, np.ndarray]:
    return np.nonzero((frame == np.array(colour, np.uint8)).all(axis=-1))


def test_compass_stays_in_a_small_top_right_box():
    top, left, bottom, right = compass_box(RES)
    assert top >= 0 and right <= RES and left > RES // 2 and bottom - top < RES // 4
    frame = _drawn(0)
    outside = frame.copy()
    outside[top:bottom, left:right] = 0
    assert not outside.any()
    assert _pixels(frame, DISC)[0].size > 0


@pytest.mark.parametrize(
    "direction, north",
    [(3, "up"), (1, "down"), (0, "left"), (2, "right")],  # 0=E 1=S 2=W 3=N
)
def test_needle_points_north_relative_to_the_facing(direction, north):
    rows, cols = _pixels(_drawn(direction), NEEDLE)
    cr, cc = _centre()
    dr, dc = rows.mean() - cr, cols.mean() - cc
    assert {"up": -dr, "down": dr, "left": -dc, "right": dc}[north] > 2 * min(abs(dr), abs(dc)) + 1


@pytest.mark.parametrize("direction", range(4))
def test_facing_letter_is_highlighted_at_the_top(direction):
    rows, cols = _pixels(_drawn(direction), HIGHLIGHT)
    cr, cc = _centre()
    assert rows.size > 0
    assert rows.mean() < cr and abs(cols.mean() - cc) < cr - rows.mean()


def test_each_heading_draws_a_different_repeatable_compass():
    frames = [_drawn(d) for d in range(4)]
    assert len({f.tobytes() for f in frames}) == 4
    assert _drawn(2).tobytes() == frames[2].tobytes()


def test_compass_is_for_the_views_that_turn_with_the_agent():
    assert set(COMPASS_CAMERAS) == {"chase", "first_person"}
    assert turns_with_agent("chase") and not turns_with_agent("top_down")
    # a demo tilt level overrides the preset: only level 0 (top-down) stays north-up
    assert not turns_with_agent("chase", tilt=0) and turns_with_agent("top_down", tilt=1)
