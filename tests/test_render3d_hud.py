"""The compass drawn on the 3D views that turn with the agent (no mujoco needed)."""

from __future__ import annotations

import numpy as np
import pytest

from gridworld.render3d import palette
from gridworld.render3d.hud import (
    COMPASS_CAMERAS,
    DISC,
    HIGHLIGHT,
    NEEDLE,
    compass_box,
    draw_compass,
    draw_status,
    status_boxes,
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
    assert set(COMPASS_CAMERAS) == {"first_person", "first_person_narrow"}
    assert turns_with_agent("first_person") and not turns_with_agent("top_down")
    # a demo tilt level overrides the preset: only level 0 (top-down) stays north-up
    assert not turns_with_agent("first_person", tilt=0) and turns_with_agent("top_down", tilt=1)


@pytest.mark.parametrize("resolution, dims", [(8, (14, 14)), (16, (14, 14)), (16, (13, 3)), (8, (8, 8))])
def test_top_down_corner_box_too_small_for_a_compass_draws_none(resolution, dims):
    """At absurdly small frames the corner wall cell + margin has no whole
    pixel to spare: skip the compass rather than crash or cover the maze."""
    from gridworld.render3d.cameras import cell_pixel_box, pose_for
    from gridworld.render3d.hud import corner_compass_box

    pose = pose_for("top_down", agent_cell=(1, 1), direction=0, maze_dims=dims, wall_height=0.4)
    box = corner_compass_box(cell_pixel_box(pose, (dims[0] - 1, 0), resolution), resolution)
    blank = np.zeros((resolution, resolution, 3), np.uint8)
    assert not draw_compass(blank, 3, box=box).any()


# --- first-person status slots (carried key, switch underfoot) --------------


def _rgb(name: str, factor: float = 1.0) -> tuple[int, int, int]:
    r, g, b, _alpha = palette.dim(palette.rgba(name), factor)
    return round(255 * r), round(255 * g), round(255 * b)


def _inside(rows, cols, box) -> bool:
    top, left, bottom, right = box
    return bool(((rows >= top) & (rows < bottom) & (cols >= left) & (cols < right)).all())


@pytest.mark.parametrize("count", [1, 2])
def test_status_slots_sit_side_by_side_centred_along_the_bottom(count):
    boxes = status_boxes(RES, count)
    assert len(boxes) == count
    assert len({(top, bottom) for top, _l, bottom, _r in boxes}) == 1  # one row
    top, bottom = boxes[0][0], boxes[0][2]
    assert top > 0.8 * RES and RES - bottom < 0.05 * RES  # along the bottom edge
    assert abs(boxes[0][1] + boxes[-1][3] - RES) <= 1  # centred, not on a side
    assert all(a[3] <= b[1] for a, b in zip(boxes, boxes[1:]))  # side by side


def test_nothing_to_show_leaves_the_frame_as_it_was():
    frame = np.zeros((RES, RES, 3), np.uint8)
    out = draw_status(frame, carrying=None, switch=None)
    assert out is not frame and out.tobytes() == frame.tobytes()


def test_a_carried_key_is_drawn_in_its_colour_in_the_bottom_slot():
    out = draw_status(np.zeros((RES, RES, 3), np.uint8), carrying="red", switch=None)
    rows, cols = _pixels(out, _rgb("red"))
    assert rows.size > 20 and _inside(rows, cols, status_boxes(RES, 1)[0])


def test_the_switch_slot_lights_up_when_the_switch_is_on():
    blank = np.zeros((RES, RES, 3), np.uint8)
    off = draw_status(blank, carrying=None, switch=("yellow", False))
    on = draw_status(blank, carrying=None, switch=("yellow", True))
    box = status_boxes(RES, 1)[0]
    assert _pixels(off, _rgb("yellow"))[0].size == 0  # off: only the dimmed button
    assert _pixels(off, _rgb("yellow", 0.35))[0].size > 20
    rows, cols = _pixels(on, _rgb("yellow"))
    assert rows.size > 20 and _inside(rows, cols, box)


def test_key_and_switch_share_the_bottom_row():
    out = draw_status(np.zeros((RES, RES, 3), np.uint8), carrying="red", switch=("yellow", True))
    key_box, switch_box = status_boxes(RES, 2)
    assert _inside(*_pixels(out, _rgb("red")), key_box)
    assert _inside(*_pixels(out, _rgb("yellow")), switch_box)
