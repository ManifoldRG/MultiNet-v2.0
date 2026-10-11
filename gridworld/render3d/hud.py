"""HUD overlays for the 3D views (no mujoco import).

- A compass on the views that turn with the agent: the facing direction sits
  at the top in the highlight colour, and the red needle points north.
- Status slots on the first-person view, centred along the bottom edge: the
  key the agent carries and the switch it stands on. The eye cannot see the
  agent's own cell, where PICKUP and a switch TOGGLE act, so without them
  neither action changes the frame.

Drawn onto the rendered frame with plain PIL shapes. Letters are line strokes,
not a font, so frames stay byte-identical across machines and Pillow versions.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from . import palette

COMPASS_CAMERAS: tuple[str, ...] = ("chase", "first_person")


def turns_with_agent(camera: str, tilt: int | None = None) -> bool:
    """Whether the view turns with the agent (and so carries a compass):
    every demo tilt level but the top-down one, else the turning presets."""
    if tilt is not None:
        return tilt > 0
    return camera in COMPASS_CAMERAS

DISC = (16, 18, 26)
RING = (150, 156, 170)
LETTER = RING
HIGHLIGHT = (255, 214, 10)
NEEDLE = (228, 58, 50)

_CLOCKWISE = "NESW"
_CLOCKWISE_INDEX = {3: 0, 0: 1, 1: 2, 2: 3}  # GridState.agent_direction: 0=E 1=S 2=W 3=N
NORTH = 3  # the "up" a north-up view shows

# Letter strokes in a unit box (x right, y down).
_GLYPHS = {
    "N": [[(0, 1), (0, 0), (1, 1), (1, 0)]],
    "E": [[(1, 0), (0, 0), (0, 1), (1, 1)], [(0, 0.5), (0.8, 0.5)]],
    "S": [[(1, 0.15), (0.8, 0), (0.2, 0), (0, 0.2), (0.2, 0.45), (0.8, 0.55), (1, 0.8), (0.8, 1), (0.2, 1), (0, 0.85)]],
    "W": [[(0, 0), (0.25, 1), (0.5, 0.45), (0.75, 1), (1, 0)]],
}


def compass_box(resolution: int) -> tuple[int, int, int, int]:
    """(top, left, bottom, right) pixels of the compass in a square frame."""
    size = round(0.16 * resolution)
    pad = round(0.02 * resolution)
    return pad, resolution - pad - size, pad + size, resolution - pad


def corner_compass_box(corner_cell: tuple[float, float, float, float], resolution: int) -> tuple[int, int, int, int]:
    """Top-down compass box: inside the top-right outer-wall cell plus the fit
    margin beyond it, so the compass never covers an interior cell (the old
    fixed box sat on the goal tile in 70 corpus mazes). ``corner_cell`` is
    that cell's (top, left, bottom, right) float pixel box
    (``cameras.cell_pixel_box``); only whole pixels right of its left edge and
    above its bottom edge are used."""
    _top, cell_left, cell_bottom, _right = corner_cell
    pad = max(1, round(0.005 * resolution))  # off the frame edge
    right, top = resolution - pad, pad
    size = min(right - math.ceil(cell_left - 1e-6), math.floor(cell_bottom + 1e-6) - top)
    return top, right - size, top + size, right


def draw_compass(
    frame: np.ndarray, direction: int, box: tuple[int, int, int, int] | None = None
) -> np.ndarray:
    """A copy of ``frame`` with the compass for an agent facing ``direction``,
    in ``box`` (top, left, bottom, right; default ``compass_box``)."""
    resolution = frame.shape[1]
    default = compass_box(resolution)
    top, left, bottom, right = box or default
    if bottom - top < 1 or right - left < 1:
        return frame.copy()  # no room left outside the maze (absurdly small frame)
    radius = (bottom - top) / 2
    cx, cy = left + radius, top + radius
    # Stroke width scales with the disc (1 px at the default box in a 256 px frame).
    width = max(1, round(resolution / 256 * (bottom - top) / (default[2] - default[0])))
    facing = _CLOCKWISE_INDEX[int(direction)]

    def at(angle: float, distance: float) -> tuple[float, float]:
        # angle in degrees, clockwise from the top of the frame
        a = math.radians(angle)
        return cx + distance * math.sin(a), cy - distance * math.cos(a)

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.ellipse((left, top, right - 1, bottom - 1), fill=DISC, outline=RING, width=width)
    north = -90.0 * facing
    draw.polygon(
        [at(north, 0.46 * radius), at(north - 90, 0.13 * radius), at(north + 90, 0.13 * radius)],
        fill=NEEDLE,
    )
    half = 0.15 * radius
    for index, letter in enumerate(_CLOCKWISE):
        lx, ly = at(90.0 * (index - facing), 0.64 * radius)
        colour = HIGHLIGHT if index == facing else LETTER
        for stroke in _GLYPHS[letter]:
            points = [(lx - half + 2 * half * x, ly - half + 2 * half * y) for x, y in stroke]
            draw.line(points, fill=colour, width=width)
    return np.array(image)


def status_boxes(resolution: int, count: int) -> list[tuple[int, int, int, int]]:
    """(top, left, bottom, right) of ``count`` status slots, side by side and
    centred along the bottom edge of a square frame."""
    size = round(0.12 * resolution)
    gap = round(0.02 * resolution)
    pad = round(0.02 * resolution)
    top = resolution - pad - size
    first = (resolution - (count * size + (count - 1) * gap)) // 2
    lefts = [first + i * (size + gap) for i in range(count)]
    return [(top, left, top + size, left + size) for left in lefts]


def _rgb(colour: palette.RGBA) -> tuple[int, int, int]:
    r, g, b, _alpha = colour
    return round(255 * r), round(255 * g), round(255 * b)


def _draw_key(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float, colour: str) -> None:
    """A key lying on its side: bow on the left, two teeth down at the tip."""
    fill = _rgb(palette.rgba(colour))
    bx = cx - 0.3 * r
    draw.ellipse((bx - 0.27 * r, cy - 0.27 * r, bx + 0.27 * r, cy + 0.27 * r), fill=fill)
    draw.ellipse((bx - 0.11 * r, cy - 0.11 * r, bx + 0.11 * r, cy + 0.11 * r), fill=DISC)
    draw.rectangle((bx + 0.2 * r, cy - 0.09 * r, cx + 0.6 * r, cy + 0.09 * r), fill=fill)
    draw.rectangle((cx + 0.33 * r, cy, cx + 0.45 * r, cy + 0.32 * r), fill=fill)
    draw.rectangle((cx + 0.5 * r, cy, cx + 0.6 * r, cy + 0.24 * r), fill=fill)


def _draw_switch(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float, colour: str, on: bool) -> None:
    """The switch as the scene draws it: a dark plate and a button, dim when
    off, lit with a halo when on."""
    lit = palette.rgba(colour)
    draw.rectangle((cx - 0.5 * r, cy - 0.5 * r, cx + 0.5 * r, cy + 0.5 * r), fill=_rgb(palette.SWITCH_PLATE))
    if on:
        draw.ellipse((cx - 0.44 * r, cy - 0.44 * r, cx + 0.44 * r, cy + 0.44 * r), outline=_rgb(lit),
                     width=max(1, round(0.08 * r)))
    button = _rgb(lit if on else palette.dim(lit, 0.35))
    draw.ellipse((cx - 0.28 * r, cy - 0.28 * r, cx + 0.28 * r, cy + 0.28 * r), fill=button)


def draw_status(
    frame: np.ndarray, *, carrying: str | None, switch: tuple[str, bool] | None
) -> np.ndarray:
    """A copy of ``frame`` with the first-person status slots: the colour of
    the key the agent carries, then the switch under it as ``(colour, on)``.
    Nothing to show: an unchanged copy."""
    slots = []
    if carrying:
        slots.append(lambda draw, cx, cy, r: _draw_key(draw, cx, cy, r, carrying))
    if switch is not None:
        slots.append(lambda draw, cx, cy, r: _draw_switch(draw, cx, cy, r, *switch))
    if not slots:
        return frame.copy()
    resolution = frame.shape[1]
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    for paint, (top, left, bottom, right) in zip(slots, status_boxes(resolution, len(slots))):
        draw.ellipse((left, top, right - 1, bottom - 1), fill=DISC, outline=RING,
                     width=max(1, round(resolution / 256)))
        radius = (bottom - top) / 2
        paint(draw, left + radius, top + radius, radius)
    return np.array(image)
