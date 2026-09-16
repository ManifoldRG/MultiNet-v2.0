"""Compass overlay for the 3D views that turn with the agent (no mujoco import).

Drawn onto the rendered frame with plain PIL shapes. Letters are line strokes,
not a font, so frames stay byte-identical across machines and Pillow versions.
The compass turns with the view: the facing direction sits at the top in the
highlight colour, and the red needle points north.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

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


def draw_compass(frame: np.ndarray, direction: int) -> np.ndarray:
    """A copy of ``frame`` with the compass for an agent facing ``direction``."""
    resolution = frame.shape[1]
    top, left, bottom, right = compass_box(resolution)
    radius = (bottom - top) / 2
    cx, cy = left + radius, top + radius
    width = max(1, round(resolution / 256))
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
