"""Compass overlay for the 3D views that turn with the agent (no mujoco import).

Drawn onto the rendered frame with plain PIL shapes. Letters are line strokes,
not a font, so frames stay byte-identical across machines and Pillow versions.
The compass turns with the view: the facing direction sits at the top in the
highlight colour, and the red needle points north.
"""

from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image, ImageDraw

COMPASS_CAMERAS: tuple[str, ...] = ("chase", "first_person", "first_person_narrow")


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


# ---------------------------------------------------------------------------
# Frozen-tile frost. Stepping on a frozen tile swallows the next N actions
# (PR #57 rulebook; the team chose visible no-ops over a silent budget
# deduction). The frost fills the screen on step-on, then each swallowed
# action thins it from the centre out, pales it, and cracks it, until the
# thaw frame is clean. Deterministic: fixed crack seed, no fonts.

FROST = (214, 232, 246)
CRACK = (58, 82, 110)
_CRACK_SEED = 7
_CRACK_COUNT = 10


def _cracks(resolution: int) -> list[list[tuple[float, float]]]:
    """Fixed polylines from the frame edge toward the middle (unit coords)."""
    rng = random.Random(_CRACK_SEED)
    cracks = []
    for _ in range(_CRACK_COUNT):
        side = rng.randrange(4)
        t = rng.uniform(0.1, 0.9)
        start = {0: (t, 0.0), 1: (1.0, t), 2: (t, 1.0), 3: (0.0, t)}[side]
        x, y = start
        points = [(x, y)]
        cx, cy = 0.5 + rng.uniform(-0.2, 0.2), 0.5 + rng.uniform(-0.2, 0.2)
        for k in range(1, 5):
            f = k / 5
            points.append((x + (cx - x) * f + rng.uniform(-0.05, 0.05), y + (cy - y) * f + rng.uniform(-0.05, 0.05)))
        cracks.append([(px * resolution, py * resolution) for px, py in points])
    return cracks


def draw_frost(frame: np.ndarray, remaining: int, total: int) -> np.ndarray:
    """A copy of ``frame`` frosted for ``remaining`` of ``total`` swallowed actions."""
    if remaining <= 0 or total <= 0:
        return frame
    frac = min(1.0, remaining / total)
    resolution = frame.shape[1]
    yy, xx = np.mgrid[0:frame.shape[0], 0:frame.shape[1]]
    half = resolution / 2.0
    dist = np.sqrt(((xx - half) / half) ** 2 + ((yy - half) / half) ** 2)  # 0 centre, ~1.41 corners
    clear_radius = (1.0 - frac) * 1.25  # the thaw opens from the centre outwards
    inside = dist >= clear_radius
    alpha = np.clip((0.25 + 0.6 * frac) * np.clip((dist - clear_radius) * 3.0 + 0.6, 0.0, 1.0), 0.0, 0.92)
    alpha = np.where(inside, alpha, 0.0)[..., None]
    out = (frame.astype(float) * (1.0 - alpha) + np.array(FROST, dtype=float) * alpha).round().astype(np.uint8)
    n_cracks = round((1.0 - frac) * _CRACK_COUNT)
    if n_cracks:
        image = Image.fromarray(out)
        draw = ImageDraw.Draw(image)
        width = max(1, round(resolution / 256))
        for crack in _cracks(resolution)[:n_cracks]:
            draw.line(crack, fill=CRACK, width=width)
        cracked = np.array(image)
        out = np.where(inside[..., None], cracked, out)
    return out


def crack_px(frame: np.ndarray) -> int:
    return int(np.all(frame == np.array(CRACK, dtype=frame.dtype), axis=-1).sum())


# ---------------------------------------------------------------------------
# Ride-direction glyph: while the agent stands on a rotating tile, MOVE_FORWARD
# carries it along the tile's arrow, which the eye cannot see when it points
# sideways or back. On the views that turn with the agent a disc in the
# top-left corner shows that direction relative to the view (up = ahead).

RIDE = (255, 150, 40)


def ride_box(resolution: int) -> tuple[int, int, int, int]:
    """(top, left, bottom, right) pixels of the ride glyph: the compass's mirror."""
    size = round(0.16 * resolution)
    pad = round(0.02 * resolution)
    return pad, pad, pad + size, pad + size


def draw_ride_arrow(frame: np.ndarray, relative_direction: int) -> np.ndarray:
    """A copy of ``frame`` with the ride arrow; 0 = ahead, 1 = right, 2 = back, 3 = left."""
    resolution = frame.shape[1]
    top, left, bottom, right = ride_box(resolution)
    radius = (bottom - top) / 2
    cx, cy = left + radius, top + radius
    width = max(1, round(resolution / 256))
    angle = 90.0 * (int(relative_direction) % 4)  # clockwise from up

    def at(a: float, distance: float) -> tuple[float, float]:
        r = math.radians(a)
        return cx + distance * math.sin(r), cy - distance * math.cos(r)

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.ellipse((left, top, right - 1, bottom - 1), fill=DISC, outline=RING, width=width)
    draw.polygon(
        [at(angle, 0.7 * radius), at(angle + 140, 0.55 * radius), at(angle, 0.15 * radius), at(angle - 140, 0.55 * radius)],
        fill=RIDE,
    )
    return np.array(image)


def ride_glyph_px(region: np.ndarray) -> int:
    return int(np.all(region == np.array(RIDE, dtype=region.dtype), axis=-1).sum())
