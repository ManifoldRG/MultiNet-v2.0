"""Colour names used in task specs -> RGBA for the 3D scene (no minigrid import).

Unknown names raise: a silent fallback colour would make two mechanisms look
alike (the OGBench fork falls back to yellow).
"""

from __future__ import annotations

RGBA = tuple[float, float, float, float]

_SPEC_RGB: dict[str, tuple[float, float, float]] = {
    "red": (0.86, 0.18, 0.16),
    "green": (0.18, 0.68, 0.28),
    "blue": (0.18, 0.38, 0.90),
    "purple": (0.55, 0.24, 0.80),
    "yellow": (0.96, 0.80, 0.12),
    "grey": (0.58, 0.58, 0.62),
    "black": (0.10, 0.10, 0.12),
    "white": (0.95, 0.95, 0.95),
    "cyan": (0.0, 0.86, 0.86),  # MiniGrid's SWITCH_RENDER_COLORS cyan
}

# Scene fixtures (not spec colours).
FLOOR: RGBA = (0.82, 0.82, 0.80, 1.0)
GRID_LINE: RGBA = (0.52, 0.53, 0.57, 1.0)
WALL: RGBA = (0.23, 0.24, 0.31, 1.0)
GOAL: RGBA = (0.20, 0.80, 0.35, 1.0)
AGENT: RGBA = (0.10, 0.85, 0.90, 1.0)
SWITCH_PLATE: RGBA = (0.20, 0.20, 0.22, 1.0)


def _rgb255(r: int, g: int, b: int) -> RGBA:
    return (r / 255.0, g / 255.0, b / 255.0, 1.0)


# PR #57 tiles: the MiniGrid glyph colours (custom_env.py), so 2D and 3D match.
PORTAL_CORE: RGBA = _rgb255(20, 20, 35)
KILL_DISC: RGBA = _rgb255(90, 12, 12)
SKULL: RGBA = _rgb255(230, 230, 230)
SKULL_DARK: RGBA = _rgb255(20, 20, 20)
ROTATOR_TILE: RGBA = _rgb255(255, 168, 64)
ROTATOR_ARROW: RGBA = _rgb255(40, 28, 16)
FROZEN_TILE: RGBA = _rgb255(214, 232, 246)
FROZEN_SPOT: RGBA = _rgb255(255, 255, 255)
FROZEN_FLECK: RGBA = _rgb255(245, 252, 255)


def normalize(name: str) -> str:
    key = name.strip().lower()
    return "grey" if key == "gray" else key


def rgba(name: str, alpha: float = 1.0) -> RGBA:
    key = normalize(name)
    if key not in _SPEC_RGB:
        raise ValueError(f"unknown colour {name!r}; known: {sorted(_SPEC_RGB)}")
    r, g, b = _SPEC_RGB[key]
    return (r, g, b, alpha)


def dim(color: RGBA, factor: float) -> RGBA:
    r, g, b, a = color
    return (r * factor, g * factor, b * factor, a)
