"""Camera presets for the 3D maze renderer: pure pose math, no mujoco import.

World frame: spec cell (x, y) (y points south) has its centre at world
(x + 0.5, -(y + 0.5)); z is up. A pose follows MuJoCo's free camera: it looks
at ``lookat`` from ``distance`` away, facing ``azimuth`` degrees in the
xy-plane (0 = +x/east, 90 = +y/north), pitched ``elevation`` degrees
(negative = down). ``project`` reproduces MuJoCo's projection (verified to
< 1 px), so auto-fit and tests reason about pixels without rendering.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np

PRESETS: tuple[str, ...] = ("top_down", "chase", "fixed_angled", "first_person")
DEFAULT_WALL_HEIGHT: dict[str, float] = {
    "top_down": 0.4,
    "chase": 0.6,
    "fixed_angled": 0.6,
    "first_person": 1.4,
}

# GridState.agent_direction (0=E, 1=S, 2=W, 3=N) -> world yaw in degrees.
DIRECTION_YAW: dict[int, float] = {0: 0.0, 1: -90.0, 2: 180.0, 3: 90.0}

PERSPECTIVE_FOVY = 45.0  # MuJoCo's default free-camera fovy (degrees)
FIRST_PERSON_FOVY = 90.0
FIT_MARGIN = 0.06  # fraction of the half-frame kept clear around the maze
FIT_TOLERANCE = 1e-9  # float-rounding slack: a corner placed exactly on the margin still fits
CHASE_ELEVATION = -55.0
FIXED_ELEVATION = -50.0
EYE_HEIGHT = 0.55
FIRST_PERSON_ELEVATION = -12.0
FIRST_PERSON_FORWARD = 0.1  # eye sits slightly ahead of the agent centre


@dataclass(frozen=True)
class CameraPose:
    lookat: tuple[float, float, float]
    distance: float
    azimuth: float
    elevation: float
    orthographic: bool
    fovy: float  # degrees (perspective) or full visible height in world units (orthographic)


def cell_center(x, y, z: float = 0.0) -> tuple[float, float, float]:
    return (float(x) + 0.5, -(float(y) + 0.5), float(z))


def basis(azimuth: float, elevation: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(forward, right, up) unit vectors of a camera with this azimuth/elevation."""
    az, el = math.radians(azimuth), math.radians(elevation)
    forward = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    right = np.array([math.sin(az), -math.cos(az), 0.0])
    up = np.cross(right, forward)
    return forward, right, up


def camera_position(pose: CameraPose) -> np.ndarray:
    forward, _, _ = basis(pose.azimuth, pose.elevation)
    return np.asarray(pose.lookat, dtype=float) - pose.distance * forward


def project(pose: CameraPose, point) -> tuple[float, float]:
    """Normalised image coords of ``point`` in a square frame: u right, v up,
    both within [-1, 1] when visible. Points behind a perspective camera -> inf."""
    forward, right, up = basis(pose.azimuth, pose.elevation)
    p = np.asarray(point, dtype=float)
    if pose.orthographic:
        v = p - np.asarray(pose.lookat, dtype=float)
        half = pose.fovy / 2.0
        return float(v @ right / half), float(v @ up / half)
    v = p - camera_position(pose)
    depth = float(v @ forward)
    if depth <= 1e-6:
        return math.inf, math.inf
    half = math.tan(math.radians(pose.fovy) / 2.0)
    return float(v @ right / depth / half), float(v @ up / depth / half)


def to_pixel(uv: tuple[float, float], resolution: int) -> tuple[float, float]:
    """(row, col) position of normalised coords in a resolution x resolution frame."""
    u, v = uv
    return (1.0 - v) / 2.0 * resolution, (u + 1.0) / 2.0 * resolution


def maze_corners(width, height, wall_height) -> list[tuple[float, float, float]]:
    return [
        (x, y, z)
        for x in (0.0, float(width))
        for y in (0.0, -float(height))
        for z in (0.0, float(wall_height))
    ]


def fits(pose: CameraPose, points, margin: float = FIT_MARGIN) -> bool:
    limit = 1.0 - margin + FIT_TOLERANCE
    return all(abs(u) <= limit and abs(v) <= limit for u, v in (project(pose, p) for p in points))


def _fit_distance(make_pose, points, lo: float = 0.5, hi: float = 500.0) -> CameraPose:
    """Smallest camera distance (binary search) at which every point fits."""
    if not fits(make_pose(hi), points):
        raise ValueError("maze does not fit in frame even at maximum camera distance")
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if fits(make_pose(mid), points):
            hi = mid
        else:
            lo = mid
    return make_pose(hi)


@functools.lru_cache(maxsize=64)
def _chase_distance(width: int, height: int, wall_height: float) -> float:
    """One chase distance per maze: aimed at the maze centre, it fits the whole
    maze at every heading, so steps never re-zoom and turns only rotate."""
    centre = (width / 2.0, -height / 2.0, 0.0)
    corners = maze_corners(width, height, wall_height)
    return max(
        _fit_distance(
            lambda d, a=yaw: CameraPose(centre, d, a, CHASE_ELEVATION, False, PERSPECTIVE_FOVY), corners
        ).distance
        for yaw in DIRECTION_YAW.values()
    )


def _heading(direction: int) -> tuple[float, float, float]:
    if direction not in DIRECTION_YAW:
        raise ValueError(f"agent_direction must be 0-3, got {direction!r}")
    yaw = DIRECTION_YAW[direction]
    return yaw, math.cos(math.radians(yaw)), math.sin(math.radians(yaw))


def pose_for(
    preset: str, *, agent_cell, direction: int, maze_dims, wall_height: float, yaw: float | None = None
) -> CameraPose:
    """``yaw`` (degrees) overrides the heading's yaw for the views that turn
    with the agent; the demo uses it to draw frames partway through a turn."""
    if preset not in PRESETS:
        raise ValueError(f"unknown camera preset {preset!r}; choose from {PRESETS}")
    width, height = maze_dims
    corners = maze_corners(width, height, wall_height)
    centre = (width / 2.0, -height / 2.0, 0.0)

    if preset == "top_down":
        span = float(max(width, height))
        # Orthographic: fovy is the visible height; distance only needs to clear the walls.
        return CameraPose(centre, span + wall_height + 5.0, 90.0, -90.0, True, span / (1.0 - FIT_MARGIN))

    if preset == "fixed_angled":
        return _fit_distance(
            lambda d: CameraPose(centre, d, 90.0, FIXED_ELEVATION, False, PERSPECTIVE_FOVY), corners
        )

    heading_yaw, fx, fy = _heading(int(direction))
    if yaw is None:
        yaw = heading_yaw
    else:
        fx, fy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    ax, ay, _ = cell_center(*agent_cell)

    if preset == "chase":
        distance = _chase_distance(width, height, wall_height)
        return CameraPose(centre, distance, yaw, CHASE_ELEVATION, False, PERSPECTIVE_FOVY)

    # first_person: the free camera sits exactly at the eye (lookat - distance*forward).
    forward, _, _ = basis(yaw, FIRST_PERSON_ELEVATION)
    eye = np.array([ax + FIRST_PERSON_FORWARD * fx, ay + FIRST_PERSON_FORWARD * fy, EYE_HEIGHT])
    distance = 0.5
    lookat = eye + distance * forward
    return CameraPose(
        tuple(float(c) for c in lookat), distance, yaw, FIRST_PERSON_ELEVATION, False, FIRST_PERSON_FOVY
    )
