"""2D top-down frames in the MiniGrid look, for any tiling the R1 rulebook covers.

A frame is drawn from the task spec and a ``PlannerState`` the way MiniGrid
draws its grid: 3x supersampled, black floor with grey grid lines, MiniGrid's
own renders for the key, switch, portal, skull, ice, lava and box icons, doors
and gates as MiniGrid's frames and rings, and the red agent triangle and the
rotating tile's arrow turned by the direction angle (90 degrees a step on
square, 60 on hex, where both triangles are drawn narrower so that each of
the six directions reads one way). The rotating tile under the agent and the frost overlay are
then drawn at full scale on top, as ``CustomMiniGridEnv.get_frame`` does.

On a square task the frame is MiniGridBackend's (tests/test_render2d.py checks
it pixel for pixel). On hex only the cell shape changes: pointy-top hexagons in
the odd-r layout, sized so a cell covers about MiniGrid's 32x32 px.

Shapes are MiniGrid's ``fill_coords`` predicates, evaluated in each cell's
"tile" coordinates: (u, v) in [0, 1] across a square of the cell's width,
centred on the cell. A hex is taller than it is wide, so its top and bottom
tips reach v = 0.5 -/+ 0.577.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .custom_env import (
    MINIGRID_COLORS,
    FrozenTile,
    GroundKey,
    KillCell,
    PushableBlock,
    Switch,
    TeleporterObj,
)
from .task_spec import TaskSpecification
from .world_model import PlannerState, TaskPlanningContext

from minigrid.core.constants import COLORS, TILE_PIXELS
from minigrid.core.world_object import Lava

# Bump on any visual change: it is in the multigrid run label and run hash, so
# a new look writes new episodes instead of reusing cached ones.
RENDER2D_VERSION = 1

# MiniGrid's supersampling factor (Grid.render_tile subdivs).
SUBDIVS = 3
# Hex centre-to-vertex size in px. A pointy-top hex of size s covers
# 3*sqrt(3)/2 * s^2 px: 1039 at 20, against MiniGrid's 32 * 32 = 1024.
HEX_SIZE = 20

GREY = np.array([100, 100, 100], dtype=np.uint8)  # walls and grid lines
FLOOR = np.array([0, 0, 0], dtype=np.uint8)
GOAL = COLORS["green"].astype(np.uint8)
ICE = np.array([214, 232, 246], dtype=np.uint8)  # FrozenTile and the frost overlay
TURNTABLE = np.array([255, 168, 64], dtype=np.uint8)  # RotatingTile
ARROW = np.array([40, 28, 16], dtype=np.uint8)
LAVA = np.array([255, 128, 0], dtype=np.uint8)
AGENT = np.array([255, 0, 0], dtype=np.uint8)
CRACK = np.array([58, 82, 110], dtype=np.uint8)

# Grid.render_tile draws a 0.031-tile line on each tile's top and left edge.
_LINE = 0.031
# MiniGrid's agent triangle, pointing east, and RotatingTile's east arrow; the
# small triangle CustomMiniGridEnv.get_frame draws on a rotating tile.
_AGENT_TRIANGLE = ((0.12, 0.19), (0.87, 0.50), (0.12, 0.81))
_ARROW_TRIANGLE = ((0.22, 0.22), (0.22, 0.78), (0.82, 0.50))
_SMALL_AGENT_TRIANGLE = ((0.32, 0.35), (0.68, 0.50), (0.32, 0.65))
# MiniGrid's triangles are close to equilateral (the agent's apex is 45
# degrees, the arrow's 50), so turned by 120 degrees they look like themselves
# shifted: fine in 90-degree steps, ambiguous in 60-degree ones (the agent
# facing northwest reads as east). Hex draws the same triangles narrower, apex
# 30-37 degrees, so each of the six directions reads one way.
_HEX_AGENT_TRIANGLE = ((0.15, 0.30), (0.87, 0.50), (0.15, 0.70))
_HEX_ARROW_TRIANGLE = ((0.22, 0.30), (0.22, 0.70), (0.82, 0.50))
_HEX_SMALL_AGENT_TRIANGLE = ((0.32, 0.40), (0.70, 0.50), (0.32, 0.60))
_ICON_SENTINEL = np.array([1, 2, 3], dtype=np.uint8)


@dataclass(frozen=True)
class _Geometry:
    """Pixel-to-cell maps for one tiling at one scale."""

    shape: tuple[int, int]
    width: int  # cells per row
    tile: float  # cell width in px: the side of the (u, v) unit square
    cell_index: np.ndarray  # (H, W) cell id y * width + x, or n_cells outside
    pixels: tuple[np.ndarray, ...]  # flat pixel indices of each cell
    line_band: np.ndarray  # (H, W) bool: the cell's own share of a grid line
    centres: np.ndarray | None = None  # (n_cells, 2) px; None = square tiles

    def local(self, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cell ``k``'s pixels and their tile coordinates (u, v): MiniGrid's
        (xf, yf) on square, the same frame centred on the hex otherwise."""
        pix = self.pixels[k]
        xs = pix % self.shape[1]
        ys = pix // self.shape[1]
        if self.centres is None:
            # Integer offsets first, as fill_coords computes (x + 0.5) / size.
            tile = int(self.tile)
            u = (xs - (k % self.width) * tile + 0.5) / tile
            v = (ys - (k // self.width) * tile + 0.5) / tile
        else:
            cx, cy = self.centres[k]
            u = (xs + 0.5 - cx) / self.tile + 0.5
            v = (ys + 0.5 - cy) / self.tile + 0.5
        return pix, u, v


def _cell_pixels(cell_index: np.ndarray, n_cells: int) -> tuple[np.ndarray, ...]:
    flat = cell_index.ravel()
    order = np.argsort(flat, kind="stable").astype(np.int32)
    bounds = np.searchsorted(flat[order], np.arange(n_cells + 1))
    return tuple(order[bounds[k]:bounds[k + 1]] for k in range(n_cells))


@lru_cache(maxsize=4)
def _square_geometry(width: int, height: int, tile: int) -> _Geometry:
    ys, xs = np.mgrid[0:height * tile, 0:width * tile]
    cell_index = ((ys // tile) * width + xs // tile).astype(np.int32)
    u = ((xs % tile) + 0.5) / tile
    v = ((ys % tile) + 0.5) / tile
    line_band = (u <= _LINE) | (v <= _LINE)
    return _Geometry(
        shape=cell_index.shape,
        width=width,
        tile=float(tile),
        cell_index=cell_index,
        pixels=_cell_pixels(cell_index, width * height),
        line_band=line_band,
    )


@lru_cache(maxsize=4)
def _hex_geometry(width: int, height: int, size: float, out_shape: tuple[int, int]) -> _Geometry:
    """Pointy-top odd-r hexes of centre-to-vertex ``size`` px; cell (0, 0)'s
    centre sits at (tile / 2, size), so the grid's bounding box starts at 0."""
    tile = math.sqrt(3) * size
    ys, xs = np.mgrid[0:out_shape[0], 0:out_shape[1]]
    px = xs + 0.5 - tile / 2
    py = ys + 0.5 - size
    # Pixel -> fractional axial (q, r), cube-rounded to the containing hex.
    qf = (math.sqrt(3) / 3 * px - py / 3) / size
    rf = (2 / 3 * py) / size
    sf = -qf - rf
    q, r, s = np.rint(qf), np.rint(rf), np.rint(sf)
    dq, dr, ds = np.abs(q - qf), np.abs(r - rf), np.abs(s - sf)
    fix_q = (dq > dr) & (dq > ds)
    fix_r = ~fix_q & (dr > ds)
    q = np.where(fix_q, -r - s, q).astype(np.int64)
    r = np.where(fix_r, -q - s, r).astype(np.int64)
    row = r
    col = q + (r - (r & 1)) // 2
    inside = (col >= 0) & (col < width) & (row >= 0) & (row < height)
    n_cells = width * height
    cell_index = np.where(inside, row * width + col, n_cells).astype(np.int32)

    cx = tile * (col + 0.5 + 0.5 * (row & 1))
    cy = size * (1 + 1.5 * row)
    du = (xs + 0.5 - cx) / tile
    dv = (ys + 0.5 - cy) / tile
    # The cell's own half of a grid line: MiniGrid's line width split across
    # the two cells an edge separates.
    line_band = (_hex_apothem_distance(du, dv) >= 0.5 - _LINE / 2) & inside
    cells = np.arange(n_cells)
    rows, cols = cells // width, cells % width
    centres = np.stack([tile * (cols + 0.5 + 0.5 * (rows & 1)), size * (1 + 1.5 * rows)], axis=1)
    return _Geometry(
        shape=cell_index.shape,
        width=width,
        tile=tile,
        cell_index=cell_index,
        pixels=_cell_pixels(cell_index, n_cells),
        line_band=line_band,
        centres=centres,
    )


def _hex_apothem_distance(du: np.ndarray, dv: np.ndarray) -> np.ndarray:
    """Distance from a pointy-top hex's centre measured along its three edge
    normals (0 and +/-60 degrees); 0.5 on the boundary of a unit-width hex."""
    a = np.abs(du)
    b = np.abs(0.5 * du + (math.sqrt(3) / 2) * dv)
    c = np.abs(0.5 * du - (math.sqrt(3) / 2) * dv)
    return np.maximum(a, np.maximum(b, c))


def hex_frame_shape(width: int, height: int, size: float = HEX_SIZE) -> tuple[int, int]:
    """(H, W) of a hex frame: odd rows reach half a cell further east."""
    return (
        int(math.ceil((1.5 * (height - 1) + 2) * size - 1e-9)),
        int(math.ceil((width + 0.5) * math.sqrt(3) * size - 1e-9)),
    )


def frame_shape(spec: TaskSpecification, hex_size: float = HEX_SIZE) -> tuple[int, int]:
    """(H, W) of a task's frames: MiniGrid's 32 px tiles on square."""
    width, height = spec.maze.dimensions
    if spec.maze.tiling == "hex":
        return hex_frame_shape(width, height, hex_size)
    return (height * TILE_PIXELS, width * TILE_PIXELS)


# --------------------------------------------------------------------------- #
# MiniGrid's fill_coords predicates, vectorised over a cell's pixels
# --------------------------------------------------------------------------- #
def _in_rect(u, v, xmin, xmax, ymin, ymax):
    return (u >= xmin) & (u <= xmax) & (v >= ymin) & (v <= ymax)


def _in_circle(u, v, cx, cy, r):
    return (u - cx) * (u - cx) + (v - cy) * (v - cy) <= r * r


def _in_triangle(u, v, a, b, c):
    """minigrid.utils.rendering.point_in_triangle, same dtypes (float32
    vertices and edge dots, float64 queries), so edge pixels agree."""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)
    v0 = c - a
    v1 = b - a
    v2x = u - a[0]
    v2y = v - a[1]
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot11 = np.dot(v1, v1)
    dot02 = v0[0] * v2x + v0[1] * v2y
    dot12 = v1[0] * v2x + v1[1] * v2y
    inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
    s = (dot11 * dot02 - dot01 * dot12) * inv_denom
    t = (dot00 * dot12 - dot01 * dot02) * inv_denom
    return (s >= 0) & (t >= 0) & ((s + t) < 1)


def _rotated(u, v, theta: float):
    """minigrid.utils.rendering.rotate_fn about the tile centre: the query
    point turned by -theta, so the shape tested appears turned by +theta."""
    x = u - 0.5
    y = v - 0.5
    return (
        0.5 + x * math.cos(-theta) - y * math.sin(-theta),
        0.5 + y * math.cos(-theta) + x * math.sin(-theta),
    )


def _in_cell(tiling: str, u, v, scale: float):
    """Inside the cell shrunk about its centre by ``scale`` (1 = the cell).
    On square this is MiniGrid's rect (0.5 -/+ scale / 2)."""
    du, dv = u - 0.5, v - 0.5
    if tiling == "hex":
        return _hex_apothem_distance(du, dv) <= scale / 2
    half = scale / 2
    return (np.abs(du) <= half) & (np.abs(dv) <= half)


def _downsample(canvas: np.ndarray) -> np.ndarray:
    """minigrid.utils.rendering.downsample then Grid.render's uint8 store,
    bit for bit but fast: ``mean(axis=3)`` is an exact integer sum over 3,
    and ``mean(axis=1)`` adds those means in order before dividing by 3."""
    h, w = canvas.shape[:2]
    x = canvas.reshape(h // SUBDIVS, SUBDIVS, w // SUBDIVS, SUBDIVS, 3).astype(np.uint16)
    rows = (x[:, :, :, 0] + x[:, :, :, 1] + x[:, :, :, 2]) / SUBDIVS
    return (((rows[:, 0] + rows[:, 1]) + rows[:, 2]) / SUBDIVS).astype(np.uint8)


def _direction_angle(tiling: str, direction: int) -> float:
    """Screen angle (y down, clockwise) of a direction: 90 or 60 degree steps."""
    return (0.5 * math.pi if tiling == "square" else math.pi / 3) * direction


@lru_cache(maxsize=256)
def _icon_tile(kind: str, colour: str, active: bool, size: int) -> tuple[np.ndarray, np.ndarray]:
    """A MiniGrid object's own render at ``size`` px, and the pixels it drew."""
    if kind == "key":
        obj = GroundKey(MINIGRID_COLORS.get(colour, colour))
    elif kind == "switch":
        obj = Switch(color=colour, initial_state="on" if active else "off")
    elif kind == "portal":
        obj = TeleporterObj(color=colour)
    elif kind == "death":
        obj = KillCell()
    elif kind == "frozen":
        obj = FrozenTile()
    elif kind == "lava":
        obj = Lava()
    elif kind == "block":
        obj = PushableBlock(color=colour)
    else:
        raise ValueError(f"no icon for {kind!r}")
    tile = np.empty((size, size, 3), dtype=np.uint8)
    tile[:] = _ICON_SENTINEL
    obj.render(tile)
    return tile, (tile != _ICON_SENTINEL).any(axis=-1)


class TilingFrameRenderer:
    """Frames for one task. Geometry (cached per maze size) and everything
    that never changes are drawn once; a frame copies that canvas and draws
    doors, gates, switches, keys, arrows and the agent over it."""

    def __init__(self, spec: TaskSpecification, *, hex_size: float = HEX_SIZE):
        self.spec = spec
        self.ctx = TaskPlanningContext(spec)
        self.tiling = self.ctx.tiling
        self.width, self.height = spec.maze.dimensions
        self.n_cells = self.width * self.height
        if self.tiling == "hex":
            out = frame_shape(spec, hex_size)
            self._fine = _hex_geometry(
                self.width, self.height, hex_size * SUBDIVS, (out[0] * SUBDIVS, out[1] * SUBDIVS)
            )
            self._full = _hex_geometry(self.width, self.height, hex_size, out)
        else:
            self._fine = _square_geometry(self.width, self.height, TILE_PIXELS * SUBDIVS)
            self._full = _square_geometry(self.width, self.height, TILE_PIXELS)
        self._icon_size = int(round(self._fine.tile))
        self._crack_size = int(round(self._full.tile))
        self._static_canvas = self._draw_static()

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        return (*self._full.shape, 3)

    def _cell(self, xy) -> int:
        return int(xy[1]) * self.width + int(xy[0])

    def _draw_static(self) -> np.ndarray:
        """The supersampled canvas of everything that never changes. Cells
        whose object fills its whole shape (walls, goal, ice, turntables,
        lava) cover the grid lines, as MiniGrid's renders do; door and gate
        cells start as floor and are drawn per frame."""
        base = np.zeros((self.n_cells + 1, 3), dtype=np.uint8)
        filled = np.zeros(self.n_cells + 1, dtype=bool)
        base[self.n_cells] = GREY  # outside the hex grid's jagged edge
        filled[self.n_cells] = True
        for xy in self.ctx.walls:
            if 0 <= xy[0] < self.width and 0 <= xy[1] < self.height:
                base[self._cell(xy)], filled[self._cell(xy)] = GREY, True
        for cells, colour in (
            ([self.ctx.goal], GOAL),
            (self.ctx.frozen_tiles, ICE),
            (self.ctx.rotating_index, TURNTABLE),
            (self.ctx.hazards, LAVA),
        ):
            for xy in cells:
                base[self._cell(xy)], filled[self._cell(xy)] = colour, True

        fine = self._fine
        canvas = base[fine.cell_index]
        canvas[fine.line_band & ~filled[fine.cell_index]] = GREY
        flat = canvas.reshape(-1, 3)
        mech = self.spec.mechanisms
        for tp in mech.teleporters:
            for end in (tp.position_a, tp.position_b):
                self._draw_icon(flat, self._cell(end.to_tuple()), "portal", tp.color)
        for cell in mech.kill_cells:
            self._draw_icon(flat, self._cell(cell.to_tuple()), "death")
        for cell in mech.frozen_tiles:
            self._draw_icon(flat, self._cell(cell.to_tuple()), "frozen")
        for hazard in mech.hazards:
            self._draw_icon(flat, self._cell(hazard.position.to_tuple()), "lava")
        for block in mech.blocks:
            self._draw_icon(flat, self._cell(block.position.to_tuple()), "block", block.color)
        return canvas

    def render(
        self,
        state: PlannerState,
        *,
        visible_cells: set[tuple[int, int]] | None = None,
        explored_cells: set[tuple[int, int]] | None = None,
    ) -> np.ndarray:
        """RGB frame of ``state``. With ``visible_cells`` (partial
        observability), cells out of view are dimmed as multigrid's renderer
        dims them: explored ones to 40%, the rest to dark grey."""
        fine = self._fine
        canvas = self._static_canvas.copy()
        flat = canvas.reshape(-1, 3)

        for k, (colour, closed, locked) in self._barriers(state).items():
            self._draw_barrier(flat, k, colour, closed, locked)
        for key_id, x, y in state.key_positions:
            self._draw_icon(flat, self._cell((x, y)), "key", self.ctx.keys_by_id[key_id]["color"])
        for switch in self.spec.mechanisms.switches:
            self._draw_icon(
                flat, self._cell(switch.position.to_tuple()), "switch", switch.color,
                switch.id in state.active_switches,
            )
        for xy, i in self.ctx.rotating_index.items():
            pix, u, v = fine.local(self._cell(xy))
            self._fill(flat, pix, ARROW, self._arrow(u, v, state.rotator_dirs[i]))
        agent = self._cell(state.agent_pos)
        pix, u, v = fine.local(agent)
        u, v = _rotated(u, v, _direction_angle(self.tiling, state.agent_dir))
        self._fill(flat, pix, AGENT, _in_triangle(u, v, *self._triangle(_AGENT_TRIANGLE, _HEX_AGENT_TRIANGLE)))

        img = _downsample(canvas)

        self._full_scale_overlays(img, state, agent)
        if visible_cells is not None:
            self._dim(img, visible_cells, explored_cells or set())
        return img

    def _barriers(self, state: PlannerState) -> dict[int, tuple[np.ndarray, bool, bool]]:
        """Door and gate cells: (colour, closed, locked). A door is locked
        until opened; a gate is closed (never locked) unless a switch opens it."""
        out = {}
        for xy, door in self.ctx.doors_by_pos.items():
            colour = COLORS[MINIGRID_COLORS.get(door["color"], "grey")].astype(np.uint8)
            is_open = door["id"] in state.open_doors
            out[self._cell(xy)] = (colour, not is_open, not is_open)
        for xy, gate in self.ctx.gates_by_pos.items():
            colour = COLORS[MINIGRID_COLORS.get(self._gate_colour(gate["id"]), "grey")].astype(np.uint8)
            out[self._cell(xy)] = (colour, gate["id"] not in state.open_gates, False)
        return out

    def _gate_colour(self, gate_id: str) -> str:
        return next(g.color for g in self.spec.mechanisms.gates if g.id == gate_id)

    @staticmethod
    def _fill(flat: np.ndarray, pix: np.ndarray, colour, mask: np.ndarray) -> None:
        flat[pix[mask]] = colour

    def _draw_barrier(self, flat, k: int, colour, closed: bool, locked: bool) -> None:
        """Door.render, with the frame and rings drawn as shrunken cells."""
        tiling = self.tiling
        pix, u, v = self._fine.local(k)
        if closed:
            # Door.render's full-tile frame covers the cell's grid lines.
            flat[pix] = colour
        else:
            # Open: the door stands against the east edge.
            self._fill(flat, pix, colour, _in_rect(u, v, 0.88, 1.00, 0.00, 1.00) & _in_cell(tiling, u, v, 1.0))
            self._fill(flat, pix, FLOOR, _in_rect(u, v, 0.92, 0.96, 0.04, 0.96) & _in_cell(tiling, u, v, 0.92))
            return
        if locked:
            self._fill(flat, pix, (0.45 * colour.astype(float)).astype(np.uint8), _in_cell(tiling, u, v, 0.88))
            self._fill(flat, pix, colour, _in_rect(u, v, 0.52, 0.75, 0.50, 0.56))
            return
        self._fill(flat, pix, FLOOR, _in_cell(tiling, u, v, 0.92))
        self._fill(flat, pix, colour, _in_cell(tiling, u, v, 0.84))
        self._fill(flat, pix, FLOOR, _in_cell(tiling, u, v, 0.76))
        self._fill(flat, pix, colour, _in_circle(u, v, 0.75, 0.50, 0.08))

    def _draw_icon(self, flat, k: int, kind: str, colour: str = "", active: bool = False) -> None:
        size = self._icon_size
        tile, drawn = _icon_tile(kind, colour, active, size)
        pix, u, v = self._fine.local(k)
        j = np.floor(u * size).astype(np.int64)
        i = np.floor(v * size).astype(np.int64)
        on_tile = (i >= 0) & (i < size) & (j >= 0) & (j < size)
        pix, i, j = pix[on_tile], i[on_tile], j[on_tile]
        keep = drawn[i, j]
        flat[pix[keep]] = tile[i[keep], j[keep]]

    def _triangle(self, square, hexed):
        return hexed if self.tiling == "hex" else square

    def _arrow(self, u, v, direction: int):
        u, v = _rotated(u, v, _direction_angle(self.tiling, direction))
        return _in_triangle(u, v, *self._triangle(_ARROW_TRIANGLE, _HEX_ARROW_TRIANGLE))

    def _full_scale_overlays(self, img: np.ndarray, state: PlannerState, agent: int) -> None:
        """CustomMiniGridEnv.get_frame: drawn on the finished frame, without
        supersampling, as MiniGrid draws them."""
        flat = img.reshape(-1, 3)
        pix, u, v = self._full.local(agent)
        if tuple(state.agent_pos) in self.ctx.rotating_index:
            # Redraw the turntable so its arrow shows, then a small agent.
            direction = state.rotator_dirs[self.ctx.rotating_index[tuple(state.agent_pos)]]
            flat[pix] = TURNTABLE
            self._fill(flat, pix, ARROW, self._arrow(u, v, direction))
            ru, rv = _rotated(u, v, _direction_angle(self.tiling, state.agent_dir))
            small = self._triangle(_SMALL_AGENT_TRIANGLE, _HEX_SMALL_AGENT_TRIANGLE)
            self._fill(flat, pix, AGENT, _in_triangle(ru, rv, *small))
        if state.freeze_remaining > 0:
            # Iced over: the agent's cell frosts on step-on and each swallowed
            # action thins it; diagonal cracks appear as it thaws.
            total = max(1, int(self.spec.mechanisms.freeze_steps))
            frac = min(1.0, state.freeze_remaining / total)
            alpha = 0.25 + 0.6 * frac
            iced = (flat[pix].astype(float) * (1.0 - alpha) + ICE.astype(float) * alpha).round().astype(np.uint8)
            t = self._crack_size
            row = np.floor(v * t).astype(np.int64)
            col = np.floor(u * t).astype(np.int64)
            for k in range(round((1.0 - frac) * 4)):
                iced[(col - row - k * (t // 4)) % t == 0] = CRACK
            flat[pix] = iced

    def _dim(self, img: np.ndarray, visible, explored) -> None:
        index = self._full.cell_index
        shown = np.zeros(self.n_cells + 1, dtype=bool)
        seen = np.zeros(self.n_cells + 1, dtype=bool)
        for xy in visible:
            shown[self._cell(xy)] = True
        for xy in explored:
            seen[self._cell(xy)] = True
        dim = ~shown[index] & seen[index]
        dark = ~shown[index] & ~seen[index] & (index < self.n_cells)
        img[dim] = (img[dim] * 0.4).astype(np.uint8)
        img[dark] = 30
