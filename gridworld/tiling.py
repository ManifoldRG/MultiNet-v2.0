"""Cell adjacency for the tilings a task spec can declare (``maze.tiling``).

Cells are integer ``(x, y)`` on both tilings. Square is the MiniGrid grid. Hex
is pointy-top in the odd-r offset layout of ``multigrid.tilings.HexTiling``:
``x`` is the column, ``y`` the row, and odd rows sit half a cell east.

Directions count clockwise on screen from east, so direction 0 is east on
both tilings and TURN_RIGHT adds one. Nothing here imports MiniGrid.
"""

from __future__ import annotations

TILINGS: tuple[str, ...] = ("square", "hex")

_SQUARE_STEPS: tuple[tuple[int, int], ...] = (
    (1, 0),   # east
    (0, 1),   # south
    (-1, 0),  # west
    (0, -1),  # north
)

# Offset steps depend on the row's parity: an even row's diagonal neighbours
# lean west, an odd row's lean east.
_HEX_STEPS_EVEN_ROW: tuple[tuple[int, int], ...] = (
    (1, 0),    # east
    (0, 1),    # southeast
    (-1, 1),   # southwest
    (-1, 0),   # west
    (-1, -1),  # northwest
    (0, -1),   # northeast
)
_HEX_STEPS_ODD_ROW: tuple[tuple[int, int], ...] = (
    (1, 0),    # east
    (1, 1),    # southeast
    (0, 1),    # southwest
    (-1, 0),   # west
    (0, -1),   # northwest
    (1, -1),   # northeast
)

DIRECTION_NAMES: dict[str, tuple[str, ...]] = {
    "square": ("EAST", "SOUTH", "WEST", "NORTH"),
    "hex": ("EAST", "SOUTHEAST", "SOUTHWEST", "WEST", "NORTHWEST", "NORTHEAST"),
}


def check_tiling(tiling: str) -> str:
    if tiling not in TILINGS:
        raise ValueError(f"unknown tiling {tiling!r}; choose from {list(TILINGS)}")
    return tiling


def direction_count(tiling: str) -> int:
    """How many directions an agent (or a rotating tile's arrow) can face."""
    return len(DIRECTION_NAMES[check_tiling(tiling)])


def direction_name(tiling: str, direction: int) -> str:
    names = DIRECTION_NAMES[check_tiling(tiling)]
    return names[int(direction) % len(names)]


def neighbor(tiling: str, pos: tuple[int, int], direction: int) -> tuple[int, int]:
    """The cell one step from ``pos`` in ``direction`` (it may lie off the grid)."""
    x, y = pos
    if tiling == "square":
        dx, dy = _SQUARE_STEPS[direction]
    elif tiling == "hex":
        dx, dy = (_HEX_STEPS_ODD_ROW if y & 1 else _HEX_STEPS_EVEN_ROW)[direction]
    else:
        check_tiling(tiling)
    return x + dx, y + dy


def direction_between(tiling: str, a: tuple[int, int], b: tuple[int, int]) -> int | None:
    """The direction that steps from ``a`` to the adjacent cell ``b``, else None."""
    for direction in range(direction_count(tiling)):
        if neighbor(tiling, a, direction) == tuple(b):
            return direction
    return None
