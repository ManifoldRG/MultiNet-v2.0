"""Layout constants, palette, and font loading for the human-play demo UI."""

from __future__ import annotations

from minigrid.core.constants import COLORS as _MINIGRID_COLORS


APP_TITLE = "MultiNet v2.0 Benchmark"
APP_SUBTITLE = "Can you solve what frontier models cannot?"

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

GRID_DISPLAY_SIZE = 640
TOP_BAR_H = 76
BOTTOM_HINT_H = 58
LEFT_RAIL_W = 230
RIGHT_RAIL_W = 230
WINDOW_HEIGHT = TOP_BAR_H + GRID_DISPLAY_SIZE + BOTTOM_HINT_H
WINDOW_WIDTH = LEFT_RAIL_W + GRID_DISPLAY_SIZE + RIGHT_RAIL_W

# Display-only wall recolor (never touches scoring/model RGB).
WALL_GRAY_SRC = (100, 100, 100)
WALL_GRAY_DST = (58, 62, 80)
GRID_BORDER_COLOR = (128, 133, 158)
GRID_BORDER_WIDTH = 4

CARD_MARGIN = 18
CARD_OUTER_RADIUS = 20
OUTER_WIDTH = WINDOW_WIDTH + 2 * CARD_MARGIN
OUTER_HEIGHT = WINDOW_HEIGHT + 2 * CARD_MARGIN

RAIL_MARGIN = 22
KEYCAP_SIZE = 26
KEYCAP_RADIUS = 7
PROGRESS_BULLET_RADIUS = 4
BUTTON_RADIUS = 8

CARD_RADIUS = 12
CARD_GAP = 10
CARD_PAD = 12
CHIP_RADIUS = 9
CHIP_PAD_X = 8

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

COLOR_PAGE_BG = (8, 9, 13)
COLOR_BG = (18, 19, 26)
COLOR_PANEL_BG = (18, 19, 26)
COLOR_HEADER_BG = (18, 19, 26)
COLOR_CARD_BORDER = (41, 44, 58)
COLOR_CARD_BG = (32, 34, 44)
COLOR_SHADOW = (0, 0, 0, 90)
COLOR_BUTTON_BG = (26, 28, 38)
COLOR_BUTTON_BORDER = (55, 59, 76)

COLOR_TEXT_TITLE = (250, 251, 253)
COLOR_TEXT_SUBTITLE = (201, 204, 217)
COLOR_TEXT = (226, 229, 238)
COLOR_TEXT_DIM = (152, 155, 170)
COLOR_TEXT_LABEL = (134, 138, 154)
COLOR_TEXT_ERROR = (255, 100, 100)
COLOR_SEPARATOR = (38, 41, 53)
COLOR_OVERLAY_TEXT = (255, 255, 255)

ACCENT = (232, 234, 242)
ACCENT_GREEN = (90, 220, 140)
ACCENT_CYAN = (64, 220, 255)
ACCENT_AMBER = (255, 186, 82)
ACCENT_BLUE = (110, 175, 255)
ACCENT_PURPLE = (185, 150, 255)

STATUS_MOVES_OK = (64, 255, 140)
STATUS_MOVES_WARN = (255, 196, 48)
STATUS_MOVES_CRIT = (255, 72, 86)
STATUS_FACING = (80, 200, 255)
STATUS_REWARD = STATUS_MOVES_OK
STATUS_REWARD_IDLE = (140, 190, 160)
CTRL_MOVE = (255, 148, 48)
CTRL_INTERACT = (255, 96, 200)
CTRL_META = COLOR_TEXT_DIM

MOVE_TOKENS = frozenset({
    "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT",
    "MOVE_NORTH", "MOVE_SOUTH", "MOVE_WEST", "MOVE_EAST",
})
INTERACT_TOKENS = frozenset({"PICKUP", "DROP", "TOGGLE", "INTERACT"})

MECH_COLOR_RGB = {
    name: (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    for name, rgb in _MINIGRID_COLORS.items()
}
MECH_COLOR_RGB["gray"] = MECH_COLOR_RGB["grey"]
# Extended maze palette (switches/keys) beyond MiniGrid's built-in COLORS.
MECH_COLOR_RGB["black"] = (40, 42, 50)
MECH_COLOR_RGB["white"] = (255, 255, 255)

DIRECTION_NAMES = {
    0: "East (right)",
    1: "South (down)",
    2: "West (left)",
    3: "North (up)",
}

KEY_REPEAT_DELAY = 200
KEY_REPEAT_INTERVAL = 100
MAX_DIFFICULTY_TIER = 6
FPS = 30


def load_font(size: int, bold: bool = False):
    import pygame

    path = pygame.font.match_font("Segoe UI", bold=bold)
    if path:
        return pygame.font.Font(path, size)
    return pygame.font.SysFont(None, size, bold=bold)


def mech_color(name: str) -> tuple:
    return MECH_COLOR_RGB[name.lower()]


def recolor_walls(rgb_array):
    """Swap MiniGrid's flat wall gray for the softer slate (display-only).

    Never mutates the env render buffer used for scoring/models.
    """
    import numpy as np

    arr = np.asarray(rgb_array, dtype=np.uint8)
    mask = np.all(np.abs(arr.astype(np.int16) - WALL_GRAY_SRC) <= 2, axis=-1)
    if mask.any():
        arr = arr.copy()
        arr[mask] = WALL_GRAY_DST
    return arr


def control_accent(token: str | None) -> tuple:
    if token in MOVE_TOKENS:
        return CTRL_MOVE
    if token in INTERACT_TOKENS:
        return CTRL_INTERACT
    return CTRL_META


def difficulty_color(tier: int) -> tuple:
    if tier >= 5:
        return STATUS_MOVES_CRIT
    if tier >= 3:
        return STATUS_MOVES_WARN
    return STATUS_MOVES_OK
