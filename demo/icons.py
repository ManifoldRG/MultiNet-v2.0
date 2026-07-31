"""Glyph and keycap drawing for the human-play demo UI."""

from __future__ import annotations

from typing import Callable, Optional

import pygame

from demo.theme import (
    ACCENT,
    BUTTON_RADIUS,
    COLOR_BUTTON_BG,
    COLOR_BUTTON_BORDER,
    COLOR_TEXT_DIM,
    CTRL_META,
    KEYCAP_RADIUS,
)


def draw_arrow_triangle(
    surface: pygame.Surface,
    center: tuple[int, int],
    size: float,
    direction: str,
    color: tuple,
) -> None:
    cx, cy = center
    half = size / 2
    if direction == "up":
        pts = [(cx, cy - half), (cx - half, cy + half * 0.7), (cx + half, cy + half * 0.7)]
    elif direction == "down":
        pts = [(cx, cy + half), (cx - half, cy - half * 0.7), (cx + half, cy - half * 0.7)]
    elif direction == "left":
        pts = [(cx - half, cy), (cx + half * 0.7, cy - half), (cx + half * 0.7, cy + half)]
    else:
        pts = [(cx + half, cy), (cx - half * 0.7, cy - half), (cx - half * 0.7, cy + half)]
    pygame.draw.polygon(surface, color, pts)


def draw_bag_icon(
    surface: pygame.Surface, center: tuple[int, int], size: float, color: tuple,
) -> None:
    cx, cy = center
    body = pygame.Rect(0, 0, size, size * 0.8)
    body.center = (cx, cy + size * 0.1)
    pygame.draw.rect(surface, color, body, width=2, border_radius=3)
    handle = pygame.Rect(0, 0, size * 0.55, size * 0.7)
    handle.center = (cx, cy - size * 0.15)
    pygame.draw.arc(surface, color, handle, 0, 3.14159, 2)


def draw_key_icon(
    surface: pygame.Surface, center: tuple[int, int], size: float, color: tuple,
) -> None:
    cx, cy = center
    r = size * 0.22
    head = (cx - size * 0.24, cy)
    pygame.draw.circle(surface, color, head, r, width=2)
    pygame.draw.line(surface, color, (head[0] + r, cy), (cx + size * 0.4, cy), 2)
    pygame.draw.line(surface, color, (cx + size * 0.2, cy), (cx + size * 0.2, cy + size * 0.18), 2)
    pygame.draw.line(surface, color, (cx + size * 0.36, cy), (cx + size * 0.36, cy + size * 0.18), 2)


def draw_trophy_icon(
    surface: pygame.Surface, center: tuple[int, int], size: float, color: tuple,
) -> None:
    cx, cy = center
    cup = pygame.Rect(0, 0, size * 0.52, size * 0.4)
    cup.center = (cx, cy - size * 0.14)
    pygame.draw.rect(surface, color, cup, width=2, border_radius=int(size * 0.12) or 1)
    left_handle = pygame.Rect(0, 0, size * 0.26, size * 0.3)
    left_handle.center = (cx - size * 0.34, cy - size * 0.1)
    pygame.draw.arc(surface, color, left_handle, 1.2, 4.9, 2)
    right_handle = pygame.Rect(0, 0, size * 0.26, size * 0.3)
    right_handle.center = (cx + size * 0.34, cy - size * 0.1)
    pygame.draw.arc(surface, color, right_handle, -1.9, 1.9, 2)
    pygame.draw.line(surface, color, (cx, cy + size * 0.06), (cx, cy + size * 0.24), 2)
    base = pygame.Rect(0, 0, size * 0.36, size * 0.08)
    base.center = (cx, cy + size * 0.3)
    pygame.draw.rect(surface, color, base, border_radius=1)


def draw_door_icon(
    surface: pygame.Surface, center: tuple[int, int], size: float, color: tuple,
) -> None:
    cx, cy = center
    body = pygame.Rect(0, 0, size * 0.56, size * 0.9)
    body.center = (cx, cy)
    pygame.draw.rect(surface, color, body, width=2, border_radius=2)
    pygame.draw.circle(surface, color, (cx + size * 0.14, cy + size * 0.05), max(1.0, size * 0.06))


def draw_switch_icon(
    surface: pygame.Surface, center: tuple[int, int], size: float, color: tuple,
) -> None:
    cx, cy = center
    pygame.draw.circle(surface, color, (cx, cy), size * 0.34, width=2)
    pygame.draw.circle(surface, color, (cx, cy), size * 0.13)


def draw_gate_icon(
    surface: pygame.Surface, center: tuple[int, int], size: float, color: tuple,
) -> None:
    cx, cy = center
    left_x = cx - size * 0.36
    right_x = cx + size * 0.36
    pygame.draw.line(surface, color, (left_x, cy - size * 0.4), (left_x, cy + size * 0.4), 2)
    pygame.draw.line(surface, color, (right_x, cy - size * 0.4), (right_x, cy + size * 0.4), 2)
    pygame.draw.line(surface, color, (left_x, cy - size * 0.16), (right_x, cy - size * 0.16), 2)
    pygame.draw.line(surface, color, (left_x, cy + size * 0.16), (right_x, cy + size * 0.16), 2)


def draw_box_icon(
    surface: pygame.Surface, center: tuple[int, int], size: float, color: tuple,
) -> None:
    cx, cy = center
    rect = pygame.Rect(0, 0, size * 0.74, size * 0.74)
    rect.center = (cx, cy)
    pygame.draw.rect(surface, color, rect, width=2, border_radius=2)
    pygame.draw.line(surface, color, rect.topleft, rect.bottomright, 1)
    pygame.draw.line(surface, color, rect.topright, rect.bottomleft, 1)


def draw_restart_icon(
    surface: pygame.Surface, center: tuple[int, int], size: float, color: tuple,
) -> None:
    cx, cy = center
    radius = size / 2
    rect = pygame.Rect(0, 0, size, size)
    rect.center = (cx, cy)
    pygame.draw.arc(surface, color, rect, 0.5, 5.6, 2)
    tip_x, tip_y = cx + radius * 0.87, cy - radius * 0.5
    pygame.draw.polygon(
        surface, color,
        [(tip_x - 5, tip_y - 1), (tip_x + 2, tip_y - 5), (tip_x + 3, tip_y + 3)],
    )


def progress_icon_fn(icon_kind: Optional[str]) -> Optional[Callable]:
    """Map a ProgressEvent.icon kind to its glyph drawer."""
    return {
        "key": draw_key_icon,
        "door": draw_door_icon,
        "switch": draw_switch_icon,
        "gate": draw_gate_icon,
        "block": draw_box_icon,
        "goal": draw_trophy_icon,
    }.get(icon_kind)


def draw_keycap(
    surface: pygame.Surface,
    x: int,
    y: int,
    w: int,
    h: int,
    content: str,
    font: pygame.font.Font,
    active: bool = False,
    accent: tuple = ACCENT,
) -> pygame.Rect:
    """Rounded keycap; arrow directions draw as triangles, else short text."""
    rect = pygame.Rect(x, y, w, h)
    is_meta = accent == CTRL_META
    if is_meta:
        border = ACCENT if active else COLOR_BUTTON_BORDER
        bg = (*ACCENT, 30) if active else COLOR_BUTTON_BG
        content_color = ACCENT if active else COLOR_TEXT_DIM
    else:
        border = accent if active else tuple(max(40, int(c * 0.7)) for c in accent)
        fill_alpha = 90 if active else 52
        bg = (*accent, fill_alpha)
        content_color = accent
    bg_surf = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(bg_surf, bg, bg_surf.get_rect(), border_radius=KEYCAP_RADIUS)
    surface.blit(bg_surf, (x, y))
    pygame.draw.rect(surface, border, rect, width=1, border_radius=KEYCAP_RADIUS)
    if content in ("up", "down", "left", "right"):
        draw_arrow_triangle(surface, rect.center, h * 0.4, content, content_color)
    else:
        text_surf = font.render(content, True, content_color)
        surface.blit(text_surf, text_surf.get_rect(center=rect.center))
    return rect


def draw_button(
    surface: pygame.Surface, x: int, y: int, w: int, h: int,
) -> pygame.Rect:
    rect = pygame.Rect(x, y, w, h)
    pygame.draw.rect(surface, COLOR_BUTTON_BG, rect, border_radius=BUTTON_RADIUS)
    pygame.draw.rect(surface, COLOR_BUTTON_BORDER, rect, width=1, border_radius=BUTTON_RADIUS)
    return rect
