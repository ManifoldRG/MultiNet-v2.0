"""Pygame front end for MiniGridPlaySession.

Owns the window, fonts, layout chrome, rail rendering, and input handling.
Shared palette/layout live in ``demo.theme``; glyphs in ``demo.icons``;
start/result cards in ``demo.overlays``. Holds a ``MiniGridPlaySession`` and
reads/drives it through its public-ish ``_`` methods; it has no game logic of
its own -- see ``demo.session`` for task loading, stepping, transcript
recording, and the progress checklist.

Layout: a three-column layout (Task+Progress rail / hero grid /
Status+Controls rail) with a slim full-width title bar and footer, all
floating as one rounded, bordered card on a darker canvas -- modern-app
chrome rather than a borderless terminal window. Rendering is done in two
passes: everything is drawn onto an offscreen ``self.screen`` surface at the
card's own size exactly as before, then ``run()`` masks its corners round and
blits it onto the real window (``self.window``) inset by ``CARD_MARGIN``.
Full debug detail (raw model-facing text, live settings, manifest metadata)
stays one keypress away via the Tab (settings) / M (model view) overlays
instead of living in the main view.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
_repo_root_str = str(_REPO_ROOT)
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)

try:
    import pygame
except ImportError:
    print(
        "Error: pygame is not installed.\n"
        "Install it with: pip install pygame\n"
        "  or: conda install -c conda-forge pygame"
    )
    sys.exit(1)

from demo.session import MiniGridPlaySession, ProgressEvent, SETTINGS_AXES
from demo.sounds import DemoSounds, sfx_for_dispatch
from demo.fx import DemoFx
from demo.compare import R1ResultCatalog, r1_task_id
from demo.r1_tasks import restrict_to_r1_tasks
from demo import icons
from demo import overlays
from demo.theme import (
    ACCENT,
    ACCENT_AMBER,
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_PURPLE,
    APP_SUBTITLE,
    APP_TITLE,
    BOTTOM_HINT_H,
    BUTTON_RADIUS,
    CARD_GAP,
    CARD_MARGIN,
    CARD_OUTER_RADIUS,
    CARD_PAD,
    CARD_RADIUS,
    CHIP_PAD_X,
    CHIP_RADIUS,
    COLOR_BG,
    COLOR_BUTTON_BG,
    COLOR_BUTTON_BORDER,
    COLOR_CARD_BG,
    COLOR_CARD_BORDER,
    COLOR_HEADER_BG,
    COLOR_PAGE_BG,
    COLOR_PANEL_BG,
    COLOR_SEPARATOR,
    COLOR_SHADOW,
    COLOR_TEXT,
    COLOR_TEXT_DIM,
    COLOR_TEXT_ERROR,
    COLOR_TEXT_LABEL,
    COLOR_TEXT_SUBTITLE,
    COLOR_TEXT_TITLE,
    DIRECTION_NAMES,
    FPS,
    GRID_BORDER_COLOR,
    GRID_BORDER_WIDTH,
    GRID_DISPLAY_SIZE,
    KEY_REPEAT_DELAY,
    KEY_REPEAT_INTERVAL,
    KEYCAP_SIZE,
    LEFT_RAIL_W,
    OUTER_HEIGHT,
    OUTER_WIDTH,
    PROGRESS_BULLET_RADIUS,
    RAIL_MARGIN,
    RIGHT_RAIL_W,
    STATUS_FACING,
    STATUS_MOVES_CRIT,
    STATUS_MOVES_OK,
    STATUS_MOVES_WARN,
    TOP_BAR_H,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    control_accent,
    load_font,
    mech_color,
    recolor_walls,
)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

class MiniGridPlayerUI:
    """Pygame-based interactive player: owns the window/fonts/drawing and
    input handling for a ``MiniGridPlaySession``."""

    def __init__(self, session: MiniGridPlaySession):
        self.session = session

        # Click targets for the top-bar nav/restart buttons, recomputed each
        # frame by _render_top_bar so _handle_click can hit-test them.
        self._btn_prev_rect: Optional[pygame.Rect] = None
        self._btn_next_rect: Optional[pygame.Rect] = None
        self._btn_restart_rect: Optional[pygame.Rect] = None

        # UI overlay state
        self.show_start_screen = True
        self.show_settings_overlay = False
        self.show_model_view_overlay = False
        self.settings_editable = False
        self.show_moves_bar = False
        self.model_view_scroll = 0
        self.text_only_scroll = 0

        # Pygame setup. self.window is the real OS window (the outer
        # canvas); self.screen is an offscreen surface at the card's own
        # size that every _draw_*/_render_* method below targets, unchanged
        # from a plain full-window layout -- run() is the only place that
        # knows the content is actually a rounded, inset card (see its
        # composite step and _card_mask).
        pygame.init()
        self.window = pygame.display.set_mode((OUTER_WIDTH, OUTER_HEIGHT))
        self.screen = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT)).convert_alpha()
        self._card_mask = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        pygame.draw.rect(
            self._card_mask, (255, 255, 255, 255), self._card_mask.get_rect(),
            border_radius=CARD_OUTER_RADIUS,
        )
        pygame.key.set_repeat(KEY_REPEAT_DELAY, KEY_REPEAT_INTERVAL)
        self.clock = pygame.time.Clock()

        # Font setup -- a clean modern UI sans-serif (never a monospace
        # terminal font, which is what made earlier passes read as "90s")
        self.font_title = self._load_font(22, bold=True)
        self.font_subtitle = self._load_font(14, bold=True)
        self.font_main = self._load_font(16)
        self.font_main_bold = self._load_font(16, bold=True)
        self.font_small = self._load_font(13)
        self.font_small_bold = self._load_font(13, bold=True)
        # Bigger/bolder than a plain field caption -- these are the four
        # top-level rail headers (TASK/PROGRESS/STATUS/CONTROLS), which need
        # to visibly anchor the hierarchy above the now-bold stat values.
        self.font_label = self._load_font(14, bold=True)
        self.font_overlay = self._load_font(48, bold=True)
        self.font_overlay_sub = self._load_font(20)
        # Splash / start screen: brand-first hero type, larger than the
        # in-play title bar so Multinet reads as the product, not chrome.
        self.font_splash_title = self._load_font(64, bold=True)
        self.font_splash_sub = self._load_font(22)
        self.font_splash_prompt = self._load_font(16, bold=True)
        # Model-view body uses a mono face so prompt text reads like the web overlay.
        try:
            self.font_mono = pygame.font.SysFont("consolas", 15)
        except Exception:
            self.font_mono = self._load_font(15)

        # Quiet interaction SFX + display-only motion feedback. Both are
        # cosmetic -- never touch env state or observations.
        self.sounds = DemoSounds()
        self.fx = DemoFx()
        self.r1_catalog = R1ResultCatalog()
        restrict_to_r1_tasks(self.session, self.r1_catalog)

        self._sync_caption()

    def _load_font(self, size: int, bold: bool = False) -> pygame.font.Font:
        return load_font(size, bold=bold)

    def _sync_caption(self) -> None:
        """Update the OS window title from the session's current task --
        called after anything that can change it (load/switch/reset)."""
        if self.session.task_spec is not None:
            pygame.display.set_caption(f"MiniGrid Player  |  {self.session.task_spec.task_id}")
        else:
            pygame.display.set_caption("MiniGrid Task Player")

    def _reset(self) -> None:
        self.sounds.play("restart")
        self.fx.clear()
        self.session._checkpoint_trajectory()
        self.session._reset_env()
        self.model_view_scroll = 0
        self.text_only_scroll = 0
        self._sync_caption()

    def _switch_task(self, delta: int) -> None:
        self.sounds.play("navigate")
        self.fx.clear()
        self.session._load_adjacent_task(delta)
        self.model_view_scroll = 0
        self.text_only_scroll = 0
        self._sync_caption()

    def _dispatch_with_feedback(self, token: str) -> None:
        """Step the session, then fire matching SFX + display-only FX.

        Snapshots the pre-step RGB frame so cell effects (key flash, door/
        gate fade) can paint the *previous* tile over the new frame briefly
        -- the env itself has already moved on, which is what we want for
        benchmark parity."""
        session = self.session
        events_before = len(session.event_log)
        prev_state = session.state
        prev_rgb = None
        if session.backend.env is not None and session.config.observation != "text_only":
            prev_rgb = session.backend.render()

        session._dispatch_token(token)
        self.sounds.play(sfx_for_dispatch(session, events_before))

        if session.backend.env is None or prev_state is None:
            return
        self.fx.trigger(
            now_ms=pygame.time.get_ticks(),
            session=session,
            token=token,
            events_before=events_before,
            prev_state=prev_state,
            prev_rgb=prev_rgb,
            display_size=GRID_DISPLAY_SIZE,
        )

    # ------------------------------------------------------------------
    # Input: physical key -> action token
    # ------------------------------------------------------------------

    def _key_to_token(self, key: int) -> Optional[str]:
        """Map a physical key to an action token, respecting action_space."""
        cardinal = self.session.config.action_space == "cardinal"
        if key in (pygame.K_UP, pygame.K_w):
            return "MOVE_NORTH" if cardinal else "MOVE_FORWARD"
        if key in (pygame.K_DOWN, pygame.K_s):
            return "MOVE_SOUTH" if cardinal else None
        if key in (pygame.K_LEFT, pygame.K_a):
            return "MOVE_WEST" if cardinal else "TURN_LEFT"
        if key in (pygame.K_RIGHT, pygame.K_d):
            return "MOVE_EAST" if cardinal else "TURN_RIGHT"
        if key == pygame.K_SPACE:
            return "PICKUP"
        if key == pygame.K_x:
            return "DROP"
        if key in (pygame.K_t, pygame.K_e):
            return "INTERACT" if cardinal else "TOGGLE"
        if key == pygame.K_BACKSPACE:
            return "DONE"
        return None

    # ------------------------------------------------------------------
    # Low-level drawing helpers
    # ------------------------------------------------------------------

    def _draw_text(
        self, text: str, x: int, y: int, font: pygame.font.Font, color: tuple,
        surface: Optional[pygame.Surface] = None,
    ) -> int:
        """Draw a single line of text and return the y position below it.

        Draws onto ``surface`` if given, else the main screen -- lets card
        content be measured/composited via ``_render_card`` (see below)."""
        target = surface if surface is not None else self.screen
        surf = font.render(text, True, color)
        target.blit(surf, (x, y))
        return y + surf.get_height() + 2

    def _draw_wrapped_text(
        self, text: str, x: int, y: int,
        font: pygame.font.Font, color: tuple, max_width: int,
        surface: Optional[pygame.Surface] = None,
    ) -> int:
        """Draw word-wrapped text and return the y position below it."""
        for line in self._wrap_lines(text.split("\n"), font, max_width):
            y = self._draw_text(line, x, y, font, color, surface=surface)
        return y

    def _draw_progress_line(
        self, x: int, y: int, event: ProgressEvent, max_width: int, object_color: tuple,
    ) -> int:
        """Render one PROGRESS entry, coloring only ``event.object_phrase``
        (the mechanism mention) and leaving the surrounding verb text in the
        plain text color -- e.g. "Opened the " / "red door" tinted / "".
        Renders as one line if it fits; otherwise falls back to a single
        flat-colored wrapped block rather than trying to color-wrap segments
        (this only happens for unusually long entries)."""
        font = self.font_small_bold
        segments = [(event.prefix, COLOR_TEXT), (event.object_phrase, object_color), (event.suffix, COLOR_TEXT)]
        total_w = sum(font.size(text)[0] for text, _ in segments if text)
        if total_w <= max_width:
            cx = x
            for text, color in segments:
                if not text:
                    continue
                surf = font.render(text, True, color)
                self.screen.blit(surf, (cx, y))
                cx += surf.get_width()
            return y + self._line_height(font)
        combined = f"{event.prefix}{event.object_phrase}{event.suffix}"
        return self._draw_wrapped_text(combined, x, y, font, COLOR_TEXT, max_width)

    def _draw_section_label(
        self, text: str, x: int, y: int, color: tuple,
        surface: Optional[pygame.Surface] = None,
    ) -> int:
        """Small, muted, letter-spaced uppercase card header."""
        spaced = " ".join(text.upper())
        return self._draw_text(spaced, x, y, self.font_label, color, surface=surface)

    def _draw_chip(
        self, text: str, x: int, y: int, font: pygame.font.Font,
        fg_color: tuple, bg_color: tuple, surface: Optional[pygame.Surface] = None,
    ) -> tuple[int, int]:
        """Draw a small pill-shaped badge. Returns (chip_width, chip_height)."""
        target = surface if surface is not None else self.screen
        text_surf = font.render(text, True, fg_color)
        w = text_surf.get_width() + CHIP_PAD_X * 2
        h = text_surf.get_height() + 6
        chip_bg = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(chip_bg, bg_color, chip_bg.get_rect(), border_radius=CHIP_RADIUS)
        target.blit(chip_bg, (x, y))
        target.blit(text_surf, (x + CHIP_PAD_X, y + 3))
        return w, h

    def _draw_arrow_triangle(
        self, surface: pygame.Surface, center: tuple[int, int], size: float,
        direction: str, color: tuple,
    ) -> None:
        icons.draw_arrow_triangle(surface, center, size, direction, color)

    def _draw_bag_icon(self, surface: pygame.Surface, center: tuple[int, int], size: float, color: tuple) -> None:
        icons.draw_bag_icon(surface, center, size, color)

    def _draw_progress_bar(
        self, x: int, y: int, width: int, height: int, fraction: float,
        fg_color: tuple, bg_color: tuple = COLOR_BUTTON_BG,
    ) -> None:
        """Horizontal rounded bar filled to ``fraction`` (0..1) -- used for
        the Moves stat, which drains as the step budget is used up."""
        bg_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, bg_color, bg_rect, border_radius=height // 2)
        fraction = max(0.0, min(1.0, fraction))
        if fraction > 0:
            fill_w = max(height, round(width * fraction))
            pygame.draw.rect(self.screen, fg_color, pygame.Rect(x, y, fill_w, height), border_radius=height // 2)
        pygame.draw.rect(self.screen, COLOR_BUTTON_BORDER, bg_rect, width=1, border_radius=height // 2)

    def _draw_key_icon(self, surface: pygame.Surface, center: tuple[int, int], size: float, color: tuple) -> None:
        icons.draw_key_icon(surface, center, size, color)

    def _draw_door_icon(self, surface: pygame.Surface, center: tuple[int, int], size: float, color: tuple) -> None:
        icons.draw_door_icon(surface, center, size, color)

    def _draw_switch_icon(self, surface: pygame.Surface, center: tuple[int, int], size: float, color: tuple) -> None:
        icons.draw_switch_icon(surface, center, size, color)

    def _draw_gate_icon(self, surface: pygame.Surface, center: tuple[int, int], size: float, color: tuple) -> None:
        icons.draw_gate_icon(surface, center, size, color)

    def _draw_box_icon(self, surface: pygame.Surface, center: tuple[int, int], size: float, color: tuple) -> None:
        icons.draw_box_icon(surface, center, size, color)

    def _progress_icon_fn(self, icon_kind: Optional[str]):
        return icons.progress_icon_fn(icon_kind)

    def _mech_color(self, name: str) -> tuple:
        return mech_color(name)

    def _draw_restart_icon(self, surface: pygame.Surface, center: tuple[int, int], size: float, color: tuple) -> None:
        icons.draw_restart_icon(surface, center, size, color)

    def _draw_keycap(
        self, surface: pygame.Surface, x: int, y: int, w: int, h: int,
        content: str, active: bool = False, accent: tuple = ACCENT,
    ) -> pygame.Rect:
        return icons.draw_keycap(
            surface, x, y, w, h, content, self.font_small_bold, active=active, accent=accent,
        )

    def _draw_button(
        self, x: int, y: int, w: int, h: int, surface: Optional[pygame.Surface] = None,
    ) -> pygame.Rect:
        target = surface if surface is not None else self.screen
        return icons.draw_button(target, x, y, w, h)

    def _render_card(
        self, x: int, y: int, width: int, content_fn, accent_color: Optional[tuple] = None,
        min_height: int = 0,
    ) -> int:
        """Render a rounded, drop-shadowed card at (x, y) with the given width.

        ``content_fn(surface, cx, cy, cw) -> end_y`` draws the card's content
        onto a transparent scratch surface at local coordinates; the content is
        measured there first so callers don't need to precompute text height.
        Returns the screen y position just below the card (including CARD_GAP).
        """
        scratch = pygame.Surface((width, 520), pygame.SRCALPHA)
        end_y = content_fn(scratch, CARD_PAD, CARD_PAD, width - 2 * CARD_PAD)
        card_h = max(min_height, end_y + CARD_PAD)

        shadow = pygame.Surface((width + 6, card_h + 6), pygame.SRCALPHA)
        pygame.draw.rect(shadow, COLOR_SHADOW, shadow.get_rect(), border_radius=CARD_RADIUS + 2)
        self.screen.blit(shadow, (x - 1, y + 3))

        card_rect = pygame.Rect(x, y, width, card_h)
        pygame.draw.rect(self.screen, COLOR_CARD_BG, card_rect, border_radius=CARD_RADIUS)

        if accent_color is not None:
            stripe = pygame.Surface((width, card_h), pygame.SRCALPHA)
            pygame.draw.rect(
                stripe, accent_color, pygame.Rect(0, 0, 4, card_h),
                border_top_left_radius=CARD_RADIUS, border_bottom_left_radius=CARD_RADIUS,
            )
            self.screen.blit(stripe, (x, y))

        self.screen.blit(scratch, (x, y), area=pygame.Rect(0, 0, width, card_h))
        return y + card_h + CARD_GAP

    def _wrap_lines(self, lines: list[str], font: pygame.font.Font, max_width: int) -> list[str]:
        """Word-wrap each input line to ``max_width``, preserving blank lines."""
        out: list[str] = []
        for line in lines:
            if not line:
                out.append("")
                continue
            words = line.split(" ")
            current = ""
            for word in words:
                test = f"{current} {word}".strip()
                if font.size(test)[0] <= max_width:
                    current = test
                else:
                    if current:
                        out.append(current)
                    current = word
            out.append(current)
        return out

    def _line_height(self, font: pygame.font.Font) -> int:
        return font.get_height() + 2

    def _draw_scrollable_text(
        self,
        lines: list[str],
        rect: pygame.Rect,
        font: pygame.font.Font,
        color: tuple,
        scroll: int,
    ) -> int:
        """Draw ``lines`` clipped to ``rect``, offset by ``scroll`` pixels.
        Returns the scroll value clamped to the actual content height."""
        lh = self._line_height(font)
        content_height = lh * len(lines)
        max_scroll = max(0, content_height - rect.height)
        scroll = max(0, min(scroll, max_scroll))

        self.screen.set_clip(rect)
        y = rect.top - scroll
        for line in lines:
            if rect.top - lh <= y <= rect.bottom:
                surf = font.render(line, True, color)
                self.screen.blit(surf, (rect.left, y))
            y += lh
        self.screen.set_clip(None)
        return scroll

    # ------------------------------------------------------------------
    # Rendering: start / splash screen
    # ------------------------------------------------------------------

    def _render_start_screen(self) -> None:
        overlays.render_start_screen(self)

    def _render_top_bar(self) -> None:
        """Full-width strip: app title/subtitle on the left, task navigation
        and restart on the right -- the only chrome spanning the whole window.
        Also records button hitboxes for click handling (see _handle_click)."""
        session = self.session
        rect = pygame.Rect(0, 0, WINDOW_WIDTH, TOP_BAR_H)
        pygame.draw.rect(self.screen, COLOR_HEADER_BG, rect)
        pygame.draw.line(self.screen, COLOR_SEPARATOR, (0, TOP_BAR_H), (WINDOW_WIDTH, TOP_BAR_H))

        title_surf = self.font_title.render(APP_TITLE, True, COLOR_TEXT_TITLE)
        self.screen.blit(title_surf, (RAIL_MARGIN, 13))

        subtitle_surf = self.font_subtitle.render(APP_SUBTITLE, True, COLOR_TEXT_SUBTITLE)
        self.screen.blit(subtitle_surf, (RAIL_MARGIN, 13 + title_surf.get_height() + 1))

        # Right-aligned cluster, built leftward from the window edge:
        # [REC] [Task i/n]  [<] [>]  [Restart]
        cy = TOP_BAR_H // 2
        bx = WINDOW_WIDTH - RAIL_MARGIN

        restart_w, restart_h = 96, 30
        bx -= restart_w
        self._btn_restart_rect = self._draw_button(bx, cy - restart_h // 2, restart_w, restart_h)
        self._draw_restart_icon(self.screen, (bx + 20, cy), 15, ACCENT)
        restart_label = self.font_small_bold.render("Restart", True, COLOR_TEXT)
        self.screen.blit(restart_label, (bx + 33, cy - restart_label.get_height() // 2))
        bx -= 16

        self._btn_prev_rect = None
        self._btn_next_rect = None
        if session.task_list:
            nav_d = 30
            bx -= nav_d
            self._btn_next_rect = self._draw_button(bx, cy - nav_d // 2, nav_d, nav_d)
            next_surf = self.font_main_bold.render(">", True, ACCENT)
            self.screen.blit(next_surf, next_surf.get_rect(center=self._btn_next_rect.center))
            bx -= 6
            bx -= nav_d
            self._btn_prev_rect = self._draw_button(bx, cy - nav_d // 2, nav_d, nav_d)
            prev_surf = self.font_main_bold.render("<", True, ACCENT)
            self.screen.blit(prev_surf, prev_surf.get_rect(center=self._btn_prev_rect.center))
            bx -= 16

            counter_surf = self.font_main_bold.render(f"Task {session.task_index + 1} / {len(session.task_list)}", True, COLOR_TEXT)
            bx -= counter_surf.get_width()
            self.screen.blit(counter_surf, (bx, cy - counter_surf.get_height() // 2))
            bx -= 16

        if session.record:
            rec_surf = self.font_small.render("REC", True, (235, 120, 120))
            bx -= rec_surf.get_width()
            self.screen.blit(rec_surf, (bx, cy - rec_surf.get_height() // 2))
            pygame.draw.circle(self.screen, (225, 70, 70), (bx - 9, cy), 4)

    def _render_grid(self) -> None:
        """Render the MiniGrid environment as the hero panel: a softened,
        enlarged copy of the raw frame framed with a strong border so it
        reads as the centerpiece rather than a flat inset image. Display-only
        FX (nudge / cell flash / fade) are composited here and never touch
        the env's own render buffer."""
        rgb_array = recolor_walls(self.session.backend.render())
        h, w, _c = rgb_array.shape
        surf = pygame.image.frombuffer(rgb_array.tobytes(), (w, h), "RGB")
        scaled = pygame.transform.smoothscale(surf, (GRID_DISPLAY_SIZE, GRID_DISPLAY_SIZE))

        env = self.session.backend.env
        grid_w = env.width if env is not None else 1
        grid_h = env.height if env is not None else 1
        framed, (ox, oy) = self.fx.apply(
            scaled, now_ms=pygame.time.get_ticks(), grid_w=grid_w, grid_h=grid_h,
        )

        grid_rect = pygame.Rect(LEFT_RAIL_W, TOP_BAR_H, GRID_DISPLAY_SIZE, GRID_DISPLAY_SIZE)
        # Fill under the grid so a 2px wall-bounce offset doesn't trail pixels.
        pygame.draw.rect(self.screen, COLOR_BG, grid_rect)
        self.screen.set_clip(grid_rect)
        self.screen.blit(framed, (LEFT_RAIL_W + ox, TOP_BAR_H + oy))
        self.screen.set_clip(None)
        pygame.draw.rect(self.screen, GRID_BORDER_COLOR, grid_rect, width=GRID_BORDER_WIDTH)

    def _render_text_only_pane(self) -> None:
        """In text_only mode the model gets no image, so neither does the
        human: render the exact model-facing text here instead of the grid.
        Occupies the same square footprint the grid image would."""
        rect = pygame.Rect(LEFT_RAIL_W, TOP_BAR_H, GRID_DISPLAY_SIZE, GRID_DISPLAY_SIZE)
        pygame.draw.rect(self.screen, COLOR_BG, rect)
        pygame.draw.rect(self.screen, GRID_BORDER_COLOR, rect, width=GRID_BORDER_WIDTH)

        lines = [
            "TEXT-ONLY MODE",
            "(matches what the model receives -- no image shown)",
            "",
        ]
        for title, text in self.session._build_model_view_sections():
            lines.append(f"-- {title} --")
            lines.extend(text.split("\n"))
            lines.append("")
        lines.append("(scroll with mouse wheel / Page Up / Page Down)")

        inner = rect.inflate(-24, -24)
        wrapped = self._wrap_lines(lines, self.font_small, inner.width)
        self.text_only_scroll = self._draw_scrollable_text(
            wrapped, inner, self.font_small, COLOR_TEXT, self.text_only_scroll
        )

    def _render_bottom_hint(self) -> None:
        """Full-width, centered two-line footer: a friendly instruction
        reminding the human to play fair (same info the model gets), stacked
        above the compact global/meta shortcut strip (task nav and restart
        have on-screen buttons now; move/pickup/toggle are the CONTROLS
        legend)."""
        rect = pygame.Rect(0, WINDOW_HEIGHT - BOTTOM_HINT_H, WINDOW_WIDTH, BOTTOM_HINT_H)
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, rect)
        pygame.draw.line(self.screen, COLOR_SEPARATOR, (0, rect.top), (WINDOW_WIDTH, rect.top))

        message = "Solve the task using only the information visible in the maze."
        msg_surf = self.font_small_bold.render(message, True, COLOR_TEXT)
        self.screen.blit(msg_surf, msg_surf.get_rect(center=(WINDOW_WIDTH // 2, rect.top + 15)))

        # Built from segments and truncated at a whole-segment boundary
        # (rather than a hard character clip) so it degrades gracefully at
        # narrower window widths.
        segments = ["[ / ] switch task", "Tab settings", "M model view", "Q quit"]
        max_width = WINDOW_WIDTH - 28
        text = ""
        for seg in segments:
            candidate = seg if not text else f"{text}      {seg}"
            if self.font_small_bold.size(candidate)[0] > max_width:
                break
            text = candidate
        surf = self.font_small_bold.render(text, True, COLOR_TEXT_DIM)
        self.screen.blit(surf, surf.get_rect(center=(WINDOW_WIDTH // 2, rect.top + 36)))

    def _render_main_pane(self) -> None:
        if self.session.backend.env is None:
            placeholder_surf = self.font_main.render(
                "No environment loaded.", True, COLOR_TEXT_DIM
            )
            self.screen.blit(placeholder_surf, (LEFT_RAIL_W + 20, TOP_BAR_H + GRID_DISPLAY_SIZE // 2))
            return
        if self.session.config.observation == "text_only":
            self._render_text_only_pane()
        else:
            self._render_grid()

    # ------------------------------------------------------------------
    # Rendering: side rails
    # ------------------------------------------------------------------
    # Deliberately card-free (unlike the Tab/M overlays below): just labels,
    # values, and the occasional divider line, to match the flatter, quieter
    # reference layout. Debug-only info that isn't needed to actually play
    # (raw config axes, manifest metadata) lives in the Tab overlay instead.

    def _draw_status_label(self, x: int, y: int, label: str) -> int:
        label_surf = self.font_small_bold.render(" ".join(label.upper()), True, COLOR_TEXT_LABEL)
        self.screen.blit(label_surf, (x, y))
        return y + label_surf.get_height() + 4

    def _draw_status_icon_row(
        self, x: int, y: int, label: str, icon_fn, value: str, value_color: tuple = COLOR_TEXT,
    ) -> int:
        """Label above, small glyph + bold value below -- the shared pattern
        used by the Facing and Reward stats (Moves is a progress bar instead,
        Inventory has its own compact layout)."""
        y = self._draw_status_label(x, y, label)
        icon_fn(self.screen, (x + 8, y + 9), 16, value_color)
        value_surf = self.font_main_bold.render(value, True, value_color)
        self.screen.blit(value_surf, (x + 24, y))
        return y + self._line_height(self.font_main) + 8

    def _render_inventory_row(self, x: int, y: int) -> int:
        """INVENTORY: just what's currently carried, as a colored glyph only
        (a key icon in the key's own color, or the empty satchel) -- no
        color-name text, since the glyph's color already says it. Deliberately
        kept to this single fact -- door/switch/gate state changes are
        already narrated live in the PROGRESS log below as they happen, so
        repeating them here as a second row of icons was the same
        information twice, in two vocabularies."""
        state = self.session.state
        y = self._draw_status_label(x, y, "Inventory")
        carrying = state.agent_carrying
        icon_cy = y + 12
        if carrying:
            self._draw_key_icon(self.screen, (x + 12, icon_cy), 24, self._mech_color(carrying))
        else:
            self._draw_bag_icon(self.screen, (x + 10, icon_cy), 18, COLOR_TEXT_DIM)
        return y + 24 + 8

    def _render_left_rail(self) -> None:
        """Left rail: task description + a live PROGRESS log of milestones
        the player has actually achieved (see session.event_log) -- a
        history of what just happened, not a checklist of what's left
        (which would hint at the solution before it's discovered)."""
        session = self.session
        x = RAIL_MARGIN
        y = TOP_BAR_H + RAIL_MARGIN
        width = LEFT_RAIL_W - 2 * RAIL_MARGIN

        y = self._draw_section_label("Task", x, y, COLOR_TEXT_TITLE)
        y += 10
        y = self._draw_wrapped_text(
            session.task_prompt_text(), x, y, self.font_small_bold, COLOR_TEXT, width
        )

        events = session.event_log
        if events:
            y += 12
            pygame.draw.line(self.screen, COLOR_SEPARATOR, (x, y), (x + width, y))
            y += 18
            y = self._draw_section_label("Progress", x, y, COLOR_TEXT)
            y += 14

            # Room for a small mechanism glyph (key/door/switch/gate/...)
            # to the left of each entry's text.
            indent = 26
            label_x = x + indent
            label_width = width - indent
            line_h = self._line_height(self.font_small_bold)

            def entry_text(event: ProgressEvent) -> str:
                return f"{event.prefix}{event.object_phrase}{event.suffix}"

            def entry_height(event: ProgressEvent) -> int:
                n_lines = max(1, len(self._wrap_lines([entry_text(event)], self.font_small_bold, label_width)))
                return n_lines * line_h + 14

            # Newest-first budget check: only the most recent entries that
            # actually fit above the footer are drawn, so a long-running
            # episode degrades to "show the latest progress" instead of
            # spilling text past the card.
            max_y = WINDOW_HEIGHT - BOTTOM_HINT_H - RAIL_MARGIN
            visible: list[ProgressEvent] = []
            probe_y = y
            for event in reversed(events):
                h = entry_height(event)
                if probe_y + h > max_y and visible:
                    break
                probe_y += h
                visible.append(event)
            visible.reverse()

            if len(visible) < len(events):
                hidden_surf = self.font_small_bold.render(f"({len(events) - len(visible)} earlier not shown)", True, COLOR_TEXT_DIM)
                self.screen.blit(hidden_surf, (x, y))
                y += hidden_surf.get_height() + 8

            for event in visible:
                # Color/icon just the mechanism mention (e.g. "red door"),
                # not the whole sentence, so a glance at the glyph + hue
                # says *what* while the plain-colored verb says *what
                # happened* -- entries with no single mechanism (a generic
                # "Collected x") fall back to the plain text color/a bullet.
                object_color = self._mech_color(event.color) if event.color else COLOR_TEXT
                icon_fn = self._progress_icon_fn(event.icon)
                cy = y + line_h // 2
                icon_cx = x + indent // 2
                if icon_fn:
                    icon_fn(self.screen, (icon_cx, cy), 15, object_color)
                else:
                    # No mechanism-specific glyph for this entry -- a plain
                    # bullet, not a checkbox, since this is a history feed
                    # of things that already happened, not a checklist.
                    pygame.draw.circle(self.screen, object_color, (icon_cx, cy), PROGRESS_BULLET_RADIUS)
                y = self._draw_progress_line(label_x, y, event, label_width, object_color)
                y += 14

    def _control_legend(self) -> list[tuple[str, str, Optional[str]]]:
        """(keycap content, description, action token to highlight when it
        was just dispatched -- None for keys outside the model's action
        vocabulary, e.g. Restart) for the current action_space."""
        if self.session.config.action_space == "cardinal":
            return [
                ("up", "Move North", "MOVE_NORTH"),
                ("down", "Move South", "MOVE_SOUTH"),
                ("left", "Move West", "MOVE_WEST"),
                ("right", "Move East", "MOVE_EAST"),
                ("Space", "Pickup Key", "PICKUP"),
                ("X", "Drop Key", "DROP"),
                ("T", "Toggle Switch/ Open Door", "INTERACT"),
                ("R", "Restart", None),
            ]
        return [
            ("up", "Move Forward", "MOVE_FORWARD"),
            ("left", "Turn Left", "TURN_LEFT"),
            ("right", "Turn Right", "TURN_RIGHT"),
            ("Space", "Pickup Key", "PICKUP"),
            ("X", "Drop Key", "DROP"),
            ("T", "Toggle Switch/ Open Door", "TOGGLE"),
            ("R", "Restart", None),
        ]

    def _render_right_rail(self) -> None:
        """Right rail: live STATUS (moves/facing/inventory, plus any
        active mechanisms) and a CONTROLS legend -- the model's exact action
        vocabulary shown as icon keycaps, with whichever one was just
        dispatched highlighted (replaces the old always-on action-chip bar)."""
        session = self.session
        x = LEFT_RAIL_W + GRID_DISPLAY_SIZE + RAIL_MARGIN
        y = TOP_BAR_H + RAIL_MARGIN
        width = RIGHT_RAIL_W - 2 * RAIL_MARGIN

        y = self._draw_section_label("Status", x, y, COLOR_TEXT)
        y += 10

        state = session.state
        if state:
            if self.show_moves_bar:
                y = self._draw_status_label(x, y, "Moves")
                remaining = max(0, state.max_steps - state.step_count)
                fraction = remaining / state.max_steps if state.max_steps else 0.0
                if fraction < 0.3:
                    moves_color = STATUS_MOVES_CRIT
                elif fraction < 0.5:
                    moves_color = STATUS_MOVES_WARN
                else:
                    moves_color = STATUS_MOVES_OK
                self._draw_progress_bar(x, y, width, 10, fraction, moves_color)
                y += 10 + 8

            direction = state.agent_direction
            dir_name = DIRECTION_NAMES.get(direction, "?").split(" (")[0]
            icon_by_dir = {0: "right", 1: "down", 2: "left", 3: "up"}
            facing_icon = lambda surf, c, s, col: self._draw_arrow_triangle(
                surf, c, s * 0.75, icon_by_dir.get(direction, "up"), col
            )
            y = self._draw_status_icon_row(x, y, "Facing", facing_icon, dir_name, STATUS_FACING)

            y = self._render_inventory_row(x, y)
        else:
            y = self._draw_text("No environment loaded", x, y, self.font_small, COLOR_TEXT_ERROR)

        y += 4
        pygame.draw.line(self.screen, COLOR_SEPARATOR, (x, y), (x + width, y))
        y += 14

        y = self._draw_section_label("Controls", x, y, COLOR_TEXT)
        y += 12
        for content, desc, token in self._control_legend():
            active = token is not None and token == session.last_dispatched_token
            accent = control_accent(token)
            is_arrow = content in ("up", "down", "left", "right")
            keycap_w = KEYCAP_SIZE if is_arrow or len(content) <= 1 else self.font_small_bold.size(content)[0] + 18
            self._draw_keycap(
                self.screen, x, y, keycap_w, KEYCAP_SIZE, content, active=active, accent=accent,
            )
            desc_color = accent if (active or token is not None) else COLOR_TEXT_DIM
            if token is None:
                desc_color = COLOR_TEXT if active else COLOR_TEXT_DIM
            desc_max_w = width - keycap_w - 12
            desc_x = x + keycap_w + 12
            desc_lines = self._wrap_lines([desc], self.font_main_bold, desc_max_w)
            line_h = self.font_main_bold.get_height()
            block_h = line_h * len(desc_lines) + 2 * (len(desc_lines) - 1)
            desc_y = y + max(0, (KEYCAP_SIZE - block_h) // 2)
            for line in desc_lines:
                line_surf = self.font_main_bold.render(line, True, desc_color)
                self.screen.blit(line_surf, (desc_x, desc_y))
                desc_y += line_h + 2
            y += max(KEYCAP_SIZE, block_h) + 7

    # ------------------------------------------------------------------
    # Rendering: overlays
    # ------------------------------------------------------------------

    def _fit_text(
        self, text: str, font: pygame.font.Font, color: tuple, max_width: int,
    ) -> pygame.Surface:
        """Render ``text``, ellipsizing with '…' if it would exceed max_width."""
        surf = font.render(text, True, color)
        if surf.get_width() <= max_width:
            return surf
        ellipsis = "..."
        lo, hi = 0, len(text)
        best = ellipsis
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = text[:mid].rstrip() + ellipsis
            if font.size(candidate)[0] <= max_width:
                best = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return font.render(best, True, color)

    def _render_episode_overlay(self) -> None:
        overlays.render_episode_overlay(self)

    def _render_settings_overlay(self) -> None:
        if not self.show_settings_overlay:
            return
        session = self.session
        scrim = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        scrim.fill((10, 10, 14, 215))
        self.screen.blit(scrim, (0, 0))

        # Centered card matching the web settings overlay.
        card_w = min(560, WINDOW_WIDTH - 80)
        pad = CARD_PAD + 6
        inner_w = card_w - 2 * pad

        # Measure content height on a scratch surface first.
        scratch = pygame.Surface((card_w, 900), pygame.SRCALPHA)
        y = self._draw_settings_card_body(scratch, pad, pad, inner_w)
        card_h = y + pad

        card_x = (WINDOW_WIDTH - card_w) // 2
        card_y = max(40, (WINDOW_HEIGHT - card_h) // 2)

        shadow = pygame.Surface((card_w + 6, card_h + 6), pygame.SRCALPHA)
        pygame.draw.rect(shadow, COLOR_SHADOW, shadow.get_rect(), border_radius=CARD_RADIUS + 2)
        self.screen.blit(shadow, (card_x - 1, card_y + 3))

        card_rect = pygame.Rect(card_x, card_y, card_w, card_h)
        pygame.draw.rect(self.screen, COLOR_CARD_BG, card_rect, border_radius=CARD_RADIUS)
        pygame.draw.rect(self.screen, COLOR_CARD_BORDER, card_rect, 1, border_radius=CARD_RADIUS)
        stripe = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
        pygame.draw.rect(
            stripe,
            ACCENT_BLUE,
            pygame.Rect(0, 0, 5, card_h),
            border_top_left_radius=CARD_RADIUS,
            border_bottom_left_radius=CARD_RADIUS,
        )
        self.screen.blit(stripe, (card_x, card_y))
        self._draw_settings_card_body(self.screen, card_x + pad, card_y + pad, inner_w)

    def _draw_settings_card_body(
        self, surface: pygame.Surface, x: int, y: int, width: int
    ) -> int:
        """Web-matching settings body: title + Esc, help, numbered axes, manifest."""
        session = self.session

        title_surf = self.font_title.render("SETTINGS", True, COLOR_TEXT_TITLE)
        surface.blit(title_surf, (x, y))
        esc_probe = self.font_small_bold.render("Esc", True, COLOR_TEXT)
        esc_w = esc_probe.get_width() + CHIP_PAD_X * 2
        esc_h = esc_probe.get_height() + 6
        esc_x = x + width - esc_w
        self._draw_chip(
            "Esc",
            esc_x,
            y + 2,
            self.font_small_bold,
            COLOR_TEXT,
            (*COLOR_BUTTON_BG, 255),
            surface=surface,
        )
        pygame.draw.rect(
            surface,
            COLOR_BUTTON_BORDER,
            pygame.Rect(esc_x, y + 2, esc_w, esc_h),
            1,
            border_radius=CHIP_RADIUS,
        )
        y += max(title_surf.get_height(), esc_h) + 8

        if self.settings_editable:
            help_text = "Press a number to cycle a value. Tab / Esc to close."
        else:
            help_text = (
                "Frozen for R1 parity (matches interface.config.ExperimentConfig). "
                "Tab / Esc to close."
            )
        y = self._draw_wrapped_text(
            help_text, x, y, self.font_small_bold, COLOR_TEXT, width, surface=surface
        )
        y += 12

        key_bg = (28, 32, 48)
        key_fg = (180, 200, 255)
        key_border = (70, 90, 140)
        for key_char, attr, _choices in SETTINGS_AXES:
            value = getattr(session.config, attr)
            # Number key badge (web settings-key look).
            key_surf = self.font_small_bold.render(str(key_char), True, key_fg)
            badge_w = max(26, key_surf.get_width() + 12)
            badge_h = key_surf.get_height() + 6
            badge = pygame.Rect(x, y, badge_w, badge_h)
            pygame.draw.rect(surface, key_bg, badge, border_radius=5)
            pygame.draw.rect(surface, key_border, badge, 1, border_radius=5)
            surface.blit(
                key_surf,
                (badge.x + (badge_w - key_surf.get_width()) // 2, badge.y + 3),
            )

            tx = x + badge_w + 10
            ty = y + 2
            name_surf = self.font_main.render(f"{attr} = ", True, COLOR_TEXT)
            surface.blit(name_surf, (tx, ty))
            val_surf = self.font_main_bold.render(str(value), True, COLOR_TEXT_TITLE)
            surface.blit(val_surf, (tx + name_surf.get_width(), ty))
            y += badge_h + 8

        manifest_row = (
            session.manifest_row_by_path.get(session.task_path)
            if session.manifest_mode
            else None
        )
        if manifest_row:
            y += 6
            pygame.draw.line(
                surface, COLOR_SEPARATOR, (x, y), (x + width, y), 1
            )
            y += 12
            label = self.font_label.render("MANIFEST ROW", True, ACCENT_PURPLE)
            surface.blit(label, (x, y))
            y += label.get_height() + 6

            bits = [
                f"experiment: {manifest_row.get('experiment', '?')}",
                f"condition: {manifest_row.get('condition', '?')}",
            ]
            if manifest_row.get("variant"):
                bits.append(f"variant: {manifest_row['variant']}")
            y = self._draw_wrapped_text(
                " · ".join(bits),
                x, y, self.font_small_bold, COLOR_TEXT, width, surface=surface,
            )
            mechanisms = manifest_row.get("expected_mechanisms") or []
            if mechanisms:
                y = self._draw_wrapped_text(
                    f"expected mechanisms: {', '.join(mechanisms)}",
                    x, y, self.font_small, ACCENT_AMBER, width, surface=surface,
                )
            if manifest_row.get("notes"):
                y = self._draw_wrapped_text(
                    manifest_row["notes"],
                    x, y, self.font_small_bold, COLOR_TEXT, width, surface=surface,
                )
        return y

    def _render_model_view_overlay(self) -> None:
        if not self.show_model_view_overlay:
            return
        session = self.session
        scrim = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        scrim.fill((10, 10, 14, 225))
        self.screen.blit(scrim, (0, 0))

        # Centered card matching the web model-view overlay.
        card_w = min(640, WINDOW_WIDTH - 80)
        card_h = min(520, WINDOW_HEIGHT - 80)
        card_x = (WINDOW_WIDTH - card_w) // 2
        card_y = max(40, (WINDOW_HEIGHT - card_h) // 2)

        shadow = pygame.Surface((card_w + 6, card_h + 6), pygame.SRCALPHA)
        pygame.draw.rect(shadow, COLOR_SHADOW, shadow.get_rect(), border_radius=CARD_RADIUS + 2)
        self.screen.blit(shadow, (card_x - 1, card_y + 3))

        card_rect = pygame.Rect(card_x, card_y, card_w, card_h)
        pygame.draw.rect(self.screen, COLOR_CARD_BG, card_rect, border_radius=CARD_RADIUS)
        pygame.draw.rect(self.screen, COLOR_CARD_BORDER, card_rect, 1, border_radius=CARD_RADIUS)
        stripe = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
        pygame.draw.rect(
            stripe,
            ACCENT_PURPLE,
            pygame.Rect(0, 0, 5, card_h),
            border_top_left_radius=CARD_RADIUS,
            border_bottom_left_radius=CARD_RADIUS,
        )
        self.screen.blit(stripe, (card_x, card_y))

        pad = CARD_PAD + 6
        x = card_x + pad
        y = card_y + pad
        inner_w = card_w - 2 * pad

        # Header: title + Esc chip (flush right)
        title_surf = self.font_title.render("MODEL VIEW", True, COLOR_TEXT_TITLE)
        self.screen.blit(title_surf, (x, y))
        esc_probe = self.font_small_bold.render("Esc", True, COLOR_TEXT)
        esc_w = esc_probe.get_width() + CHIP_PAD_X * 2
        esc_h = esc_probe.get_height() + 6
        esc_x = x + inner_w - esc_w
        self._draw_chip(
            "Esc",
            esc_x,
            y + 2,
            self.font_small_bold,
            COLOR_TEXT,
            (*COLOR_BUTTON_BG, 255),
        )
        pygame.draw.rect(
            self.screen,
            COLOR_BUTTON_BORDER,
            pygame.Rect(esc_x, y + 2, esc_w, esc_h),
            1,
            border_radius=CHIP_RADIUS,
        )
        y += max(title_surf.get_height(), esc_h) + 8

        y = self._draw_wrapped_text(
            f"observation={session.config.observation} · "
            f"context_window={session.config.context_window}",
            x, y, self.font_small_bold, COLOR_TEXT, inner_w,
        )
        y += 8
        pygame.draw.line(self.screen, COLOR_SEPARATOR, (x, y), (x + inner_w, y), 1)
        y += 12

        sections = session._build_model_view_sections()
        if not sections:
            sections = [("Empty", "No model-view sections yet.")]

        body = pygame.Rect(x, y, inner_w, card_y + card_h - pad - y)
        self.model_view_scroll = self._draw_model_view_sections(
            sections, body, self.model_view_scroll
        )

    def _draw_model_view_sections(
        self,
        sections: list[tuple[str, str]],
        rect: pygame.Rect,
        scroll: int,
    ) -> int:
        """Scrollable section labels + inset monospace boxes (web model-view look)."""
        mono = self.font_mono
        box_pad_x = 12
        box_pad_y = 10
        box_radius = 8
        section_gap = 14
        inset_bg = (12, 14, 20)
        inset_border = (55, 60, 78)
        text_width = max(40, rect.width - 2 * box_pad_x)

        blocks: list[dict] = []
        for title, text in sections:
            lines = self._wrap_lines((text or "").split("\n"), mono, text_width)
            if not lines:
                lines = [""]
            lh = self._line_height(mono)
            label_h = self.font_label.get_height() + 6
            box_h = 2 * box_pad_y + lh * len(lines)
            blocks.append(
                {
                    "title": title,
                    "lines": lines,
                    "label_h": label_h,
                    "box_h": box_h,
                    "height": label_h + box_h + section_gap,
                }
            )

        total_h = sum(b["height"] for b in blocks)
        max_scroll = max(0, total_h - rect.height)
        scroll = max(0, min(scroll, max_scroll))

        self.screen.set_clip(rect)
        y = rect.top - scroll
        for block in blocks:
            bottom = y + block["height"]
            if bottom >= rect.top and y <= rect.bottom:
                # Section header (uppercase purple, matches web overlay).
                label_surf = self.font_label.render(
                    block["title"].upper(), True, ACCENT_PURPLE
                )
                self.screen.blit(label_surf, (rect.left, y))

                box = pygame.Rect(
                    rect.left,
                    y + block["label_h"],
                    rect.width,
                    block["box_h"],
                )
                pygame.draw.rect(self.screen, inset_bg, box, border_radius=box_radius)
                pygame.draw.rect(
                    self.screen, inset_border, box, 1, border_radius=box_radius
                )

                ty = box.top + box_pad_y
                for line in block["lines"]:
                    line_surf = mono.render(line, True, COLOR_TEXT_TITLE)
                    self.screen.blit(line_surf, (box.left + box_pad_x, ty))
                    ty += self._line_height(mono)
            y += block["height"]
        self.screen.set_clip(None)
        return scroll

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the main event loop."""
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break

                if event.type == pygame.KEYDOWN:
                    result = self._handle_keydown(event)
                    if result == "quit":
                        running = False
                        break
                    elif result == "reset":
                        self._reset()

                elif event.type == pygame.MOUSEWHEEL:
                    self._handle_scroll(event)

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    result = self._handle_click(event.pos)
                    if result == "reset":
                        self._reset()

            self.screen.fill(COLOR_BG)
            if self.show_start_screen:
                self._render_start_screen()
            else:
                self.fx.tick(pygame.time.get_ticks())
                self._render_top_bar()
                self._render_main_pane()
                self._render_left_rail()
                self._render_right_rail()
                self._render_bottom_hint()
                self._render_episode_overlay()
                self._render_settings_overlay()
                self._render_model_view_overlay()

            # Composite: round the card's corners (zero their alpha via the
            # precomputed mask), then lay it on the darker page canvas with
            # its outer border -- this is what makes it read as a floating
            # card rather than a borderless full-bleed window.
            self.screen.blit(self._card_mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
            self.window.fill(COLOR_PAGE_BG)
            self.window.blit(self.screen, (CARD_MARGIN, CARD_MARGIN))
            pygame.draw.rect(
                self.window, COLOR_CARD_BORDER,
                pygame.Rect(CARD_MARGIN, CARD_MARGIN, WINDOW_WIDTH, WINDOW_HEIGHT),
                width=1, border_radius=CARD_OUTER_RADIUS,
            )

            pygame.display.flip()
            self.clock.tick(FPS)

        self.session._checkpoint_trajectory()

        self.session.close()
        self.sounds.close()
        pygame.quit()

    def _handle_click(self, pos: tuple[int, int]) -> Optional[str]:
        """Hit-test the top-bar nav/restart buttons rendered by
        _render_top_bar. Purely a visual affordance layered on top of the
        existing keyboard shortcuts ([ / ] and R) -- not a new input path.

        ``pos`` comes from a pygame mouse event, i.e. real *window*
        coordinates -- shift back to the card-local coordinates the button
        rects were recorded in (see run()'s CARD_MARGIN composite step)."""
        if self.show_start_screen:
            return None
        pos = (pos[0] - CARD_MARGIN, pos[1] - CARD_MARGIN)
        if self._btn_prev_rect and self._btn_prev_rect.collidepoint(pos):
            self._switch_task(-1)
        elif self._btn_next_rect and self._btn_next_rect.collidepoint(pos):
            self._switch_task(1)
        elif self._btn_restart_rect and self._btn_restart_rect.collidepoint(pos):
            return "reset"
        return None

    def _handle_scroll(self, event: pygame.event.Event) -> None:
        if self.show_start_screen:
            return
        delta = -event.y * 3 * self._line_height(self.font_small)
        if self.show_model_view_overlay:
            self.model_view_scroll = max(0, self.model_view_scroll + delta)
        elif self.session.config.observation == "text_only":
            self.text_only_scroll = max(0, self.text_only_scroll + delta)

    def _handle_keydown(self, event: pygame.event.Event) -> Optional[str]:
        """
        Handle a pygame KEYDOWN event. Dispatches actions/overlay toggles
        directly (side effects), returning only 'quit' / 'reset' / None for
        the main loop to act on.
        """
        session = self.session
        key = event.key

        if self.show_start_screen:
            if key == pygame.K_q:
                return "quit"
            if key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                self.show_start_screen = False
                self.sounds.play("navigate")
            return None

        if key == pygame.K_ESCAPE:
            if self.show_settings_overlay or self.show_model_view_overlay:
                self.show_settings_overlay = False
                self.show_model_view_overlay = False
            return None

        if key == pygame.K_q:
            return "quit"

        if key == pygame.K_TAB:
            self.show_settings_overlay = not self.show_settings_overlay
            self.show_model_view_overlay = False
            return None

        if key == pygame.K_m:
            self.show_model_view_overlay = not self.show_model_view_overlay
            self.show_settings_overlay = False
            return None

        if self.show_settings_overlay and self.settings_editable:
            for key_char, _attr, _choices in SETTINGS_AXES:
                if key == getattr(pygame, f"K_{key_char}", None):
                    session._cycle_setting(key_char)
                    break
            return None

        if self.show_model_view_overlay:
            if key == pygame.K_PAGEDOWN:
                self.model_view_scroll += 200
            elif key == pygame.K_PAGEUP:
                self.model_view_scroll = max(0, self.model_view_scroll - 200)
            return None

        if key == pygame.K_r:
            return "reset"

        if key == pygame.K_LEFTBRACKET:
            self._switch_task(-1)
            return None
        if key == pygame.K_RIGHTBRACKET:
            self._switch_task(1)
            return None

        if session.episode_done:
            return None

        token = self._key_to_token(key)
        if token is not None:
            self._dispatch_with_feedback(token)
        return None
