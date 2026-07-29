"""Display-only motion feedback for the human-play demo.

Purely cosmetic: every effect composites on top of (or nudges) the pygame
blit of the MiniGrid RGB frame. The live environment, observations, and
scoring path are never touched -- so benchmark parity is preserved.

Effects are short and restrained (eval-terminal feedback, not game juice):
camera wall-bounce, one-shot key flash, door/gate fade, switch
press, goal pulse before the success overlay.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import pygame
except ImportError:  # pragma: no cover
    pygame = None  # type: ignore


BOUNCE_MS = 100
KEY_FLASH_MS = 140
DOOR_FADE_MS = 150
GATE_FADE_MS = 110
SWITCH_PRESS_MS = 80
GOAL_PULSE_MS = 320

BOUNCE_PX = 2

# MiniGrid agent_dir: 0=E, 1=S, 2=W, 3=N
_DIR_DELTA = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
_CARDINAL_DELTA = {
    "MOVE_EAST": (1, 0),
    "MOVE_SOUTH": (0, 1),
    "MOVE_WEST": (-1, 0),
    "MOVE_NORTH": (0, -1),
}


def _sin_pulse(t: float) -> float:
    """0→1→0 over t in [0, 1]."""
    return math.sin(max(0.0, min(1.0, t)) * math.pi)


@dataclass
class _Clip:
    kind: str
    start_ms: int
    duration_ms: int
    offset: tuple[int, int] = (0, 0)
    cell: Optional[tuple[int, int]] = None
    tile_surf: Optional["pygame.Surface"] = None


@dataclass
class DemoFx:
    """Active animation clips + helpers to trigger / composite them."""

    enabled: bool = True
    _clips: list[_Clip] = field(default_factory=list)
    _success_pulse_pending: bool = False

    def clear(self) -> None:
        self._clips.clear()
        self._success_pulse_pending = False

    def tick(self, now_ms: int) -> None:
        self._clips = [c for c in self._clips if now_ms - c.start_ms < c.duration_ms]
        if self._success_pulse_pending and not any(c.kind == "pulse" for c in self._clips):
            self._success_pulse_pending = False

    def success_overlay_ready(self) -> bool:
        """True once the goal-pulse (if any) has finished -- SUCCESS dialog
        waits on this so the pulse is visible first."""
        return not self._success_pulse_pending

    def trigger(
        self,
        *,
        now_ms: int,
        token: str,
        event_type: Optional[str],
        events_before: int,
        event_log: list,
        prev_state,
        new_state,
        prev_rgb: Optional[np.ndarray],
        task_spec,
        grid_w: int,
        grid_h: int,
        display_size: int,
        episode_success: bool = False,
    ) -> None:
        if not self.enabled or pygame is None:
            return

        # Rapid key-repeat shouldn't stack camera offsets.
        self._clips = [c for c in self._clips if c.kind != "bounce"]

        travel = self._travel_delta(token, prev_state)
        new_events = event_log[events_before:]
        mech = task_spec.mechanisms if task_spec is not None else None

        if episode_success and new_state is not None:
            # Only arm the delayed SUCCESS dialog when we can actually show a
            # goal pulse on the grid; text-only mode skips straight to overlay.
            pos = new_state.agent_position
            self._clips.append(
                _Clip(
                    kind="pulse",
                    start_ms=now_ms,
                    duration_ms=GOAL_PULSE_MS,
                    cell=(int(pos[0]), int(pos[1])),
                )
            )
            if prev_rgb is not None:
                self._success_pulse_pending = True

        for event in new_events:
            if event.icon == "key" and event.prefix.startswith("Picked") and prev_state is not None:
                # Same-cell pickup: key was under the agent.
                cell = (int(prev_state.agent_position[0]), int(prev_state.agent_position[1]))
                tile = self._extract_tile(prev_rgb, cell, grid_w, grid_h, display_size)
                if tile is not None:
                    self._clips.append(
                        _Clip("flash", now_ms, KEY_FLASH_MS, cell=cell, tile_surf=tile)
                    )

            elif event.icon == "door" and mech is not None:
                cell = self._door_cell_for_event(mech, event)
                tile = self._extract_tile(prev_rgb, cell, grid_w, grid_h, display_size)
                if cell and tile is not None:
                    self._clips.append(
                        _Clip("fade", now_ms, DOOR_FADE_MS, cell=cell, tile_surf=tile)
                    )

            elif event.icon == "gate" and mech is not None and prev_state is not None and new_state is not None:
                cell = self._first_changed_cell(
                    mech.gates, prev_state.open_gates, new_state.open_gates
                )
                tile = self._extract_tile(prev_rgb, cell, grid_w, grid_h, display_size)
                if cell and tile is not None:
                    self._clips.append(
                        _Clip("fade", now_ms, GATE_FADE_MS, cell=cell, tile_surf=tile)
                    )

            elif event.icon == "switch" and mech is not None and prev_state is not None and new_state is not None:
                cell = self._first_changed_cell(
                    mech.switches, prev_state.active_switches, new_state.active_switches
                )
                if cell is not None:
                    self._clips.append(_Clip("press", now_ms, SWITCH_PRESS_MS, cell=cell))

        # Wall bump only -- successful moves stay visually still (a per-step
        # camera nudge read as screen shake under key-repeat).
        if event_type == "BLOCKED" and travel != (0, 0):
            dx, dy = travel
            self._clips.append(
                _Clip("bounce", now_ms, BOUNCE_MS, offset=(-dx * BOUNCE_PX, -dy * BOUNCE_PX))
            )

    def apply(
        self,
        scaled: "pygame.Surface",
        *,
        now_ms: int,
        grid_w: int,
        grid_h: int,
    ) -> tuple["pygame.Surface", tuple[int, int]]:
        """Return ``(surface, blit_offset_px)``.

        Camera clips (wall bounce) contribute only an offset; cell clips paint
        onto a copy of ``scaled``. Caller should clip blits to the grid rect
        so a 2px bounce never spills into the rails.
        """
        self.tick(now_ms)
        if not self._clips or pygame is None:
            return scaled, (0, 0)

        offset = (0, 0)
        out = scaled
        dirty = False
        cell_w = scaled.get_width() / max(1, grid_w)
        cell_h = scaled.get_height() / max(1, grid_h)

        for clip in self._clips:
            t = max(0.0, min(1.0, (now_ms - clip.start_ms) / max(1, clip.duration_ms)))

            if clip.kind == "bounce":
                amp = (1.0 - t) ** 2
                offset = (round(clip.offset[0] * amp), round(clip.offset[1] * amp))
                continue

            if clip.cell is None:
                continue
            if not dirty:
                out = scaled.copy()
                dirty = True

            cx, cy = clip.cell
            dest = pygame.Rect(
                int(cx * cell_w),
                int(cy * cell_h),
                max(1, int(cell_w)),
                max(1, int(cell_h)),
            )

            if clip.kind == "flash" and clip.tile_surf is not None:
                overlay = clip.tile_surf.copy()
                # Brief white flash, then fade the pre-pickup key out.
                if t < 0.35:
                    white = pygame.Surface(overlay.get_size(), pygame.SRCALPHA)
                    white.fill((255, 255, 255, int(150 * (t / 0.35))))
                    overlay.blit(white, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
                overlay.set_alpha(int(255 * (1.0 - t)))
                out.blit(pygame.transform.smoothscale(overlay, dest.size), dest.topleft)

            elif clip.kind == "fade" and clip.tile_surf is not None:
                overlay = clip.tile_surf.copy()
                overlay.set_alpha(int(255 * (1.0 - t)))
                out.blit(pygame.transform.smoothscale(overlay, dest.size), dest.topleft)

            elif clip.kind == "press":
                scale = 1.0 - 0.06 * _sin_pulse(t)
                inset = dest.inflate(
                    -int(dest.width * (1.0 - scale)),
                    -int(dest.height * (1.0 - scale)),
                )
                if inset.width > 2 and inset.height > 2:
                    cell = out.subsurface(dest).copy()
                    pygame.draw.rect(out, (18, 19, 26), dest)
                    out.blit(pygame.transform.smoothscale(cell, inset.size), inset.topleft)

            elif clip.kind == "pulse":
                glow = pygame.Surface(dest.size, pygame.SRCALPHA)
                glow.fill((90, 220, 140, int(110 * _sin_pulse(t))))
                out.blit(glow, dest.topleft, special_flags=pygame.BLEND_RGBA_ADD)

        return out, offset

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _travel_delta(token: str, prev_state) -> tuple[int, int]:
        if token in _CARDINAL_DELTA:
            return _CARDINAL_DELTA[token]
        if token == "MOVE_FORWARD" and prev_state is not None:
            return _DIR_DELTA.get(prev_state.agent_direction, (0, 0))
        return (0, 0)

    @staticmethod
    def _door_cell_for_event(mech, event) -> Optional[tuple[int, int]]:
        # Progress text is "red door" etc. -- match requires_key color.
        color = (event.color or "").lower()
        for door in mech.doors:
            if door.requires_key.lower() == color:
                return (door.position.x, door.position.y)
        if len(mech.doors) == 1:
            d = mech.doors[0]
            return (d.position.x, d.position.y)
        return None

    @staticmethod
    def _first_changed_cell(specs, prev_ids, new_ids) -> Optional[tuple[int, int]]:
        changed = set(prev_ids) ^ set(new_ids)
        for spec in specs:
            if spec.id in changed:
                return (spec.position.x, spec.position.y)
        return None

    @staticmethod
    def _extract_tile(
        prev_rgb: Optional[np.ndarray],
        cell: Optional[tuple[int, int]],
        grid_w: int,
        grid_h: int,
        display_size: int,
    ) -> Optional["pygame.Surface"]:
        if prev_rgb is None or cell is None or pygame is None:
            return None
        cx, cy = cell
        if not (0 <= cx < grid_w and 0 <= cy < grid_h):
            return None
        h, w, _ = prev_rgb.shape
        tile_h = h // grid_h
        tile_w = w // grid_w
        tile = prev_rgb[cy * tile_h : (cy + 1) * tile_h, cx * tile_w : (cx + 1) * tile_w]
        if tile.size == 0:
            return None
        surf = pygame.image.frombuffer(
            np.ascontiguousarray(tile).tobytes(), (tile_w, tile_h), "RGB"
        ).convert()
        dw = max(1, display_size // grid_w)
        dh = max(1, display_size // grid_h)
        return pygame.transform.smoothscale(surf, (dw, dh)).convert_alpha()
