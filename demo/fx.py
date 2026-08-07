"""Display-only motion feedback for the human-play demo.

Purely cosmetic: every effect composites on top of (or nudges) the pygame
blit of the MiniGrid RGB frame. The live environment, observations, and
scoring path are never touched -- so benchmark parity is preserved.

Effects are short and restrained (eval-terminal feedback, not game juice):
camera wall-bounce, one-shot key flash, door/gate fade, switch
press, goal pulse before the success overlay.

``plan_effects`` is the shared decision path for desktop ``DemoFx`` and the
web API serializer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from demo.theme import recolor_walls

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


def travel_delta(token: str, prev_state) -> tuple[int, int]:
    if token in _CARDINAL_DELTA:
        return _CARDINAL_DELTA[token]
    if token == "MOVE_FORWARD" and prev_state is not None:
        return _DIR_DELTA.get(prev_state.agent_direction, (0, 0))
    return (0, 0)


def _door_cell_for_event(mech, event) -> Optional[tuple[int, int]]:
    color = (event.color or "").lower()
    for door in mech.doors:
        if door.requires_key.lower() == color:
            return (door.position.x, door.position.y)
    if len(mech.doors) == 1:
        d = mech.doors[0]
        return (d.position.x, d.position.y)
    return None


def _first_changed_cell(specs, prev_ids, new_ids) -> Optional[tuple[int, int]]:
    changed = set(prev_ids) ^ set(new_ids)
    for spec in specs:
        if spec.id in changed:
            return (spec.position.x, spec.position.y)
    return None


def plan_effects(session, token: str, prev_state, events_before: int) -> list[dict]:
    """Decide display-only effects for one dispatch (no RGB / tile payloads).

    Each item has ``kind`` plus geometry/timing fields. Callers attach platform
    tiles (web PNG / pygame Surface) as needed.
    """
    plan: list[dict] = []

    last = next(
        (rec for rec in reversed(session.transcript) if rec.get("kind") == "step"),
        None,
    )
    event_type = last.get("event_type") if last else None

    travel = travel_delta(token, prev_state)
    if event_type == "BLOCKED" and travel != (0, 0):
        dx, dy = travel
        plan.append(
            {
                "kind": "bounce",
                "dx": int(-dx * BOUNCE_PX),
                "dy": int(-dy * BOUNCE_PX),
                "durationMs": BOUNCE_MS,
            }
        )

    mech = session.task_spec.mechanisms if session.task_spec is not None else None
    new_state = session.state

    for event in session.event_log[events_before:]:
        if event.icon == "key" and event.prefix.startswith("Picked") and prev_state is not None:
            plan.append(
                {
                    "kind": "flash",
                    "cell": [
                        int(prev_state.agent_position[0]),
                        int(prev_state.agent_position[1]),
                    ],
                    "durationMs": KEY_FLASH_MS,
                }
            )
        elif event.icon == "door" and mech is not None:
            cell = _door_cell_for_event(mech, event)
            if cell is not None:
                plan.append(
                    {
                        "kind": "fade",
                        "cell": [int(cell[0]), int(cell[1])],
                        "durationMs": DOOR_FADE_MS,
                    }
                )
        elif (
            event.icon == "gate"
            and mech is not None
            and prev_state is not None
            and new_state is not None
        ):
            cell = _first_changed_cell(mech.gates, prev_state.open_gates, new_state.open_gates)
            if cell is not None:
                plan.append(
                    {
                        "kind": "fade",
                        "cell": [int(cell[0]), int(cell[1])],
                        "durationMs": GATE_FADE_MS,
                    }
                )
        elif (
            event.icon == "switch"
            and mech is not None
            and prev_state is not None
            and new_state is not None
        ):
            cell = _first_changed_cell(
                mech.switches, prev_state.active_switches, new_state.active_switches
            )
            if cell is not None:
                plan.append(
                    {
                        "kind": "press",
                        "cell": [int(cell[0]), int(cell[1])],
                        "durationMs": SWITCH_PRESS_MS,
                    }
                )

    if session.episode_done and session.episode_success and new_state is not None:
        pos = new_state.agent_position
        plan.append(
            {
                "kind": "pulse",
                "cell": [int(pos[0]), int(pos[1])],
                "durationMs": GOAL_PULSE_MS,
            }
        )

    return plan


def _grid_size(session) -> tuple[int, int]:
    env = session.backend.env
    return int(env.width), int(env.height)


def _tile_image_b64(
    rgb: Optional[np.ndarray],
    cell: tuple[int, int],
    grid_w: int,
    grid_h: int,
) -> Optional[str]:
    """PNG data-URL of one grid cell (caller should wall-recolor ``rgb``)."""
    if rgb is None:
        return None
    cx, cy = cell
    if not (0 <= cx < grid_w and 0 <= cy < grid_h):
        return None
    arr = np.asarray(rgb, dtype=np.uint8)
    h, w, _ = arr.shape
    tile_h = h // grid_h
    tile_w = w // grid_w
    if tile_h < 1 or tile_w < 1:
        return None
    tile = np.ascontiguousarray(
        arr[cy * tile_h : (cy + 1) * tile_h, cx * tile_w : (cx + 1) * tile_w, :3]
    )
    if tile.size == 0:
        return None
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(tile, mode="RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _cell_effect(
    kind: str,
    cell: list[int],
    grid_w: int,
    grid_h: int,
    duration_ms: int,
    *,
    tile_image: Optional[str] = None,
) -> dict:
    out = {
        "kind": kind,
        "cell": cell,
        "gridW": grid_w,
        "gridH": grid_h,
        "durationMs": duration_ms,
    }
    if tile_image:
        out["tileImage"] = tile_image
    return out


def effects_for_dispatch(
    session,
    token: str,
    prev_state,
    events_before: int,
    *,
    prev_rgb: Optional[np.ndarray] = None,
) -> list[dict]:
    """Serialize ``plan_effects`` for the web player (adds tile PNGs)."""
    plan = plan_effects(session, token, prev_state, events_before)
    grid_w, grid_h = _grid_size(session)
    prev_frame = recolor_walls(prev_rgb) if prev_rgb is not None else None
    post_frame = None
    effects: list[dict] = []

    for item in plan:
        kind = item["kind"]
        if kind == "bounce":
            effects.append(item)
            continue

        cell = item["cell"]
        duration = item["durationMs"]

        if kind == "pulse":
            effects.append(
                {
                    "kind": "pulse",
                    "cell": cell,
                    "gridW": grid_w,
                    "gridH": grid_h,
                    "durationMs": duration,
                }
            )
            continue

        if kind == "press":
            if post_frame is None:
                post_frame = recolor_walls(
                    np.asarray(session.backend.render(), dtype=np.uint8)
                )
            tile = _tile_image_b64(post_frame, tuple(cell), grid_w, grid_h)
            effects.append(
                _cell_effect("press", cell, grid_w, grid_h, duration, tile_image=tile)
            )
            continue

        # flash / fade use the pre-step tile
        tile = _tile_image_b64(prev_frame, tuple(cell), grid_w, grid_h)
        if tile is not None:
            effects.append(
                _cell_effect(kind, cell, grid_w, grid_h, duration, tile_image=tile)
            )

    return effects


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
        session,
        token: str,
        events_before: int,
        prev_state,
        prev_rgb: Optional[np.ndarray],
        display_size: int,
    ) -> None:
        if not self.enabled or pygame is None:
            return
        env = session.backend.env
        if env is None or prev_state is None:
            return

        # Rapid key-repeat shouldn't stack camera offsets.
        self._clips = [c for c in self._clips if c.kind != "bounce"]

        grid_w, grid_h = int(env.width), int(env.height)
        for item in plan_effects(session, token, prev_state, events_before):
            kind = item["kind"]
            duration = item["durationMs"]

            if kind == "bounce":
                self._clips.append(
                    _Clip(
                        "bounce",
                        now_ms,
                        duration,
                        offset=(item["dx"], item["dy"]),
                    )
                )
                continue

            cell = (int(item["cell"][0]), int(item["cell"][1]))

            if kind == "pulse":
                self._clips.append(_Clip("pulse", now_ms, duration, cell=cell))
                if prev_rgb is not None:
                    self._success_pulse_pending = True
                continue

            if kind == "press":
                self._clips.append(_Clip("press", now_ms, duration, cell=cell))
                continue

            tile = self._extract_tile(prev_rgb, cell, grid_w, grid_h, display_size)
            if tile is not None:
                self._clips.append(
                    _Clip(kind, now_ms, duration, cell=cell, tile_surf=tile)
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

    @staticmethod
    def _extract_tile(
        prev_rgb: Optional[np.ndarray],
        cell: tuple[int, int],
        grid_w: int,
        grid_h: int,
        display_size: int,
    ) -> Optional["pygame.Surface"]:
        if prev_rgb is None or pygame is None:
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
