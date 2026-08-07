"""Minimal UI sound effects for the human-play demo.

Synthesizes short, restrained interaction feedback (Notion / VS Code / Apple
UI vibes -- not game SFX) as in-memory pygame Sounds. No asset files: every
clip is a few dozen milliseconds of sine/noise shaped with an envelope so the
demo stays self-contained and headless-safe (mixer init failures become no-ops).
"""

from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np

try:
    import pygame
except ImportError:  # pragma: no cover - pygame is required by the UI already
    pygame = None  # type: ignore


SAMPLE_RATE = 22050
# Keep overall gain modest -- these should sit under the interaction, not
# announce themselves. Relative volumes are tuned per-clip below.
MASTER_GAIN = 0.22


def _envelope(n: int, attack: float, release: float) -> np.ndarray:
    """Linear attack / sustain / release envelope over ``n`` samples."""
    attack_n = max(1, int(n * attack))
    release_n = max(1, int(n * release))
    sustain_n = max(0, n - attack_n - release_n)
    env = np.concatenate(
        [
            np.linspace(0.0, 1.0, attack_n, endpoint=False),
            np.ones(sustain_n),
            np.linspace(1.0, 0.0, release_n),
        ]
    )
    if len(env) < n:
        env = np.pad(env, (0, n - len(env)))
    return env[:n].astype(np.float32)


def _sine(
    freq: float,
    duration_ms: int,
    volume: float = 1.0,
    attack: float = 0.05,
    release: float = 0.4,
) -> np.ndarray:
    n = max(1, int(SAMPLE_RATE * duration_ms / 1000.0))
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    wave = np.sin(2.0 * math.pi * freq * t) * volume
    return (wave * _envelope(n, attack, release)).astype(np.float32)


def _noise(
    duration_ms: int,
    volume: float = 1.0,
    attack: float = 0.02,
    release: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    n = max(1, int(SAMPLE_RATE * duration_ms / 1000.0))
    rng = np.random.default_rng(seed)
    wave = rng.uniform(-1.0, 1.0, n).astype(np.float32) * volume
    return wave * _envelope(n, attack, release)


def _lowpass(wave: np.ndarray, alpha: float) -> np.ndarray:
    """One-pole low-pass (alpha closer to 0 = darker)."""
    out = np.empty_like(wave)
    acc = 0.0
    for i, sample in enumerate(wave):
        acc = acc + alpha * (float(sample) - acc)
        out[i] = acc
    return out


def _to_sound(wave: np.ndarray) -> Optional["pygame.mixer.Sound"]:
    if pygame is None or pygame.mixer.get_init() is None:
        return None
    clipped = np.clip(wave * MASTER_GAIN, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    # Match mixer channel count so pygame accepts the buffer.
    channels = pygame.mixer.get_init()[2]
    if channels == 2:
        pcm = np.column_stack([pcm, pcm])
    return pygame.mixer.Sound(buffer=pcm.tobytes())


def _mix(*parts: np.ndarray) -> np.ndarray:
    length = max(len(p) for p in parts)
    out = np.zeros(length, dtype=np.float32)
    for part in parts:
        out[: len(part)] += part
    peak = float(np.max(np.abs(out))) if length else 0.0
    if peak > 1.0:
        out /= peak
    return out


def _build_library() -> dict[str, Optional["pygame.mixer.Sound"]]:
    """Compile the small fixed set of UI clips once at mixer init."""
    # Move: soft high tick (step confirmation)
    step = _mix(
        _sine(920, 55, volume=0.55, attack=0.02, release=0.7),
        _noise(40, volume=0.08, attack=0.01, release=0.8, seed=1),
    )

    # Turn: quieter mid click / swish
    turn = _mix(
        _sine(1400, 40, volume=0.35, attack=0.01, release=0.75),
        _lowpass(_noise(45, volume=0.18, attack=0.01, release=0.85, seed=2), 0.25),
    )

    # Wall bump: muted low thunk
    wall = _mix(
        _sine(110, 110, volume=0.7, attack=0.01, release=0.65),
        _sine(220, 80, volume=0.25, attack=0.01, release=0.7),
        _lowpass(_noise(90, volume=0.2, attack=0.01, release=0.7, seed=3), 0.15),
    )

    # Pickup / drop: bright restrained chime (two partials)
    pickup = _mix(
        _sine(880, 200, volume=0.55, attack=0.02, release=0.55),
        _sine(1320, 180, volume=0.28, attack=0.05, release=0.6),
    )

    # Door open/close: mechanical latch (low thud + brief mid click)
    door = _mix(
        _sine(95, 260, volume=0.65, attack=0.01, release=0.55),
        _sine(380, 90, volume=0.35, attack=0.01, release=0.5),
        _noise(60, volume=0.12, attack=0.01, release=0.7, seed=4),
    )

    # Switch: distinct sharp click (different family from door)
    switch = _mix(
        _sine(2100, 35, volume=0.55, attack=0.005, release=0.85),
        _sine(1050, 45, volume=0.25, attack=0.01, release=0.8),
        _noise(25, volume=0.1, attack=0.005, release=0.9, seed=5),
    )

    # Invalid: soft short beep
    invalid = _sine(480, 120, volume=0.4, attack=0.05, release=0.55)

    # Success: short ascending 3-note chime (C5-E5-G5), ~550ms
    note_ms = 160
    gap = int(SAMPLE_RATE * 0.04)
    n1 = _sine(523.25, note_ms, volume=0.5, attack=0.04, release=0.45)
    n2 = _sine(659.25, note_ms, volume=0.5, attack=0.04, release=0.45)
    n3 = _sine(783.99, 220, volume=0.55, attack=0.04, release=0.5)
    success = np.concatenate(
        [
            n1,
            np.zeros(gap, dtype=np.float32),
            n2,
            np.zeros(gap, dtype=np.float32),
            n3,
        ]
    )

    # Restart: quick downward whoosh (filtered noise + falling sine)
    whoosh_raw = _noise(180, volume=0.45, attack=0.02, release=0.55, seed=6)
    n = len(whoosh_raw)
    freq = np.linspace(900.0, 220.0, n, dtype=np.float32)
    phase = 2.0 * math.pi * np.cumsum(freq) / SAMPLE_RATE
    whoosh = _mix(
        whoosh_raw * 0.7,
        (np.sin(phase) * 0.25 * _envelope(n, 0.05, 0.5)).astype(np.float32),
    )
    whoosh = _lowpass(whoosh, 0.2)

    # Task switch ([ / ]): soft upward page-flip, distinct from restart's
    # downward whoosh -- short enough to fire on every nav without fatigue.
    nav_raw = _noise(120, volume=0.28, attack=0.02, release=0.6, seed=7)
    n_nav = len(nav_raw)
    nav_freq = np.linspace(320.0, 720.0, n_nav, dtype=np.float32)
    nav_phase = 2.0 * math.pi * np.cumsum(nav_freq) / SAMPLE_RATE
    navigate = _mix(
        _lowpass(nav_raw, 0.22) * 0.55,
        (np.sin(nav_phase) * 0.3 * _envelope(n_nav, 0.04, 0.55)).astype(np.float32),
        _sine(980, 50, volume=0.2, attack=0.02, release=0.8),
    )

    return {
        "step": _to_sound(step),
        "turn": _to_sound(turn),
        "wall": _to_sound(wall),
        "pickup": _to_sound(pickup),
        "door": _to_sound(door),
        "switch": _to_sound(switch),
        "invalid": _to_sound(invalid),
        "success": _to_sound(success),
        "restart": _to_sound(whoosh),
        "navigate": _to_sound(navigate),
    }


def sfx_for_dispatch(session, events_before: int) -> str:
    """Pick one short UI clip for the action that just ran.

    Prefers semantic Progress milestones (pickup / door / switch) when they
    fired this step, then falls back to the transcript's ``event_type``
    (move / turn / wall bump / invalid). Success always wins so completing a
    task gets the ascending chime even if the last primitive was a plain move
    onto the goal.
    """
    if session.episode_done and session.episode_success:
        return "success"

    for event in reversed(session.event_log[events_before:]):
        if event.icon == "key":
            return "pickup"
        if event.icon == "door":
            return "door"
        if event.icon in ("switch", "gate"):
            # Gate state changes are switch-driven; prefer the distinct
            # switch click over the door latch so the two mechanisms stay
            # audibly different.
            return "switch"
        if event.icon == "goal":
            return "success"

    last = next(
        (rec for rec in reversed(session.transcript) if rec.get("kind") == "step"),
        None,
    )
    event_type = last.get("event_type") if last else None
    if event_type == "MOVED":
        return "step"
    if event_type == "TURNED":
        return "turn"
    if event_type == "BLOCKED":
        return "wall"
    if event_type in ("PICKUP",):
        return "pickup"
    if event_type == "OPENED":
        return "door"
    if event_type == "TOGGLED":
        return "switch"
    if event_type == "DONE":
        return "success"
    if event_type == "DROPPED":
        # Successful drops already returned "pickup" via the Progress event
        # above; reaching here means X with an empty inventory.
        return "invalid"
    if event_type in ("NOTHING", "INVALID", "WRONG_DONE"):
        return "invalid"
    return "invalid"


class DemoSounds:
    """Tiny facade: init once, ``play(name)`` thereafter. Silent if mixer
    unavailable (headless CI, missing audio device, etc.)."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._sounds: dict[str, Optional["pygame.mixer.Sound"]] = {}
        self._ready = False
        if enabled:
            self._ready = self._init_mixer()
            if self._ready:
                self._sounds = _build_library()

    def _init_mixer(self) -> bool:
        if pygame is None:
            return False
        # Headless / CI runs often set a dummy audio driver; pygame.mixer.init
        # can hang indefinitely on some Windows setups in that mode, so stay
        # silent rather than blocking the whole demo.
        if os.environ.get("SDL_AUDIODRIVER", "").lower() in {"dummy", "disk"}:
            return False
        try:
            if pygame.mixer.get_init() is None:
                pygame.mixer.init(
                    frequency=SAMPLE_RATE, size=-16, channels=1, buffer=512
                )
                pygame.mixer.set_num_channels(8)
            return pygame.mixer.get_init() is not None
        except Exception:
            return False

    def play(self, name: str) -> None:
        if not self.enabled or not self._ready:
            return
        sound = self._sounds.get(name)
        if sound is not None:
            sound.play()

    def close(self) -> None:
        if self._ready and pygame is not None:
            try:
                pygame.mixer.stop()
            except Exception:
                pass
