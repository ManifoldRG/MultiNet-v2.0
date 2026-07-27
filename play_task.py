#!/usr/bin/env python3
"""
Interactive MiniGrid Task Player — Playable Human Demo

A pygame-based interactive player for MiniGrid task JSON files that mirrors
the *exact* observation the model-facing interface (``interface/``) would
build for a given ``ExperimentConfig``. This lets a human play a maze under
the same information constraints as an LLM agent (e.g. text-only, or with a
``text_summary`` of prior activity instead of full history), to get a feel
for how hard a task actually is.

Usage:
    python play_task.py mazes/validation_10/V04_single_key.json
    python play_task.py mazes/validation_10/V01_empty_room.json --record

    # Play with the same information the model would get in text-only mode,
    # with a text_summary of prior activity instead of raw history:
    python play_task.py mazes/validation_10/V06_chain_ks.json \\
        --observation text_only --context-window text_summary

    # Browse a whole directory of task files with [ / ] (non-recursive: point
    # at the leaf directory that directly contains the task JSONs):
    python play_task.py --tasks-dir mazes/exp_maze_jsons/S1

    # Browse a manifest task catalog with [ / ] instead of a directory -- rows
    # can point at files in different folders, and the info panel shows each
    # row's experiment/condition/expected_mechanisms (mirrors the task
    # selection scripts/run_pipeline.py uses for real runs):
    python play_task.py --manifest gridworld/fixtures/manifest.json --experiment test1

Controls:
    Arrow Up / W        : Move forward (egocentric) / North (cardinal)
    Arrow Down / S       : South (cardinal action space only)
    Arrow Left / A       : Turn left (egocentric) / West (cardinal)
    Arrow Right / D      : Turn right (egocentric) / East (cardinal)
    Space               : Pick up item
    T / E               : Toggle (open door, press switch) / Interact (cardinal)
    X                   : Drop item (human-only -- not in the model's action space)
    Backspace           : Wait / done (no-op)
    R                   : Reset current task
    [ / ]               : Previous / next task in the current directory / manifest
    Tab                 : Toggle the settings overlay (cycle observation/context/etc.)
    M                   : Toggle a full-screen view of the exact model-facing text
    Q                   : Quit
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

_SCRIPT_DIR = Path(__file__).resolve().parent

# Ensure the repository root is on sys.path for gridworld/interface imports
_script_dir_str = str(_SCRIPT_DIR)
if _script_dir_str not in sys.path:
    sys.path.insert(0, _script_dir_str)

try:
    import pygame
except ImportError:
    print(
        "Error: pygame is not installed.\n"
        "Install it with: pip install pygame\n"
        "  or: conda install -c conda-forge pygame"
    )
    sys.exit(1)

from gridworld.task_spec import TaskSpecification
from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.backends.base import GridState
from gridworld.actions import MiniGridActions

from interface.config import ExperimentConfig
from interface import action_space as action_space_mod
from interface.actions_map import nlu_action_to_int
from interface.coords import agent_facing, agent_row_col
from interface.episode_log import state_snapshot
from interface.feedback import format_step_feedback
from interface.observation import (
    current_observation_text,
    history_text,
    text_summary_history,
)
from interface.renderer import render_initial_maze_text

# Reuse the same manifest-catalog resolution logic real eval runs use, so
# browsing a manifest here (task rows whose ``source`` files can live in any
# folder) always matches what scripts/run_pipeline.py would actually run.
from scripts.run_pipeline import (
    _EXPERIMENT_KEYWORDS,
    _resolve_source,
    load_manifest,
    resolve_task_rows,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Window layout
GRID_DISPLAY_SIZE = 512      # Grid rendering area (square, left side)
INFO_PANEL_WIDTH = 380       # Info panel width (right side)
WINDOW_HEIGHT = 640
WINDOW_WIDTH = GRID_DISPLAY_SIZE + INFO_PANEL_WIDTH

# Colors
COLOR_BG = (30, 30, 30)
COLOR_PANEL_BG = (40, 40, 48)
COLOR_TEXT = (220, 220, 220)
COLOR_TEXT_DIM = (140, 140, 150)
COLOR_TEXT_HIGHLIGHT = (100, 220, 130)
COLOR_TEXT_WARNING = (255, 180, 60)
COLOR_TEXT_ERROR = (255, 80, 80)
COLOR_TEXT_TITLE = (180, 200, 255)
COLOR_TEXT_MODEL = (150, 200, 255)
COLOR_SEPARATOR = (70, 70, 80)
COLOR_OVERLAY_TEXT = (255, 255, 255)

# Direction labels
DIRECTION_NAMES = {0: "East (right)", 1: "South (down)", 2: "West (left)", 3: "North (up)"}
DIRECTION_ARROWS = {0: "->", 1: "v", 2: "<-", 3: "^"}

# Key repeat settings (milliseconds)
KEY_REPEAT_DELAY = 200
KEY_REPEAT_INTERVAL = 100

# Frame rate
FPS = 30

# Settings that can be toggled live via the in-app settings overlay (Tab).
# (hotkey, ExperimentConfig attribute, choices | None for a bool toggle)
SETTINGS_AXES: tuple[tuple[str, str, Optional[tuple[str, ...]]], ...] = (
    ("1", "observation", ("text_only", "image_text", "image_only")),
    ("2", "context_window", ("current", "last3", "text_summary", "text_summary_and_last3")),
    ("3", "include_current_observation_description", None),
    ("4", "observation_text_includes_facing", None),
    ("5", "action_space", ("egocentric", "cardinal")),
)


# ---------------------------------------------------------------------------
# Task discovery: browse a directory of task JSON files (tier-agnostic)
# ---------------------------------------------------------------------------

def discover_tasks_in_dir(directory: Path) -> list[Path]:
    """Return sorted task JSON files in ``directory`` (empty if not a directory)."""
    if not directory.exists() or not directory.is_dir():
        return []
    return sorted(directory.glob("*.json"))


def load_manifest_tasks(
    manifest_path: Path, experiment: Optional[str]
) -> list[tuple[Path, dict]]:
    """Resolve a manifest catalog to an ordered ``(resolved_path, row)`` list.

    Mirrors the task selection ``scripts/run_pipeline.py`` uses for real runs:
    a manifest row's ``source`` can live in any folder, so browsing a manifest
    (instead of one flat directory) lets [ / ] step through exactly the task
    set a given experiment actually runs, in manifest order. Rows whose
    ``source`` file can't be found are skipped with a warning rather than
    aborting the whole browse list; rows that resolve to a path already seen
    (e.g. the same maze re-used under a different condition) are skipped too,
    since [ / ] navigates files, not per-row metadata.
    """
    catalog = load_manifest(manifest_path)
    entries = [experiment] if experiment else ["all"]
    rows = resolve_task_rows(entries, catalog, manifest_path)

    resolved: list[tuple[Path, dict]] = []
    seen: set[Path] = set()
    for row in rows:
        try:
            path = _resolve_source(row, manifest_path)
        except FileNotFoundError as exc:
            print(f"Warning: skipping manifest row {row.get('task_id')!r}: {exc}")
            continue
        if path in seen:
            continue
        seen.add(path)
        resolved.append((path, row))
    return resolved


# ---------------------------------------------------------------------------
# Interactive player
# ---------------------------------------------------------------------------

class MiniGridPlayer:
    """
    Pygame-based interactive player for MiniGrid task JSON files, hooked into
    the same ``interface/`` observation-building code the LLM pipeline uses.
    """

    def __init__(
        self,
        task_path: Optional[str],
        record: bool = False,
        config: Optional[ExperimentConfig] = None,
        tasks_dir: Optional[str] = None,
        manifest: Optional[str] = None,
        experiment: Optional[str] = None,
    ):
        self.base_dir = _SCRIPT_DIR
        self.record = record
        self.config = config or ExperimentConfig()
        self.tasks_dir_override: Optional[Path] = (
            self._resolve_path(tasks_dir) if tasks_dir else None
        )

        self.task_path: Optional[Path] = None
        self.task_spec: Optional[TaskSpecification] = None
        self.task_list: list[Path] = []
        self.task_index: int = 0

        # Manifest mode: [ / ] steps through a curated task catalog (rows can
        # point at files scattered across many folders) instead of one flat
        # directory. self.manifest_row_by_path supplies the metadata shown in
        # the info panel; self.task_list holds the resolved paths in manifest
        # order and is left alone by _load_task's directory rediscovery.
        self.manifest_mode = manifest is not None
        self.manifest_experiment = experiment
        self.manifest_row_by_path: dict[Path, dict] = {}
        if self.manifest_mode:
            manifest_resolved = self._resolve_path(manifest)
            manifest_tasks = load_manifest_tasks(manifest_resolved, experiment)
            if not manifest_tasks:
                print(f"Warning: manifest {manifest_resolved} resolved no tasks; falling back to directory browsing.")
                self.manifest_mode = False
            else:
                self.task_list = [p for p, _row in manifest_tasks]
                self.manifest_row_by_path = {p: row for p, row in manifest_tasks}
                if task_path is None:
                    task_path = str(self.task_list[0])

        if task_path is None:
            task_path = "mazes/validation_10/V01_empty_room.json"

        # Backend for environment logic
        self.backend = MiniGridBackend(render_mode="rgb_array")

        # Episode state
        self.state: Optional[GridState] = None
        self.episode_done = False
        self.episode_success = False
        self.total_reward: float = 0.0
        self.last_action_name: str = ""
        self.step_index: int = 0

        # Model-parity transcript: enriched step records built the same way
        # interface/runner.py builds them, so interface/observation.py's
        # history/text_summary helpers work unmodified on it.
        self.transcript: list[dict] = []

        # UI overlay state
        self.show_settings_overlay = False
        self.show_model_view_overlay = False
        self.model_view_scroll = 0
        self.text_only_scroll = 0

        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("MiniGrid Task Player")
        pygame.key.set_repeat(KEY_REPEAT_DELAY, KEY_REPEAT_INTERVAL)
        self.clock = pygame.time.Clock()

        # Font setup -- use a clean monospace font
        self.font_title = self._load_font(22, bold=True)
        self.font_main = self._load_font(16)
        self.font_small = self._load_font(13)
        self.font_overlay = self._load_font(48, bold=True)
        self.font_overlay_sub = self._load_font(20)

        # Load the initial task
        self._load_task(task_path)

    def _load_font(self, size: int, bold: bool = False) -> pygame.font.Font:
        """Load a monospace font, falling back to the default if needed."""
        mono_names = ["DejaVu Sans Mono", "Consolas", "Courier New", "monospace"]
        for name in mono_names:
            path = pygame.font.match_font(name, bold=bold)
            if path:
                try:
                    return pygame.font.Font(path, size)
                except Exception:
                    pass
        return pygame.font.SysFont(None, size, bold=bold)

    def _resolve_path(self, path: str) -> Path:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = self.base_dir / resolved
        return resolved

    # ------------------------------------------------------------------
    # Task loading
    # ------------------------------------------------------------------

    def _load_task(self, path: str) -> None:
        """Load a task JSON file, refresh directory browsing, and reset."""
        resolved = self._resolve_path(path)

        if not resolved.exists():
            print(f"Error: task file not found: {resolved}")
            return

        self._checkpoint_trajectory()

        self.task_path = resolved
        self.task_spec = TaskSpecification.from_json(str(resolved))

        if self.manifest_mode:
            # self.task_list is the manifest's resolved order; leave it alone
            # so [ / ] keeps stepping through the curated catalog rather than
            # whatever else happens to sit in this file's directory.
            if resolved not in self.task_list:
                self.task_list = sorted(set(self.task_list) | {resolved})
        else:
            tasks_dir = self.tasks_dir_override or resolved.parent
            self.task_list = discover_tasks_in_dir(tasks_dir)
            if resolved not in self.task_list:
                self.task_list = sorted(set(self.task_list) | {resolved})
        try:
            self.task_index = self.task_list.index(resolved)
        except ValueError:
            self.task_index = 0

        self._reset_env()

    def _reset_env(self) -> None:
        """Reset the environment from the current task spec."""
        if self.task_spec is None:
            return

        self.backend.configure(self.task_spec)
        _obs, self.state, _info = self.backend.reset(seed=self.task_spec.seed)

        self.episode_done = False
        self.episode_success = False
        self.total_reward = 0.0
        self.last_action_name = ""
        self.step_index = 0
        self.transcript = [
            {
                "kind": "reset",
                "state": state_snapshot(self.state),
            }
        ]
        self.model_view_scroll = 0
        self.text_only_scroll = 0

        pygame.display.set_caption(f"MiniGrid Player  |  {self.task_spec.task_id}")

    def _load_adjacent_task(self, delta: int) -> None:
        """Load the next (+1) or previous (-1) task in the current directory."""
        if not self.task_list:
            return
        self.task_index = (self.task_index + delta) % len(self.task_list)
        self._load_task(str(self.task_list[self.task_index]))

    # ------------------------------------------------------------------
    # Step execution
    # ------------------------------------------------------------------

    def _key_to_token(self, key: int) -> Optional[str]:
        """Map a physical key to an action token, respecting action_space."""
        cardinal = self.config.action_space == "cardinal"
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

    def _dispatch_token(self, token: str) -> None:
        """Execute a token, expanding cardinal moves into primitives (as the
        runner does) so step economy matches the model exactly."""
        if token == "DROP":
            self._step_drop()
            return
        if self.config.action_space == "cardinal" and token in action_space_mod.CARDINAL_ACTIONS:
            primitives = action_space_mod.cardinal_to_primitives(token, agent_facing(self.state))
            for primitive in primitives:
                if self.episode_done:
                    break
                self._step_token(primitive, cardinal_source=token)
        else:
            self._step_token(token)

    def _step_token(self, token: str, cardinal_source: Optional[str] = None) -> None:
        """Execute a single egocentric primitive action."""
        if self.episode_done or self.state is None:
            return

        prev_state = self.state
        action_int = nlu_action_to_int(token)
        _rgb, reward, terminated, truncated, self.state, info = self.backend.step(action_int)
        self.total_reward += reward
        self.last_action_name = token

        feedback_text, event_type = format_step_feedback(
            token, prev_state, self.state, reward, terminated, self.task_spec
        )
        self._record_step(
            token, cardinal_source, prev_state, feedback_text, event_type,
            reward, terminated, truncated, info,
        )

        if terminated or truncated:
            self.episode_done = True
            self.episode_success = self.state.goal_reached

    def _step_drop(self) -> None:
        """Human-only DROP action; not part of the model's action space, so
        it is handled outside interface/feedback.py and excluded from the
        model-parity transcript view (see ``_model_transcript``)."""
        if self.episode_done or self.state is None:
            return

        prev_state = self.state
        _rgb, reward, terminated, truncated, self.state, info = self.backend.step(
            MiniGridActions.DROP
        )
        self.total_reward += reward
        self.last_action_name = "DROP"

        dropped = prev_state.agent_carrying and prev_state.agent_carrying != self.state.agent_carrying
        if dropped:
            feedback_text = f"You drop the {prev_state.agent_carrying}. (human-only action)"
        else:
            feedback_text = "Nothing to drop. (human-only action)"
        self._record_step(
            "DROP", None, prev_state, feedback_text, "DROPPED",
            reward, terminated, truncated, info,
        )

        if terminated or truncated:
            self.episode_done = True
            self.episode_success = self.state.goal_reached

    def _record_step(
        self,
        action: str,
        cardinal_source: Optional[str],
        prev_state: GridState,
        feedback_text: str,
        event_type: str,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> None:
        self.step_index += 1
        self.transcript.append(
            {
                "kind": "step",
                "step_index": self.step_index,
                "action": action,
                "cardinal_source": cardinal_source,
                "event_type": event_type,
                "prompt_feedback": feedback_text,
                "feedback": feedback_text,
                "facing_before": agent_facing(prev_state),
                "facing_after": agent_facing(self.state),
                "position_before": list(agent_row_col(prev_state)),
                "position_after": list(agent_row_col(self.state)),
                "state_before": state_snapshot(prev_state),
                "state_after": state_snapshot(self.state),
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "backend_info": info,
            }
        )

    def _model_transcript(self) -> list[dict]:
        """Transcript restricted to actions the model could actually take
        (excludes the human-only DROP action) so history/text_summary text
        stays a faithful preview of what the model would see."""
        return [rec for rec in self.transcript if rec.get("event_type") != "DROPPED"]

    # ------------------------------------------------------------------
    # Model-view text (exactly what interface/ would build for the model)
    # ------------------------------------------------------------------

    def _build_model_view_sections(self) -> list[tuple[str, str]]:
        if self.task_spec is None or self.state is None:
            return []
        obs = self.config.observation
        ctx = self.config.context_window
        transcript = self._model_transcript()
        sections: list[tuple[str, str]] = []

        if obs in ("text_only", "image_text"):
            sections.append(("Initial maze (system prompt)", render_initial_maze_text(self.task_spec)))
        else:
            sections.append(
                ("Initial maze (system prompt)", "(not sent to the model in image_only mode)")
            )

        obs_text = current_observation_text(
            obs,
            self.task_spec,
            self.state,
            include_description=self.config.include_current_observation_description,
            include_facing=self.config.observation_text_includes_facing,
        )
        if obs_text:
            sections.append(("Current observation", obs_text))

        hist = history_text(obs, ctx, transcript, self.task_spec)
        if not hist and ctx == "text_summary_and_last3" and obs == "image_only":
            # Delivered as a separate leading block ahead of last3 images in
            # the real prompt (see interface/observation.leading_summary_blocks).
            hist = text_summary_history(transcript, self.task_spec)
        if hist:
            sections.append(("History", hist))

        if obs in ("image_only", "image_text") and ctx in ("last3", "text_summary_and_last3"):
            sections.append(
                (
                    "History (images)",
                    "The model also receives the last 3 decision-frame images "
                    "here -- not rendered in this demo.",
                )
            )

        return sections

    # ------------------------------------------------------------------
    # Recording / trajectory saving
    # ------------------------------------------------------------------

    def _checkpoint_trajectory(self) -> None:
        """Save the in-progress transcript if --record is on, using whatever
        task_path/task_spec/manifest row are *currently* set. Callers must
        invoke this before mutating those fields (e.g. before switching to a
        new task in _load_task) so the saved task_id/task_file/manifest_row
        match the transcript's actual task, not the one being loaded next."""
        if self.record and self.transcript:
            self._save_trajectory()

    def _save_trajectory(self) -> None:
        """Save the recorded transcript to a JSON file."""
        if not self.transcript:
            return

        task_id = self.task_spec.task_id if self.task_spec else "unknown"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"trajectory_{task_id}_{timestamp}.json"
        output_path = self.base_dir / filename

        manifest_row = self.manifest_row_by_path.get(self.task_path) if self.manifest_mode else None
        data = {
            "task_id": task_id,
            "task_file": str(self.task_path) if self.task_path else None,
            "manifest_row": manifest_row,
            "config": self.config.to_dict(),
            "total_steps": self.step_index,
            "total_reward": self.total_reward,
            "success": self.episode_success,
            "episode_done": self.episode_done,
            "transcript": self.transcript,
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Trajectory saved to: {output_path}")

    # ------------------------------------------------------------------
    # Low-level drawing helpers
    # ------------------------------------------------------------------

    def _draw_text(self, text: str, x: int, y: int, font: pygame.font.Font, color: tuple) -> int:
        """Draw a single line of text and return the y position below it."""
        surf = font.render(text, True, color)
        self.screen.blit(surf, (x, y))
        return y + surf.get_height() + 2

    def _draw_wrapped_text(
        self, text: str, x: int, y: int,
        font: pygame.font.Font, color: tuple, max_width: int
    ) -> int:
        """Draw word-wrapped text and return the y position below it."""
        for line in self._wrap_lines(text.split("\n"), font, max_width):
            y = self._draw_text(line, x, y, font, color)
        return y

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
    # Rendering: main play area (grid image or text-only pane)
    # ------------------------------------------------------------------

    def _render_grid(self) -> None:
        """Render the MiniGrid environment onto the left side of the screen."""
        rgb_array = self.backend.render()
        h, w, _c = rgb_array.shape
        surf = pygame.image.frombuffer(rgb_array.tobytes(), (w, h), "RGB")
        scaled = pygame.transform.smoothscale(surf, (GRID_DISPLAY_SIZE, GRID_DISPLAY_SIZE))
        self.screen.blit(scaled, (0, 0))

    def _render_text_only_pane(self) -> None:
        """In text_only mode the model gets no image, so neither does the
        human: render the exact model-facing text here instead of the grid."""
        rect = pygame.Rect(0, 0, GRID_DISPLAY_SIZE, WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, COLOR_BG, rect)

        lines = [
            "TEXT-ONLY MODE",
            "(matches what the model receives -- no image shown)",
            "",
        ]
        for title, text in self._build_model_view_sections():
            lines.append(f"-- {title} --")
            lines.extend(text.split("\n"))
            lines.append("")
        lines.append("(scroll with mouse wheel / Page Up / Page Down)")

        inner = rect.inflate(-24, -24)
        wrapped = self._wrap_lines(lines, self.font_small, inner.width)
        self.text_only_scroll = self._draw_scrollable_text(
            wrapped, inner, self.font_small, COLOR_TEXT, self.text_only_scroll
        )

    def _render_main_pane(self) -> None:
        if self.backend.env is None:
            placeholder_surf = self.font_main.render(
                "No environment loaded.", True, COLOR_TEXT_DIM
            )
            self.screen.blit(placeholder_surf, (20, GRID_DISPLAY_SIZE // 2))
        elif self.config.observation == "text_only":
            self._render_text_only_pane()
        else:
            self._render_grid()

    # ------------------------------------------------------------------
    # Rendering: info panel
    # ------------------------------------------------------------------

    def _render_info_panel(self) -> None:
        """Render the info panel on the right side of the screen."""
        panel_x = GRID_DISPLAY_SIZE
        panel_rect = pygame.Rect(panel_x, 0, INFO_PANEL_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, panel_rect)
        pygame.draw.line(self.screen, COLOR_SEPARATOR, (panel_x, 0), (panel_x, WINDOW_HEIGHT), 2)

        x = panel_x + 12
        y = 10
        content_width = INFO_PANEL_WIDTH - 24

        # -- Title + description --
        task_id = self.task_spec.task_id if self.task_spec else "No task loaded"
        y = self._draw_text(f"Task: {task_id}", x, y, self.font_title, COLOR_TEXT_TITLE)
        if self.task_spec and self.task_spec.description:
            y = self._draw_wrapped_text(
                self.task_spec.description, x, y, self.font_small, COLOR_TEXT_DIM, content_width
            )
        y += 4
        pygame.draw.line(self.screen, COLOR_SEPARATOR, (x, y), (panel_x + INFO_PANEL_WIDTH - 12, y))
        y += 8

        # -- Agent State --
        if self.state:
            y = self._draw_text("AGENT STATE", x, y, self.font_main, COLOR_TEXT_HIGHLIGHT)
            y += 2

            pos = self.state.agent_position
            y = self._draw_text(f"Position:  ({pos[0]}, {pos[1]})", x, y, self.font_main, COLOR_TEXT)

            dir_name = DIRECTION_NAMES.get(self.state.agent_direction, "?")
            arrow = DIRECTION_ARROWS.get(self.state.agent_direction, "?")
            y = self._draw_text(f"Direction: {arrow} {dir_name}", x, y, self.font_main, COLOR_TEXT)

            carrying = self.state.agent_carrying or "nothing"
            color = COLOR_TEXT_WARNING if self.state.agent_carrying else COLOR_TEXT_DIM
            y = self._draw_text(f"Carrying:  {carrying}", x, y, self.font_main, color)

            y += 2
            y = self._draw_text(
                f"Steps: {self.state.step_count} / {self.state.max_steps}", x, y, self.font_main, COLOR_TEXT
            )
            y = self._draw_text(f"Reward: {self.total_reward:.3f}", x, y, self.font_main, COLOR_TEXT)

            if self.last_action_name:
                y = self._draw_text(
                    f"Last action: {self.last_action_name}", x, y, self.font_main, COLOR_TEXT_DIM
                )
        else:
            y = self._draw_text("No environment loaded", x, y, self.font_main, COLOR_TEXT_ERROR)

        y += 4
        pygame.draw.line(self.screen, COLOR_SEPARATOR, (x, y), (panel_x + INFO_PANEL_WIDTH - 12, y))
        y += 8

        # -- Mechanism State --
        if self.state:
            has_mechanisms = (
                self.state.active_switches
                or self.state.open_gates
                or self.state.block_positions
                or self.state.teleporter_cooldowns
            )
            if has_mechanisms:
                y = self._draw_text("MECHANISMS", x, y, self.font_main, COLOR_TEXT_HIGHLIGHT)
                y += 2
                if self.state.active_switches:
                    y = self._draw_text(
                        f"Active switches: {', '.join(sorted(self.state.active_switches))}",
                        x, y, self.font_small, COLOR_TEXT_WARNING,
                    )
                if self.state.open_gates:
                    y = self._draw_text(
                        f"Open gates: {', '.join(sorted(self.state.open_gates))}",
                        x, y, self.font_small, COLOR_TEXT_HIGHLIGHT,
                    )
                for bid, bpos in self.state.block_positions.items():
                    y = self._draw_text(f"Block {bid}: ({bpos[0]}, {bpos[1]})", x, y, self.font_small, COLOR_TEXT)
                for tid, cd in self.state.teleporter_cooldowns.items():
                    cd_text = "ready" if cd == 0 else f"cooldown {cd}"
                    y = self._draw_text(f"Teleporter {tid}: {cd_text}", x, y, self.font_small, COLOR_TEXT)
                y += 4
                pygame.draw.line(self.screen, COLOR_SEPARATOR, (x, y), (panel_x + INFO_PANEL_WIDTH - 12, y))
                y += 8

        # -- Model view (settings summary) --
        y = self._draw_text("MODEL VIEW", x, y, self.font_main, COLOR_TEXT_HIGHLIGHT)
        y += 2
        y = self._draw_text(f"observation: {self.config.observation}", x, y, self.font_small, COLOR_TEXT_MODEL)
        y = self._draw_text(f"context_window: {self.config.context_window}", x, y, self.font_small, COLOR_TEXT_MODEL)
        y = self._draw_text(f"action_space: {self.config.action_space}", x, y, self.font_small, COLOR_TEXT_MODEL)
        y = self._draw_text(
            f"obs. description: {self.config.include_current_observation_description}"
            f"  facing: {self.config.observation_text_includes_facing}",
            x, y, self.font_small, COLOR_TEXT_MODEL,
        )
        y = self._draw_text(
            f"Valid actions: {action_space_mod.actions_hint(self.config.action_space)}",
            x, y, self.font_small, COLOR_TEXT_DIM,
        )
        y = self._draw_text("Tab: settings   M: full model-view text", x, y, self.font_small, COLOR_TEXT_DIM)
        y += 4
        pygame.draw.line(self.screen, COLOR_SEPARATOR, (x, y), (panel_x + INFO_PANEL_WIDTH - 12, y))
        y += 8

        # -- Manifest row (task catalog metadata for the current maze) --
        manifest_row = self.manifest_row_by_path.get(self.task_path) if self.manifest_mode else None
        if manifest_row:
            y = self._draw_text("MANIFEST", x, y, self.font_main, COLOR_TEXT_HIGHLIGHT)
            y += 2
            y = self._draw_text(
                f"experiment: {manifest_row.get('experiment', '?')}   condition: {manifest_row.get('condition', '?')}",
                x, y, self.font_small, COLOR_TEXT,
            )
            if manifest_row.get("variant"):
                y = self._draw_text(f"variant: {manifest_row['variant']}", x, y, self.font_small, COLOR_TEXT)
            mechanisms = manifest_row.get("expected_mechanisms") or []
            if mechanisms:
                y = self._draw_text(
                    f"expected mechanisms: {', '.join(mechanisms)}", x, y, self.font_small, COLOR_TEXT_WARNING
                )
            if manifest_row.get("notes"):
                y = self._draw_wrapped_text(
                    manifest_row["notes"], x, y, self.font_small, COLOR_TEXT_DIM, content_width
                )
            y += 4
            pygame.draw.line(self.screen, COLOR_SEPARATOR, (x, y), (panel_x + INFO_PANEL_WIDTH - 12, y))
            y += 8

        # -- Task navigation --
        if self.task_list:
            if self.manifest_mode:
                label = f"manifest ({self.manifest_experiment or 'all'})"
            else:
                nav_dir = (self.tasks_dir_override or (self.task_path.parent if self.task_path else Path(".")))
                label = f"{nav_dir.name}/"
            y = self._draw_text(
                f"Task {self.task_index + 1}/{len(self.task_list)} in {label}",
                x, y, self.font_small, COLOR_TEXT_DIM,
            )
            y += 4

        if self.record:
            y = self._draw_text("REC", x, y, self.font_main, COLOR_TEXT_ERROR)
            y += 4

        # -- Controls Reference --
        y += 4
        pygame.draw.line(self.screen, COLOR_SEPARATOR, (x, y), (panel_x + INFO_PANEL_WIDTH - 12, y))
        y += 6
        y = self._draw_text("CONTROLS", x, y, self.font_main, COLOR_TEXT_HIGHLIGHT)
        y += 2

        cardinal = self.config.action_space == "cardinal"
        move_desc = "N / S / W / E" if cardinal else "Fwd / -- / Left / Right"
        controls = [
            ("Up/Down/Left/Right", move_desc),
            ("W/S/A/D", "same as arrows"),
            ("Space", "Pick up"),
            ("T / E", "Interact" if cardinal else "Toggle"),
            ("X", "Drop (human-only)"),
            ("Backspace", "Wait / done"),
            ("R", "Reset"),
            ("[ / ]", "Prev / next task"),
            ("Tab", "Settings overlay"),
            ("M", "Full model-view text"),
            ("Q", "Quit"),
        ]
        for key, desc in controls:
            y = self._draw_text(f"{key:>19s}  {desc}", x, y, self.font_small, COLOR_TEXT_DIM)

    # ------------------------------------------------------------------
    # Rendering: overlays
    # ------------------------------------------------------------------

    def _render_episode_overlay(self) -> None:
        """Render success/failure overlay when episode ends."""
        if not self.episode_done:
            return

        overlay = pygame.Surface((GRID_DISPLAY_SIZE, GRID_DISPLAY_SIZE), pygame.SRCALPHA)
        if self.episode_success:
            overlay.fill((20, 100, 40, 160))
            main_text, main_color = "SUCCESS!", (100, 255, 130)
        else:
            overlay.fill((120, 20, 20, 160))
            main_text, main_color = "FAILED", (255, 100, 100)
        self.screen.blit(overlay, (0, 0))

        text_surf = self.font_overlay.render(main_text, True, main_color)
        text_rect = text_surf.get_rect(center=(GRID_DISPLAY_SIZE // 2, GRID_DISPLAY_SIZE // 2 - 20))
        self.screen.blit(text_surf, text_rect)

        if self.state:
            sub_text = f"Steps: {self.state.step_count} / {self.state.max_steps}   Reward: {self.total_reward:.3f}"
        else:
            sub_text = ""
        sub_surf = self.font_overlay_sub.render(sub_text, True, COLOR_OVERLAY_TEXT)
        sub_rect = sub_surf.get_rect(center=(GRID_DISPLAY_SIZE // 2, GRID_DISPLAY_SIZE // 2 + 30))
        self.screen.blit(sub_surf, sub_rect)

        hint_surf = self.font_small.render(
            "Press R to reset, Q to quit, [ ] to switch task", True, COLOR_TEXT_DIM
        )
        hint_rect = hint_surf.get_rect(center=(GRID_DISPLAY_SIZE // 2, GRID_DISPLAY_SIZE // 2 + 65))
        self.screen.blit(hint_surf, hint_rect)

    def _cycle_setting(self, key_char: str) -> None:
        for k, attr, choices in SETTINGS_AXES:
            if k != key_char:
                continue
            current = getattr(self.config, attr)
            if choices is None:
                setattr(self.config, attr, not current)
            else:
                setattr(self.config, attr, choices[(choices.index(current) + 1) % len(choices)])
            return

    def _render_settings_overlay(self) -> None:
        if not self.show_settings_overlay:
            return
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((15, 15, 20, 230))
        self.screen.blit(overlay, (0, 0))

        x, y = 40, 40
        y = self._draw_text(
            "SETTINGS -- these change what a human sees/controls, mirroring", x, y, self.font_title, COLOR_TEXT_TITLE
        )
        y = self._draw_text(
            "interface.config.ExperimentConfig. Press a number to cycle a value.",
            x, y, self.font_main, COLOR_TEXT_DIM,
        )
        y += 16
        for key_char, attr, choices in SETTINGS_AXES:
            value = getattr(self.config, attr)
            y = self._draw_text(f"[{key_char}]  {attr} = {value}", x, y, self.font_main, COLOR_TEXT)
        y += 16
        y = self._draw_text(
            "These only change what is displayed/hinted -- the running episode",
            x, y, self.font_small, COLOR_TEXT_DIM,
        )
        y = self._draw_text(
            "and its step count are unaffected. Tab / Esc to close.", x, y, self.font_small, COLOR_TEXT_DIM
        )

    def _render_model_view_overlay(self) -> None:
        if not self.show_model_view_overlay:
            return
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((15, 15, 20, 235))
        self.screen.blit(overlay, (0, 0))

        title = (
            f"MODEL VIEW  (observation={self.config.observation}, "
            f"context_window={self.config.context_window})"
        )
        y = 12
        y = self._draw_text(title, 16, y, self.font_main, COLOR_TEXT_TITLE)
        y = self._draw_text(
            "This is exactly the text interface/ would build for the model right now.",
            16, y, self.font_small, COLOR_TEXT_DIM,
        )
        y = self._draw_text(
            "M / Esc to close -- mouse wheel or Page Up/Down to scroll.", 16, y, self.font_small, COLOR_TEXT_DIM
        )
        y += 6

        lines: list[str] = []
        for section_title, text in self._build_model_view_sections():
            lines.append(f"== {section_title} ==")
            lines.extend(text.split("\n"))
            lines.append("")

        rect = pygame.Rect(16, y, WINDOW_WIDTH - 32, WINDOW_HEIGHT - y - 12)
        wrapped = self._wrap_lines(lines, self.font_small, rect.width)
        self.model_view_scroll = self._draw_scrollable_text(
            wrapped, rect, self.font_small, COLOR_TEXT, self.model_view_scroll
        )

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
                        self._checkpoint_trajectory()
                        self._reset_env()

                elif event.type == pygame.MOUSEWHEEL:
                    self._handle_scroll(event)

            self.screen.fill(COLOR_BG)
            self._render_main_pane()
            self._render_info_panel()
            self._render_episode_overlay()
            self._render_settings_overlay()
            self._render_model_view_overlay()

            pygame.display.flip()
            self.clock.tick(FPS)

        self._checkpoint_trajectory()

        self.backend.close()
        pygame.quit()

    def _handle_scroll(self, event: pygame.event.Event) -> None:
        delta = -event.y * 3 * self._line_height(self.font_small)
        if self.show_model_view_overlay:
            self.model_view_scroll = max(0, self.model_view_scroll + delta)
        elif self.config.observation == "text_only":
            self.text_only_scroll = max(0, self.text_only_scroll + delta)

    def _handle_keydown(self, event: pygame.event.Event) -> Optional[str]:
        """
        Handle a pygame KEYDOWN event. Dispatches actions/overlay toggles
        directly (side effects), returning only 'quit' / 'reset' / None for
        the main loop to act on.
        """
        key = event.key

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

        if self.show_settings_overlay:
            for key_char, _attr, _choices in SETTINGS_AXES:
                if key == getattr(pygame, f"K_{key_char}", None):
                    self._cycle_setting(key_char)
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
            self._load_adjacent_task(-1)
            return None
        if key == pygame.K_RIGHTBRACKET:
            self._load_adjacent_task(1)
            return None

        if self.episode_done:
            return None

        token = self._key_to_token(key)
        if token is not None:
            self._dispatch_token(token)
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Interactive MiniGrid task player -- playable human demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "task_file",
        nargs="?",
        default=None,
        help="Path to a task JSON file (default: a small validation_10 maze, or the first "
        "task in --manifest if that's given)",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record the transcript to a JSON file on exit or task switch",
    )
    parser.add_argument(
        "--tasks-dir",
        type=str,
        default=None,
        help="Directory to browse with [ / ] instead of the task file's parent directory "
        "(non-recursive: only *.json files directly inside it). Mutually exclusive with "
        "--manifest.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Browse a task-catalog manifest (e.g. gridworld/fixtures/manifest.json) with "
        "[ / ] instead of a directory -- resolves each row's 'source' file across folders "
        "and shows its catalog metadata (experiment/condition/expected_mechanisms) in the "
        "info panel, mirroring scripts/run_pipeline.py's task selection for real runs. "
        "Mutually exclusive with --tasks-dir.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=sorted(_EXPERIMENT_KEYWORDS),
        default=None,
        help="Filter --manifest to one experiment keyword (default: 'all'). Ignored without "
        "--manifest.",
    )
    parser.add_argument(
        "--observation",
        choices=["text_only", "image_text", "image_only"],
        default=ExperimentConfig.__dataclass_fields__["observation"].default,
        help="Mirrors ExperimentConfig.observation (default: image_only, today's plain grid view)",
    )
    parser.add_argument(
        "--context-window",
        dest="context_window",
        choices=["current", "last3", "text_summary", "text_summary_and_last3"],
        default=ExperimentConfig.__dataclass_fields__["context_window"].default,
        help="Mirrors ExperimentConfig.context_window; 'text_summary' shows the model's activity summary",
    )
    parser.add_argument(
        "--include-current-observation-description",
        action="store_true",
        help="Mirrors ExperimentConfig.include_current_observation_description",
    )
    parser.add_argument(
        "--observation-text-includes-facing",
        action="store_true",
        help="Mirrors ExperimentConfig.observation_text_includes_facing",
    )
    parser.add_argument(
        "--action-space",
        dest="action_space",
        choices=["egocentric", "cardinal"],
        default=ExperimentConfig.__dataclass_fields__["action_space"].default,
        help="Mirrors ExperimentConfig.action_space; cardinal remaps arrow keys to absolute N/S/E/W",
    )
    args = parser.parse_args()

    if args.manifest and args.tasks_dir:
        parser.error("--manifest and --tasks-dir are mutually exclusive.")

    config = ExperimentConfig(
        observation=args.observation,
        context_window=args.context_window,
        include_current_observation_description=args.include_current_observation_description,
        observation_text_includes_facing=args.observation_text_includes_facing,
        action_space=args.action_space,
    )

    player = MiniGridPlayer(
        task_path=args.task_file,
        record=args.record,
        config=config,
        tasks_dir=args.tasks_dir,
        manifest=args.manifest,
        experiment=args.experiment,
    )
    player.run()


if __name__ == "__main__":
    main()
