"""Task/environment session for the interactive MiniGrid player.

Everything here mirrors the *exact* observation/step behavior the
model-facing interface (``interface/``) would produce for a given
``ExperimentConfig`` -- task discovery, stepping, transcript recording, the
progress checklist, and the model-view text -- with **no pygame dependency**.
That means it can be constructed and driven headlessly (e.g. from tests or a
future non-pygame front end); ``demo.ui.MiniGridPlayerUI`` is the only thing
that knows this is being shown in a window.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import NamedTuple, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
_repo_root_str = str(_REPO_ROOT)
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)

from gridworld.task_spec import TaskSpecification
from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.backends.base import GridState
from gridworld.actions import MiniGridActions

from demo.compare import R1ResultCatalog, r1_task_id
from interface.config import ExperimentConfig
from interface.actions_map import nlu_action_to_int
from interface.coords import agent_facing, agent_row_col
from interface.episode_log import state_snapshot
from interface.feedback import format_step_feedback
from interface.observation import (
    current_observation_text,
    history_text,
    text_summary_history,
)
from interface import action_space as action_space_mod
from interface.progress_watchdog import ProgressStallWatchdog
from interface.renderer import render_initial_maze_text

from scripts.run_pipeline import (
    _resolve_source,
    load_manifest,
    resolve_task_rows,
)

# Shown in the Task panel and end-screen summary (human-facing, not the model system prompt).
TASK_INSTRUCTION = "Solve the maze by reaching the goal."


# Power decay on ``optimal/steps``, then scaled linearly to 0 at the step cap.
# Exponent 1.3: ~2x optimal ≈ 40% when cap is large (softer than pure high powers).
_EFFICIENCY_DECAY_EXPONENT = 1.3


def efficiency_score(steps: int, optimal: int, cap: int) -> float:
    """Human efficiency score in [0, 1]: 1 at ``optimal`` or better, 0 at ``cap``+."""
    if steps >= cap:
        return 0.0
    if steps <= optimal:
        return 1.0
    ratio_score = (optimal / steps) ** _EFFICIENCY_DECAY_EXPONENT
    cap_scale = 1.0 - (steps - optimal) / (cap - optimal)
    return ratio_score * cap_scale


class ProgressEvent(NamedTuple):
    """One PROGRESS log entry, split into (prefix, object_phrase, suffix) so
    the UI can color/icon just the mechanism mention -- e.g. for "Opened the
    red door", ``prefix="Opened the "``, ``object_phrase="red door"``,
    ``suffix=""`` -- rather than tinting the whole sentence. ``color`` is a
    MiniGrid color name (or None for entries with no single mechanism, e.g.
    a generic ``collect_all`` item); ``icon`` selects which glyph the UI
    draws next to the entry (None falls back to a plain bullet)."""

    prefix: str
    object_phrase: str
    suffix: str
    color: Optional[str]
    icon: Optional[str]  # "key" | "door" | "switch" | "gate" | "block" | "goal" | None


# Settings that can be toggled live via the UI's settings overlay (Tab).
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
# Session
# ---------------------------------------------------------------------------

class MiniGridPlaySession:
    """Task/environment state and stepping logic -- no rendering, no pygame.

    Owns everything a front end needs to *know* (task spec, live GridState,
    transcript, progress checklist, model-view text) and everything it needs
    to *do* (load/switch tasks, dispatch an action token, reset, cycle a
    setting, save a recorded trajectory). ``demo.ui.MiniGridPlayerUI`` reads
    this and calls into it; it never reaches back into the UI.
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
        self.base_dir = _REPO_ROOT
        self.record = record
        self.config = config or ExperimentConfig()
        self.tasks_dir_override: Optional[Path] = (
            self._resolve_path(tasks_dir) if tasks_dir else None
        )

        self.task_path: Optional[Path] = None
        self.task_spec: Optional[TaskSpecification] = None
        self.task_list: list[Path] = []
        self.task_index: int = 0
        # When True, [ / ] keeps this curated list instead of rediscovering
        # whatever else sits in the task's directory (used for R1-only demos).
        self.task_list_locked: bool = False

        # Manifest mode: [ / ] steps through a curated task catalog (rows can
        # point at files scattered across many folders) instead of one flat
        # directory. self.manifest_row_by_path supplies the metadata shown in
        # the info panel; self.task_list holds the resolved paths in manifest
        # order and is left alone by _load_task's directory rediscovery.
        self.manifest_mode = manifest is not None
        self.manifest_experiment = experiment
        self.manifest_row_by_path: dict[Path, dict] = {}
        self._r1_catalog = R1ResultCatalog()
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
            task_path = "ogbench/ogbench/procgen/maze_jsons/D1/10x10_dense_wrong_ky_kr_sg_kb_0.json"

        # Backend for environment logic
        self.backend = MiniGridBackend(render_mode="rgb_array")

        # Episode state
        self.state: Optional[GridState] = None
        self.episode_done = False
        self.episode_success = False
        self.end_reason: str = ""
        self._stall: ProgressStallWatchdog | None = None
        self.total_reward: float = 0.0
        # BFS optimum + R1 step cap from pipeline canonical_paths.
        self.optimal_steps: int = 0
        self.last_action_name: str = ""
        self.last_dispatched_token: str = ""
        self.step_index: int = 0

        # Live PROGRESS log: a running history of milestones the player has
        # actually achieved this episode (see _record_events), rebuilt fresh
        # each reset. Deliberately *not* a checklist of remaining objectives
        # -- pre-listing "collect 2 keys / open 4 doors" would hint at the
        # solution before the player has seen any of it. See ProgressEvent
        # for why each entry is split into prefix/object/suffix + color/icon.
        self.event_log: list[ProgressEvent] = []

        # Model-parity transcript: enriched step records built the same way
        # interface/runner.py builds them, so interface/observation.py's
        # history/text_summary helpers work unmodified on it.
        self.transcript: list[dict] = []

        # Load the initial task
        self._load_task(task_path)

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
        raw_spec = TaskSpecification.from_json(str(resolved))
        manifest_row = self.manifest_row_by_path.get(resolved)
        task_id = manifest_row["task_id"] if manifest_row else r1_task_id(resolved)
        try:
            self.optimal_steps = self._r1_catalog.lookup(task_id).optimal_steps
        except KeyError:
            # Not an R1 task, or no results table is available at all --
            # R1-comparison is simply off for this task; keep the maze's own
            # max_steps rather than crashing the whole load.
            self.optimal_steps = 0
            self.task_spec = raw_spec
        else:
            cap = max(1, self.optimal_steps * 3)
            self.task_spec = (
                raw_spec if raw_spec.max_steps <= cap else dataclasses.replace(raw_spec, max_steps=cap)
            )

        if self.task_list_locked:
            if resolved not in self.task_list:
                raise ValueError(f"{resolved} is not in the locked task list")
        elif self.manifest_mode:
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

    @property
    def display_reward(self) -> float:
        """Human-facing efficiency score for the end screen (0..1)."""
        if not self.episode_done:
            return 0.0
        if self.end_reason == "stalled":
            return 0.0
        return efficiency_score(
            self.state.step_count,
            self.optimal_steps,
            self.state.max_steps,
        )

    def _reset_env(self) -> None:
        """Reset the environment from the current task spec."""
        if self.task_spec is None:
            return

        self.backend.configure(self.task_spec)
        _obs, self.state, _info = self.backend.reset(seed=self.task_spec.seed)

        self.episode_done = False
        self.episode_success = False
        self.end_reason = ""
        k = self.config.progress_stall_k
        self._stall = ProgressStallWatchdog(k, self.state) if k else None
        self.total_reward = 0.0
        self.last_action_name = ""
        self.last_dispatched_token = ""
        self.step_index = 0
        self.event_log = []
        self.transcript = [
            {
                "kind": "reset",
                "state": state_snapshot(self.state),
            }
        ]

    def _load_adjacent_task(self, delta: int) -> None:
        """Load the next (+1) or previous (-1) task in the current directory."""
        if not self.task_list:
            return
        self.task_index = (self.task_index + delta) % len(self.task_list)
        self._load_task(str(self.task_list[self.task_index]))

    # ------------------------------------------------------------------
    # Step execution
    # ------------------------------------------------------------------

    def _dispatch_token(self, token: str) -> None:
        """Execute a token, expanding cardinal moves into primitives (as the
        runner does) so step economy matches the model exactly."""
        self.last_dispatched_token = token
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

    def _after_step(self, event_type: str, terminated: bool, truncated: bool) -> None:
        if event_type == "DONE":
            self.episode_done = True
            self.episode_success = self.state.goal_reached
            self.end_reason = "success"
        elif self._stall and not (terminated or truncated) and self._stall.observe(self.state):
            self.episode_done = True
            self.end_reason = "stalled"
        elif terminated or truncated:
            self.episode_done = True
            self.episode_success = self.state.goal_reached
            self.end_reason = "truncated" if truncated else "success"

    def _step_token(self, token: str, cardinal_source: Optional[str] = None) -> None:
        """Execute a single egocentric primitive action."""
        if self.episode_done or self.state is None:
            return

        prev_state = self.state
        prev_doors = self._physical_door_states()
        action_int = nlu_action_to_int(token)
        _rgb, reward, terminated, truncated, self.state, info = self.backend.step(action_int)
        self.total_reward += reward
        self.last_action_name = token
        self._record_events(prev_state, self.state, prev_doors)

        feedback_text, event_type = format_step_feedback(
            token, prev_state, self.state, reward, terminated, self.task_spec
        )
        self._record_step(
            token, cardinal_source, prev_state, feedback_text, event_type,
            reward, terminated, truncated, info,
        )

        self._after_step(event_type, terminated, truncated)

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
            color = prev_state.agent_carrying
            self.event_log.append(
                ProgressEvent("Dropped the ", f"{color} key", "", color, "key")
            )
            feedback_text = f"You drop the {color}. (human-only action)"
        else:
            feedback_text = "Nothing to drop. (human-only action)"
        self._record_step(
            "DROP", None, prev_state, feedback_text, "DROPPED",
            reward, terminated, truncated, info,
        )
        self._after_step("DROPPED", terminated, truncated)

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
                "position_before_row_col": list(agent_row_col(prev_state)),
                "position_after_row_col": list(agent_row_col(self.state)),
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

    def task_prompt_text(self) -> str:
        return TASK_INSTRUCTION

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
    # Progress log (a history of what actually happened, not a checklist of
    # what's left -- see event_log)
    # ------------------------------------------------------------------

    def _physical_door_states(self) -> dict[str, bool]:
        """True ``is_open`` per door read directly off the live grid cell.

        ``GridState.open_doors`` (from the backend) counts a door as "open"
        once it's ever been *unlocked*, even if the player has since closed
        it again -- the right notion for goal/scoring purposes, since an
        unlocked door no longer blocks progress. But it's the wrong notion
        for this live progress log: closing a door you just opened should
        show up as closed, not stay stuck saying "opened" forever. So the
        progress log tracks true physical open/closed state separately, by
        reading each door's ``is_open`` straight off the grid."""
        if self.task_spec is None or self.backend.env is None:
            return {}
        result: dict[str, bool] = {}
        for door in self.task_spec.mechanisms.doors:
            cell = self.backend.env.grid.get(door.position.x, door.position.y)
            result[door.id] = bool(getattr(cell, "is_open", False))
        return result

    def _record_events(
        self, prev: GridState, new: GridState, prev_doors: dict[str, bool]
    ) -> None:
        """Append a ``ProgressEvent`` to ``event_log`` for each milestone
        that newly became true on this step (diffing ``prev`` -> ``new``,
        and ``prev_doors`` -> the door states read fresh off the grid right
        now).

        Every entry is grounded in the task's actual ``TaskSpecification``
        mechanisms and the backend's own state flags -- nothing here is
        inferred/guessed. Deliberately keyed off a state *transition* (not
        the current state) so re-triggerable mechanisms like gates log again
        each time they're actually reopened/closed, giving an honest history
        rather than a one-shot checklist."""
        spec = self.task_spec
        if spec is None:
            return
        mech = spec.mechanisms

        for key in mech.keys:
            if key.id in new.collected_keys and key.id not in prev.collected_keys:
                self.event_log.append(
                    ProgressEvent("Picked up the ", f"{key.color} key", "", key.color, "key")
                )

        new_doors = self._physical_door_states()
        for door in mech.doors:
            was_open = prev_doors.get(door.id, False)
            now_open = new_doors.get(door.id, False)
            object_phrase = f"{door.requires_key} door"
            if now_open and not was_open:
                self.event_log.append(
                    ProgressEvent("Opened the ", object_phrase, "", door.requires_key, "door")
                )
            elif was_open and not now_open:
                self.event_log.append(
                    ProgressEvent("Closed the ", object_phrase, "", door.requires_key, "door")
                )

        for switch in mech.switches:
            was_on = switch.id in prev.active_switches
            now_on = switch.id in new.active_switches
            object_phrase = f"{switch.color} switch"
            if now_on and not was_on:
                self.event_log.append(
                    ProgressEvent("Turned on the ", object_phrase, "", switch.color, "switch")
                )
            elif was_on and not now_on:
                self.event_log.append(
                    ProgressEvent("Turned off the ", object_phrase, "", switch.color, "switch")
                )

        for gate in mech.gates:
            was_open = gate.id in prev.open_gates
            now_open = gate.id in new.open_gates
            object_phrase = f"{gate.color} gate"
            if now_open and not was_open:
                self.event_log.append(
                    ProgressEvent("Opened the ", object_phrase, "", gate.color, "gate")
                )
            elif was_open and not now_open:
                self.event_log.append(
                    ProgressEvent("Closed the ", object_phrase, "", gate.color, "gate")
                )

        if mech.blocks and spec.goal.goal_type == "push_block_to":
            blocks_by_id = {block.id: block for block in mech.blocks}
            targets = dict(zip(spec.goal.target_ids, spec.goal.target_positions))
            for block_id, pos in targets.items():
                was_placed = prev.block_positions.get(block_id) == pos.to_tuple()
                now_placed = new.block_positions.get(block_id) == pos.to_tuple()
                if now_placed and not was_placed:
                    color = blocks_by_id[block_id].color if block_id in blocks_by_id else None
                    object_phrase = f"{color} block" if color else "block"
                    self.event_log.append(
                        ProgressEvent("Pushed the ", object_phrase, " into place", color, "block")
                    )

        if spec.goal.goal_type == "collect_all" and not mech.keys:
            newly_collected = set(new.collected_keys) - set(prev.collected_keys)
            for item_id in newly_collected & set(spec.goal.target_ids):
                self.event_log.append(ProgressEvent("Collected ", item_id, "", None, None))

        if new.goal_reached and not prev.goal_reached:
            self.event_log.append(ProgressEvent("Reached the ", "goal", "!", "green", "goal"))

    # ------------------------------------------------------------------
    # Settings (mutates the shared ExperimentConfig; UI triggers this)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Recording / trajectory saving
    # ------------------------------------------------------------------

    def trajectory_dict(self) -> dict:
        """Transcript payload shared by desktop ``--record`` and the web API."""
        task_id = self.task_spec.task_id if self.task_spec else "unknown"
        manifest_row = (
            self.manifest_row_by_path.get(self.task_path) if self.manifest_mode else None
        )
        return {
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

        data = self.trajectory_dict()
        task_id = data["task_id"]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"trajectory_{task_id}_{timestamp}.json"
        output_path = self.base_dir / filename

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Trajectory saved to: {output_path}")

    def close(self) -> None:
        self.backend.close()
