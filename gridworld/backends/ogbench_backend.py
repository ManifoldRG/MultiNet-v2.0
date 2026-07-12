"""
OGBench PointyEnv Backend Implementation

Wraps OGBench's continuous-physics ``PointyEnv`` maze (via
``ogbench.locomaze.maze.make_maze_env('pointy', 'maze', ...)``) with the
AbstractGridBackend interface.

Unlike MiniGridBackend/MultiGridBackend, navigation here is genuinely
continuous: MOVE_FORWARD/TURN_LEFT/TURN_RIGHT accept an optional numeric
magnitude (see ``actions_hint``/``parse_action``) and position/heading are
not snapped to grid cells or cardinal directions between steps. Only the
non-navigation logic (PICKUP/DROP/TOGGLE/DONE, inventory, success) mirrors
MiniGrid's semantics — see ``gridworld/backends/base.py`` for the resulting
``GridState`` typing note (continuous backends populate ``agent_position``/
``agent_direction`` with floats, not ints).
"""

from __future__ import annotations

import math
import re
from typing import Any, Optional

import numpy as np

from ..task_spec import TaskSpecification
from .base import AbstractGridBackend, GridState

_ACTION_RE = re.compile(
    r"^(TURN_LEFT|TURN_RIGHT|MOVE_FORWARD|PICKUP|DROP|TOGGLE|DONE)"
    r"(\(\s*(-?\d+(?:\.\d+)?)\s*\))?$"
)

_DEFAULT_TURN_DEGREES = 90.0
_DEFAULT_FORWARD_UNITS = 1.0


class OgbenchBackend(AbstractGridBackend):
    """
    Backend implementation using OGBench's continuous-physics PointyEnv maze.

    Navigation (MOVE_FORWARD/TURN_LEFT/TURN_RIGHT) is continuous — see the
    module docstring. PICKUP/DROP/TOGGLE/DONE mirror MiniGrid's semantics as
    closely as OGBench's proximity-based interaction model allows.
    """

    def __init__(
        self,
        render_mode: Optional[str] = "rgb_array",
        width: int = 512,
        height: int = 512,
        maze_unit: float = 1.4,
        maze_height: float = 0.5,
        ob_type: str = "states",
    ):
        super().__init__()
        self.render_mode = render_mode
        self.width = width
        self.height = height
        self.maze_unit = maze_unit
        self.maze_height = maze_height
        self.ob_type = ob_type

        self.env = None
        self._parsed = None
        self._start_ij: Optional[tuple[int, int]] = None
        self._has_open_action = False
        self._forward_scale = 0.2
        self._heading_scale = 0.35
        self._step_count = 0
        self._last_rgb: Optional[np.ndarray] = None
        self._last_terminated = False
        self._last_truncated = False
        self._last_reward = 0.0

    # ------------------------------------------------------------------ #
    # AbstractGridBackend interface
    # ------------------------------------------------------------------ #
    def configure(self, task_spec: TaskSpecification) -> None:
        self._validate_supported(task_spec)
        self.task_spec = task_spec

        from ogbench.procgen.maze_json_interface import make_pointymaze_env_from_payload

        self.env, self._parsed, self._start_ij, self._goal_ij = make_pointymaze_env_from_payload(
            task_spec.to_dict(),
            json_origin="top_left",
            maze_unit=self.maze_unit,
            maze_height=self.maze_height,
            render_mode=self.render_mode,
            width=self.width,
            height=self.height,
            ob_type=self.ob_type,
            add_noise_to_goal=False,
            terminate_at_goal=True,
        )
        self._has_open_action = bool(self.env.unwrapped._has_open_action)
        self._forward_scale = float(self.env.unwrapped._forward_scale)
        self._heading_scale = float(self.env.unwrapped._heading_scale)
        self._configured = True

    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, GridState, dict]:
        if not self._configured:
            raise RuntimeError("Backend must be configured before reset")

        self.env.reset(seed=seed, options={"task_id": 1})

        # MazeEnv.reset() unconditionally jitters the initial position
        # (independent of add_noise_to_goal, which only affects the goal) —
        # snap back to the exact start cell center for a deterministic,
        # reproducible starting pose. Heading needs no manual fix: reset()'s
        # second internal super().reset() call always resets the marker
        # heading to exact EAST via PointyEnv.reset_model().
        start_xy = self.env.unwrapped.ij_to_xy(self._start_ij)
        self.env.unwrapped.set_xy(np.array(start_xy, dtype=np.float64))

        self._step_count = 0
        self._last_terminated = False
        self._last_truncated = False
        self._last_reward = 0.0
        self._last_rgb = self.env.render()

        state = self._get_grid_state()
        return self._last_rgb, state, {}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, GridState, dict]:
        if self.env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        self._step_count += 1
        ob, reward, terminated, truncated, info = self.env.step(action)
        truncated = bool(truncated) or self._step_count >= self.task_spec.max_steps

        self._last_terminated = bool(terminated)
        self._last_truncated = bool(truncated)
        self._last_reward = float(reward)
        self._last_rgb = self.env.render()

        state = self._get_grid_state()
        return self._last_rgb, reward, terminated, truncated, state, info

    def render(self) -> np.ndarray:
        if self.env is not None:
            return self.env.render()
        return np.zeros((64, 64, 3), dtype=np.uint8)

    def get_mission_text(self) -> str:
        if self.task_spec is not None:
            return self.task_spec.get_mission_text()
        return "Navigate to the goal"

    def get_state(self) -> GridState:
        return self._get_grid_state()

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    # ------------------------------------------------------------------ #
    # Action vocabulary
    # ------------------------------------------------------------------ #
    def actions_hint(self) -> str:
        return (
            "TURN_LEFT, TURN_RIGHT, MOVE_FORWARD, PICKUP, DROP, TOGGLE, DONE. "
            "Navigation is continuous: MOVE_FORWARD and TURN_LEFT/TURN_RIGHT "
            "accept an optional numeric magnitude in parentheses. "
            "MOVE_FORWARD(0.4) moves 0.4 of a full forward step (omit for a "
            "full step, i.e. MOVE_FORWARD == MOVE_FORWARD(1.0)); "
            "TURN_LEFT(30)/TURN_RIGHT(30) rotates by 30 degrees (omit for a "
            "default 90-degree turn). PICKUP, DROP, TOGGLE, DONE take no argument."
        )

    def parse_action(self, token: str) -> Any:
        match = _ACTION_RE.match(token.strip())
        if not match:
            raise ValueError(f"Unknown or malformed OgbenchBackend action: {token!r}")

        verb = match.group(1)
        magnitude = float(match.group(3)) if match.group(3) is not None else None

        fwd_units = 0.0
        heading_deg = 0.0
        interact_cmd = 0

        if verb == "MOVE_FORWARD":
            m = magnitude if magnitude is not None else _DEFAULT_FORWARD_UNITS
            if m < 0:
                raise ValueError(
                    f"MOVE_FORWARD magnitude must be >= 0 (no native backward move), got {m}"
                )
            fwd_units = m
        elif verb in ("TURN_LEFT", "TURN_RIGHT"):
            deg = magnitude if magnitude is not None else _DEFAULT_TURN_DEGREES
            deg = max(0.0, min(180.0, deg))
            heading_deg = deg if verb == "TURN_LEFT" else -deg
        elif verb == "PICKUP":
            interact_cmd = 1
        elif verb == "TOGGLE":
            interact_cmd = 2
        # DROP / DONE: no native ogbench effect, zero move + no-op interact.

        fwd_actuator = fwd_units * (self.maze_unit / self._forward_scale)
        heading_actuator = math.radians(heading_deg) / self._heading_scale
        mv_arr = np.array([fwd_actuator, heading_actuator], dtype=np.float32)

        if self._has_open_action:
            return mv_arr, interact_cmd
        return mv_arr

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _validate_supported(self, task_spec: TaskSpecification) -> None:
        mech = task_spec.mechanisms
        if mech.blocks or mech.teleporters or mech.hazards:
            raise ValueError(
                "OgbenchBackend does not support blocks/teleporters/hazards mechanisms."
            )
        if task_spec.goal.goal_type != "reach_position":
            raise ValueError(
                "OgbenchBackend only supports goal_type='reach_position', got "
                f"{task_spec.goal.goal_type!r}."
            )
        if task_spec.rules.observability != "full":
            raise ValueError(
                "OgbenchBackend only supports full observability "
                "(no view_cone/fog_of_war)."
            )
        if mech.keys and not (mech.doors or mech.switches):
            raise ValueError(
                "OgbenchBackend: mechanisms.keys requires at least one door or "
                "switch — MazeEnv only enables the pickup/use interact command "
                "when doors or switches are present, so a keys-only task would "
                "make PICKUP a permanent no-op."
            )

    def _get_grid_state(self) -> GridState:
        if self.env is None:
            return GridState(agent_position=(0.0, 0.0), agent_direction=0.0)

        env = self.env.unwrapped
        x_env, y_env = np.asarray(env.get_xy(), dtype=np.float64).flat[:2]

        # Continuous inverse of ij_to_xy (no floor to an integer cell).
        i_cell = (y_env + env._offset_y) / env._maze_unit
        j_cell = (x_env + env._offset_x) / env._maze_unit
        height = self.task_spec.maze.dimensions[1]
        row = height - 1 - i_cell
        col = j_cell

        hx, hy = np.asarray(env._marker_heading, dtype=np.float64)
        heading_deg = math.degrees(math.atan2(hy, hx))

        colors = sorted(env._inventory_key_colors)
        if not colors:
            agent_carrying = None
        elif len(colors) == 1:
            agent_carrying = colors[0]
        else:
            agent_carrying = ",".join(colors)

        open_doors = {item["id"] for item in env._door_items if item.get("opened")}
        collected_keys = {item["id"] for item in env._key_items if item.get("collected")}
        active_switches = {item["id"] for item in env._switch_items if item.get("is_on")}
        open_gates = {item["id"] for item in env._gate_items if item.get("opened")}

        return GridState(
            agent_position=(float(col), float(row)),
            agent_direction=float(heading_deg),
            agent_carrying=agent_carrying,
            step_count=self._step_count,
            max_steps=self.task_spec.max_steps,
            terminated=self._last_terminated,
            truncated=self._last_truncated,
            reward=self._last_reward,
            open_doors=open_doors,
            collected_keys=collected_keys,
            active_switches=active_switches,
            open_gates=open_gates,
            block_positions={},
            teleporter_cooldowns={},
            goal_reached=bool(env.compute_success()),
            observability_mode="full",
            visible_cells=set(),
            explored_cells=set(),
        )

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        if self.env is not None:
            img = self.env.render()
            return img.shape
        return (self.height, self.width, 3)
