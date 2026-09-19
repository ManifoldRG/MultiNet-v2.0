"""
Custom MiniGrid Environment

A configurable MiniGrid environment that can be populated from TaskSpecification.
Supports all mechanism types: keys, doors, switches, gates, blocks, hazards.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Any

from .bootstrap import disable_gymnasium_env_plugins

disable_gymnasium_env_plugins()

# Import from gymnasium's minigrid package (no naming conflict after rename to gridworld/)
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import WorldObj, Key, Door, Goal, Wall, Lava, Box, Ball
from minigrid.utils.rendering import fill_coords, point_in_circle
from minigrid.minigrid_env import MiniGridEnv

from .task_spec import TaskSpecification, Position
from .world_model import PlannerState, TaskPlanningContext, apply, successors


# Color mapping for MiniGrid
MINIGRID_COLORS = {
    "red": "red",
    "blue": "blue",
    "green": "green",
    "yellow": "yellow",
    "purple": "purple",
    "grey": "grey",
    "gray": "grey",
    "cyan": "blue",
}

SWITCH_RENDER_COLORS = {
    "red": np.array([255, 0, 0]),
    "green": np.array([0, 255, 0]),
    "blue": np.array([0, 0, 255]),
    "purple": np.array([112, 39, 195]),
    "yellow": np.array([255, 255, 0]),
    "grey": np.array([100, 100, 100]),
    "gray": np.array([100, 100, 100]),
    "white": np.array([255, 255, 255]),
    "cyan": np.array([0, 220, 220]),
}


class Switch(Ball):
    """
    Switch object that can control gates.
    Rendered as a ball with special interaction behavior.
    """

    def __init__(
        self,
        color: str = "yellow",
        switch_id: str = "",
        controls: list[str] = None,
        switch_type: str = "toggle",
        initial_state: str = "off",
    ):
        self.visual_color = color
        super().__init__(MINIGRID_COLORS.get(color, "grey"))
        self.switch_id = switch_id
        self.controls = controls or []
        self.switch_type = switch_type
        self.is_active = initial_state == "on"
        self.used = self.is_active and switch_type == "one_shot"

    def can_pickup(self):
        return False

    def can_overlap(self):
        return True

    def activate(self):
        """Apply switch-type-specific activation semantics."""
        if self.switch_type == "one_shot":
            if self.used:
                return False
            self.used = True
            self.is_active = True
            return True
        if self.switch_type == "hold":
            if not self.is_active:
                self.is_active = True
                return True
            return False
        self.is_active = not self.is_active
        return True

    def deactivate(self):
        """Deactivate hold-type switches when the agent leaves the tile."""
        if self.switch_type == "hold" and self.is_active:
            self.is_active = False
            return True
        return False

    def render(self, img):
        color = SWITCH_RENDER_COLORS.get(self.visual_color, SWITCH_RENDER_COLORS["grey"])
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), color)
        # Draw the on/off state so image_only observations can perceive it: an
        # active switch gets a bright white core, an inactive one a hollow
        # (dark) core. Without this branch on/off rendered pixel-identical
        # (R1 render investigation, 2026-07-22).
        core = np.array([255, 255, 255]) if self.is_active else np.array([30, 30, 30])
        fill_coords(img, point_in_circle(0.5, 0.5, 0.13), core)

    def encode(self):
        obj_type, color_idx, state = super().encode()
        # Low bit preserves the custom (non-MiniGrid) color flag; the +2 makes
        # the encoded tuple differ on activation so the MiniGrid tile cache
        # (keyed on encode()) refreshes when is_active flips.
        state = (1 if self.visual_color not in MINIGRID_COLORS else 0) + (2 if self.is_active else 0)
        return (obj_type, color_idx, state)


class GroundKey(Key):
    """Key that can be entered before same-cell PICKUP."""

    def can_overlap(self):
        return True


class Gate(Door):
    """
    Gate object controlled by switches.
    When closed, blocks movement like a wall. When open, passable.
    Extends Door for proper rendering.
    """

    def __init__(self, color: str = "grey", gate_id: str = "", is_open: bool = False):
        self.visual_color = color
        # Initialize as unlocked door
        super().__init__(MINIGRID_COLORS.get(color, "grey"), is_locked=False)
        self.gate_id = gate_id
        self.is_open = is_open

    def can_overlap(self):
        return self.is_open

    def see_behind(self):
        return self.is_open

    def toggle(self, env, pos):
        # Gates can only be toggled by switches, not directly
        return False


class TeleporterObj(Ball):
    """Portal endpoint. Stepping on it lands the agent on the partner cell."""

    def __init__(self, color: str = "purple", teleporter_id: str = "",
                 partner: "TeleporterObj | None" = None, cooldown_max: int = 1):
        self.visual_color = color
        super().__init__(MINIGRID_COLORS.get(color, "purple"))
        self.teleporter_id = teleporter_id
        self.partner: TeleporterObj | None = partner
        self.cooldown = 0
        self.cooldown_max = cooldown_max

    def can_overlap(self):
        return True

    def can_pickup(self):
        return False

    def render(self, img):
        color = SWITCH_RENDER_COLORS.get(self.visual_color, SWITCH_RENDER_COLORS["purple"])
        fill_coords(img, point_in_circle(0.5, 0.5, 0.42), color)
        fill_coords(img, point_in_circle(0.5, 0.5, 0.26), np.array([20, 20, 35]))
        fill_coords(img, point_in_circle(0.5, 0.5, 0.12), color)

    def encode(self):
        obj_type, color_idx, _state = super().encode()
        return (obj_type, color_idx, abs(hash(self.visual_color)) % 6)


class PushableBlock(Box):
    """
    A block that can be pushed by the agent.
    Extends Box to leverage existing rendering.
    """

    def __init__(self, color: str = "grey", block_id: str = ""):
        super().__init__(color)
        self.block_id = block_id
        self.pushable = True

    def can_pickup(self):
        return False


class CustomMiniGridEnv(MiniGridEnv):
    """
    Custom MiniGrid environment that can be configured from a TaskSpecification.

    This environment supports:
    - Arbitrary maze layouts
    - Keys and colored doors
    - Switches and gates
    - Pushable blocks
    - Hazards (lava)
    - Custom goal conditions
    """

    def __init__(
        self,
        width: int = 8,
        height: int = 8,
        max_steps: int = 100,
        agent_start_pos: Optional[tuple[int, int]] = None,
        agent_start_dir: int = 0,
        goal_pos: Optional[tuple[int, int]] = None,
        mission_text: str = "Navigate to the goal",
        render_mode: Optional[str] = None,
        task_spec: Optional[TaskSpecification] = None,
        see_through_walls: bool = True,
        agent_view_size: int = 7,
        highlight: bool = True,
        agent_pov: bool = False,
        drop_available: bool = False,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.goal_pos = goal_pos
        self._custom_mission_text = mission_text  # Store our custom mission text
        self.task_spec = task_spec
        self.drop_available = drop_available
        self._planning_ctx: TaskPlanningContext | None = None

        # Mechanism tracking
        self.key_objects: dict[str, Key] = {}
        self.collected_keys: set[str] = set()
        self.switches: dict[str, Switch] = {}
        self.gates: dict[str, Gate] = {}
        self.blocks: dict[str, PushableBlock] = {}
        self.teleporters: dict[str, TeleporterObj] = {}
        self.switch_gate_map: dict[str, list[str]] = {}  # switch_id -> [gate_ids]
        self.gate_initial_state: dict[str, bool] = {}

        # Fog of war tracking: set of (x, y) cells the agent has visited/seen
        self.explored_cells: set[tuple[int, int]] = set()

        # Mission space for the environment - the func returns our custom text
        mission_space = MissionSpace(mission_func=lambda: mission_text)

        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            max_steps=max_steps,
            see_through_walls=see_through_walls,
            agent_view_size=agent_view_size,
            highlight=highlight,
            agent_pov=agent_pov,
            render_mode=render_mode,
            **kwargs,
        )

        # After super().__init__, self.mission is set by the parent class
        # We can update it to our custom text if needed
        self.mission = mission_text

    def _gen_grid(self, width: int, height: int):
        """Generate the grid. Called by reset()."""
        # Create empty grid
        self.grid = Grid(width, height)

        # Add border walls
        self.grid.wall_rect(0, 0, width, height)

        # Reset fog-of-war tracking
        self.explored_cells = set()
        self.key_objects.clear()
        self.collected_keys.clear()

        # If we have a task spec, it will be populated after _gen_grid by the parser
        # For now, set basic start/goal if provided

        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            # Default: place agent at (1, 1)
            self.agent_pos = (1, 1)
            self.agent_dir = 0

        if self.goal_pos is not None:
            self.put_obj(Goal(), self.goal_pos[0], self.goal_pos[1])

    def place_wall(self, x: int, y: int):
        """Place a wall at the given position."""
        self.grid.set(x, y, Wall())

    def place_key(self, x: int, y: int, color: str, key_id: str | None = None):
        """Place a key at the given position."""
        color = MINIGRID_COLORS.get(color, color)
        key = GroundKey(color)
        if key_id is not None:
            key.key_id = key_id
            self.key_objects[key_id] = key
        self.put_obj(key, x, y)

    def place_door(self, x: int, y: int, color: str, is_locked: bool = True):
        """Place a door at the given position."""
        color = MINIGRID_COLORS.get(color, color)
        door = Door(color, is_locked=is_locked)
        self.grid.set(x, y, door)

    def place_switch(
        self,
        x: int,
        y: int,
        switch_id: str,
        controls: list[str],
        switch_type: str = "toggle",
        initial_state: str = "off",
        color: str = "yellow",
    ):
        """Place a switch at the given position."""
        switch = Switch(
            color=color,
            switch_id=switch_id,
            controls=controls,
            switch_type=switch_type,
            initial_state=initial_state,
        )
        self.switches[switch_id] = switch
        self.switch_gate_map[switch_id] = controls
        self.put_obj(switch, x, y)
        self._refresh_gates()

    def place_gate(self, x: int, y: int, gate_id: str, is_open: bool = False, color: str = "grey"):
        """Place a gate at the given position."""
        gate = Gate(color=color, gate_id=gate_id, is_open=is_open)
        self.gates[gate_id] = gate
        self.gate_initial_state[gate_id] = is_open
        self.grid.set(x, y, gate)

    def place_block(self, x: int, y: int, block_id: str, color: str = "grey"):
        """Place a pushable block at the given position."""
        block = PushableBlock(color=color, block_id=block_id)
        self.blocks[block_id] = block
        self.put_obj(block, x, y)

    def place_hazard(self, x: int, y: int, hazard_type: str = "lava"):
        """Place a hazard at the given position."""
        # All hazards use Lava for now
        self.grid.set(x, y, Lava())

    def place_teleporter(self, teleporter_id: str, x_a: int, y_a: int,
                         x_b: int, y_b: int, bidirectional: bool = True,
                         color: str = "purple"):
        """Place a teleporter pair at the given positions."""
        tp_a = TeleporterObj(color=color, teleporter_id=f"{teleporter_id}_a")
        tp_b = TeleporterObj(color=color, teleporter_id=f"{teleporter_id}_b")
        tp_a.partner = tp_b
        if bidirectional:
            tp_b.partner = tp_a
        self.teleporters[f"{teleporter_id}_a"] = tp_a
        self.teleporters[f"{teleporter_id}_b"] = tp_b
        self.put_obj(tp_a, x_a, y_a)
        self.put_obj(tp_b, x_b, y_b)

    def place_goal(self, x: int, y: int):
        """Place the goal at the given position."""
        self.put_obj(Goal(), x, y)

    def set_agent_position(self, x: int, y: int, direction: int = 0):
        """Set the agent's starting position and direction."""
        self.agent_pos = (x, y)
        self.agent_dir = direction

    def toggle_gate(self, gate_id: str):
        """Toggle a gate's open/closed state."""
        if gate_id in self.gates:
            gate = self.gates[gate_id]
            gate.is_open = not gate.is_open

    def _refresh_gates(self):
        """Recompute gate states from initial configuration and switch activity."""
        for gate_id, gate in self.gates.items():
            is_open = self.gate_initial_state.get(gate_id, False)
            for switch_id, controls in self.switch_gate_map.items():
                switch = self.switches.get(switch_id)
                if switch is not None and gate_id in controls and switch.is_active:
                    is_open = True
            gate.is_open = is_open

    def _update_hold_switches(self):
        """Keep hold-type switches active only while the agent stands on them."""
        changed = False
        for x in range(self.width):
            for y in range(self.height):
                cell = self.grid.get(x, y)
                if isinstance(cell, Switch) and cell.switch_type == "hold":
                    if (x, y) == self.agent_pos:
                        changed = cell.activate() or changed
                    else:
                        changed = cell.deactivate() or changed
        if changed:
            self._refresh_gates()

    def _key_is_collected(self, key_id: str) -> bool:
        """Return whether a tracked key is no longer on the grid."""
        if key_id in self.collected_keys:
            return True
        key = self.key_objects.get(key_id)
        if key is None:
            return False
        for x in range(self.width):
            for y in range(self.height):
                if self.grid.get(x, y) is key:
                    return False
        return True

    def _block_position(self, block_id: str) -> tuple[int, int] | None:
        """Find the current grid position for a tracked pushable block."""
        block = self.blocks.get(block_id)
        if block is None:
            return None
        for x in range(self.width):
            for y in range(self.height):
                if self.grid.get(x, y) is block:
                    return (x, y)
        return None

    def _target_id_completed(self, target_id: str) -> bool:
        if target_id in self.key_objects:
            return self._key_is_collected(target_id)
        if target_id in self.switches:
            return self.switches[target_id].is_active
        if target_id in self.gates:
            return self.gates[target_id].is_open
        if target_id in self.blocks:
            return self._block_position(target_id) is None
        return False

    def _check_goal_completion(self) -> bool:
        """Check all supported task goal types against the current runtime state."""
        if self.task_spec is None or self.task_spec.goal is None:
            if self.goal_pos is not None and self.agent_pos == self.goal_pos:
                return True
            return isinstance(self.grid.get(*self.agent_pos), Goal)

        goal = self.task_spec.goal
        goal_type = goal.goal_type

        if goal_type == "reach_position":
            target = goal.target.to_tuple() if goal.target is not None else self.goal_pos
            return target is not None and self.agent_pos == target

        if goal_type in {"pickup_key", "collect_key"}:
            return bool(goal.target_ids) and all(
                self._key_is_collected(key_id) for key_id in goal.target_ids
            )

        if goal_type == "activate_switch":
            return bool(goal.target_ids) and all(
                switch_id in self.switches and self.switches[switch_id].is_active
                for switch_id in goal.target_ids
            )

        if goal_type == "collect_all":
            return bool(goal.target_ids) and all(
                self._target_id_completed(target_id) for target_id in goal.target_ids
            )

        if goal_type == "push_block_to":
            if len(goal.target_ids) != len(goal.target_positions):
                return False
            return all(
                self._block_position(block_id) == target_pos.to_tuple()
                for block_id, target_pos in zip(goal.target_ids, goal.target_positions)
            )

        if goal_type == "survive_steps":
            return self.step_count >= self.max_steps

        return False

    def _uses_minigrid_goal_tile(self) -> bool:
        return (
            self.task_spec is None
            or self.task_spec.goal is None
            or self.task_spec.goal.goal_type == "reach_position"
        )

    def _cell_can_overlap(self, cell: WorldObj | None) -> bool:
        if cell is None:
            return False
        can_overlap = getattr(cell, "can_overlap", False)
        if isinstance(can_overlap, bool):
            return can_overlap
        return bool(can_overlap())

    def _finalize_step_result(
        self,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ):
        if self._check_goal_completion():
            return max(float(reward), float(self._reward())), True, truncated, info
        if terminated and reward > 0 and not self._uses_minigrid_goal_tile():
            return 0, False, truncated, info
        return reward, terminated, truncated, info

    def _planning_context(self) -> TaskPlanningContext:
        if self.task_spec is None:
            raise RuntimeError("CustomMiniGridEnv.step requires a task_spec")
        if (
            self._planning_ctx is None
            or self._planning_ctx.spec is not self.task_spec
            or self._planning_ctx.drop_available != self.drop_available
        ):
            self._planning_ctx = TaskPlanningContext(
                self.task_spec, drop_available=self.drop_available
            )
        return self._planning_ctx

    def _read_planner_state(self, ctx: TaskPlanningContext) -> PlannerState:
        carrying_key = None
        if self.carrying is not None:
            carrying_key = getattr(self.carrying, "key_id", None)
        open_doors = set()
        for pos, door in ctx.doors_by_pos.items():
            cell = self.grid.get(*pos)
            if isinstance(cell, Door) and not isinstance(cell, Gate):
                if cell.is_open or not cell.is_locked:
                    open_doors.add(door["id"])
        key_positions = []
        for x in range(self.width):
            for y in range(self.height):
                cell = self.grid.get(x, y)
                if isinstance(cell, Key):
                    key_id = getattr(cell, "key_id", None)
                    if key_id is not None:
                        key_positions.append((key_id, x, y))
        return PlannerState(
            agent_pos=tuple(self.agent_pos),
            agent_dir=int(self.agent_dir),
            carrying_key=carrying_key,
            collected_keys=frozenset(self.collected_keys),
            active_switches=frozenset(
                sid for sid, sw in self.switches.items() if sw.is_active
            ),
            used_switches=frozenset(
                sid for sid, sw in self.switches.items() if getattr(sw, "used", False)
            ),
            open_gates=frozenset(gid for gid, gate in self.gates.items() if gate.is_open),
            open_doors=frozenset(open_doors),
            key_positions=frozenset(key_positions),
        )

    def _write_planner_state(self, ctx: TaskPlanningContext, state: PlannerState) -> None:
        self.agent_pos = (int(state.agent_pos[0]), int(state.agent_pos[1]))
        self.agent_dir = int(state.agent_dir)
        self.collected_keys = set(state.collected_keys)

        for sid, switch in self.switches.items():
            switch.is_active = sid in state.active_switches
            switch.used = sid in state.used_switches
        for gid, gate in self.gates.items():
            gate.is_open = gid in state.open_gates
        for pos, door in ctx.doors_by_pos.items():
            cell = self.grid.get(*pos)
            if isinstance(cell, Door) and not isinstance(cell, Gate):
                is_open = door["id"] in state.open_doors
                cell.is_open = is_open
                cell.is_locked = not is_open

        for x in range(self.width):
            for y in range(self.height):
                cell = self.grid.get(x, y)
                if isinstance(cell, Key):
                    self.grid.set(x, y, None)

        self.carrying = None
        for key_id, x, y in state.key_positions:
            obj = self.key_objects.get(key_id)
            if obj is None:
                continue
            self.grid.set(x, y, obj)
            if hasattr(obj, "cur_pos"):
                obj.cur_pos = (x, y)
        if state.carrying_key is not None:
            self.carrying = self.key_objects.get(state.carrying_key)

    def step(self, action: int):
        """Advance the live env by applying the R1 rulebook, then observing."""
        action = int(action)
        if self.task_spec is None:
            return super().step(action)

        ctx = self._planning_context()
        before = self._read_planner_state(ctx)
        after = apply(ctx, before, action)
        legal = any(t.action == action for t in successors(ctx, before))
        self._write_planner_state(ctx, after)

        info: dict = {}
        if not legal and action != int(self.actions.done):
            # Turns are always legal. Everything else that apply treats as a
            # no-op is an invalid action (bump wall, empty pickup, gated DROP).
            if action not in (
                int(self.actions.left),
                int(self.actions.right),
            ):
                info["invalid_action"] = True

        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        obs = self.gen_obs()
        reward, terminated, truncated, info = self._finalize_step_result(
            0, False, truncated, info
        )
        return obs, reward, terminated, truncated, info

    def get_mission_text(self) -> str:
        """Return the mission text."""
        return self._custom_mission_text

    def get_visible_cells(self) -> set[tuple[int, int]]:
        """Get the set of (x, y) cells currently visible to the agent via view cone.

        Uses the same coordinate mapping as MiniGrid's get_frame highlight logic:
        the vis_mask from gen_obs_grid is in rotated agent-relative space, and we
        map back to absolute grid coordinates using dir_vec / right_vec.
        """
        _, vis_mask = self.gen_obs_grid()
        visible = set()

        # MiniGrid coordinate mapping: agent is at bottom-center of rotated view
        f_vec = self.dir_vec
        r_vec = np.array((-f_vec[1], f_vec[0]))
        top_left = (
            np.array(self.agent_pos)
            + f_vec * (self.agent_view_size - 1)
            - r_vec * (self.agent_view_size // 2)
        )

        for vis_i in range(self.agent_view_size):
            for vis_j in range(self.agent_view_size):
                if not vis_mask[vis_i, vis_j]:
                    continue
                abs_pos = top_left - (f_vec * vis_j) + (r_vec * vis_i)
                abs_x, abs_y = int(abs_pos[0]), int(abs_pos[1])
                if 0 <= abs_x < self.width and 0 <= abs_y < self.height:
                    visible.add((abs_x, abs_y))
        return visible

    def update_explored(self):
        """Update fog-of-war: add currently visible cells to explored set."""
        self.explored_cells |= self.get_visible_cells()
