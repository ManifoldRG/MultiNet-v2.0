# gridworld/backends/multigrid_backend.py

"""
MultiGrid Backend Implementation

Adapter for the custom MultiGrid system (`multigrid/`) that implements
the AbstractGridBackend interface. This allows evaluation of custom tilings
(square, hex, triangle) using the same pipeline as MiniGrid.

Square and hex tasks step through the R1 rulebook (gridworld.world_model),
exactly as CustomMiniGridEnv.step does: read a PlannerState off the multigrid
WorldState, apply the action, write the result back. Square through this
backend is step-for-step identical to MiniGridBackend, and a hex task (spec
``maze.tiling: "hex"``) gets the same mechanics on six directions. Frames come
from gridworld.render2d in the MiniGrid look. Triangle, 3464 and 488 keep the
native multigrid ``execute_action`` semantics (legacy, experimental).

Usage:
    from gridworld.backends import MultiGridBackend

    # Tiling taken from the task spec (square or hex)
    backend = MultiGridBackend(render_mode="rgb_array")
    backend.configure(task_spec)
    obs, state, info = backend.reset(seed=42)
    obs, reward, terminated, truncated, state, info = backend.step(action)
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np

from .base import AbstractGridBackend, GridState
from ..actions import MiniGridActions
from ..task_spec import TaskSpecification
from ..tiling import DIRECTION_NAMES
from ..world_model import PlannerState, TaskPlanningContext, apply, classify_mechanic, successors

# Add parent directory to path for multigrid imports
_multigrid_path = Path(__file__).parent.parent.parent / "multigrid"
if str(_multigrid_path.parent) not in sys.path:
    sys.path.insert(0, str(_multigrid_path.parent))

# Tilings the R1 rulebook covers: their tasks are TaskSpecification-native.
RULEBOOK_TILINGS = ("square", "hex")


def _rulebook_only_mechanisms(spec: TaskSpecification) -> list[str]:
    """PR #57 tiles that only the rulebook gives meaning to."""
    found = []
    if spec.mechanisms.kill_cells:
        found.append("death_portals")
    if spec.mechanisms.frozen_tiles:
        found.append("frozen_tiles")
    if spec.mechanisms.rotating_tiles:
        found.append("rotating_tiles")
    return found


class MultiGridBackend(AbstractGridBackend):
    """
    Backend adapter for the custom MultiGrid system.

    Supports exotic tilings: square, hex, triangle.

    Args:
        tiling: Tiling type ("square", "hex", "triangle", "3464", "488"), or
            None to take the task spec's ``maze.tiling``. A square or hex
            backend only accepts a spec of the same tiling.
        render_mode: Render mode ("rgb_array" or "human")
        render_width: Width of rendered image (default 640, legacy tilings)
        render_height: Height of rendered image (default 640, legacy tilings)
    """

    def __init__(
        self,
        tiling: Optional[str] = None,
        render_mode: str = "rgb_array",
        render_width: int = 640,
        render_height: int = 640,
    ):
        super().__init__()
        self.requested_tiling = tiling
        self.tiling_type = tiling or "square"
        self.render_mode = render_mode
        self.render_width = render_width
        self.render_height = render_height
        # Same default as MiniGridBackend: DROP is a legal action.
        self.drop_available = True

        # Will be initialized on configure()
        self.env = None
        self._step_count = 0
        self._max_steps = 100

        # Rulebook mode (square/hex) only
        self._planning_ctx: Optional[TaskPlanningContext] = None
        self._frame_renderer = None
        self._key_objects: dict = {}
        self._cell_by_xy: dict[tuple[int, int], str] = {}
        self._xy_by_cell: dict[str, tuple[int, int]] = {}
        self._native_facing: tuple[int, ...] = ()

    @property
    def uses_rulebook(self) -> bool:
        """True when steps go through gridworld.world_model (square/hex)."""
        return self._planning_ctx is not None

    def configure(self, task_spec: TaskSpecification) -> None:
        """
        Configure the backend with a task specification.

        Converts the TaskSpecification to the multigrid format and creates
        the environment.

        Args:
            task_spec: The task specification defining the puzzle

        Raises:
            ValueError: If the spec's tiling does not match the backend's, or
                a legacy tiling meets a spec only the rulebook can run
        """
        tiling = self.requested_tiling or task_spec.maze.tiling
        if tiling in RULEBOOK_TILINGS:
            if task_spec.maze.tiling != tiling:
                raise ValueError(
                    f"Task {task_spec.task_id!r} declares a {task_spec.maze.tiling} tiling "
                    f"but this backend draws {tiling}; cells would not connect as the "
                    "planner assumes"
                )
            is_valid, errors = task_spec.validate()
            if not is_valid:
                raise ValueError(f"Invalid task specification: {'; '.join(errors)}")
        else:
            needs_rulebook = _rulebook_only_mechanisms(task_spec)
            if task_spec.maze.tiling != "square" or needs_rulebook:
                raise ValueError(
                    f"The {tiling} tiling runs multigrid's native rules, which have no "
                    f"{task_spec.maze.tiling} cells or {', '.join(needs_rulebook) or 'PR #57 tiles'}; "
                    f"run task {task_spec.task_id!r} on a {list(RULEBOOK_TILINGS)} backend"
                )
        self.tiling_type = tiling
        self.task_spec = task_spec

        # Convert TaskSpecification to multigrid task_spec dict
        multigrid_spec = self._convert_task_spec(task_spec)

        # Extract observability settings from task_spec
        obs_mode = task_spec.rules.observability if task_spec.rules else "full"
        view_size = task_spec.rules.view_size if task_spec.rules else 7
        partial = obs_mode != "full"

        # Import and create MultiGridEnv
        from multigrid.env import MultiGridEnv

        self.env = MultiGridEnv(
            task_spec=multigrid_spec,
            tiling=self.tiling_type,
            render_mode=self.render_mode,
            partial_obs=partial,
            obs_radius=view_size // 2,
            observability_mode=obs_mode,
        )

        if tiling in RULEBOOK_TILINGS:
            from ..render2d import TilingFrameRenderer

            self._planning_ctx = TaskPlanningContext(task_spec, drop_available=self.drop_available)
            self._frame_renderer = TilingFrameRenderer(task_spec)
            names = [name.lower() for name in DIRECTION_NAMES[tiling]]
            self._native_facing = tuple(self.env.tiling.directions.index(name) for name in names)
        else:
            self._planning_ctx = None
            self._frame_renderer = None

        self._max_steps = task_spec.max_steps
        self._configured = True

    def _convert_task_spec(self, spec: TaskSpecification) -> dict:
        """
        Convert TaskSpecification to multigrid task_spec dict format.

        This method bridges the gap between the standard MiniGrid TaskSpecification
        format (used for consistency across backends) and the MultiGrid-specific
        format required by the custom MultiGrid environment.

        This preserves the canonical TaskSpecification semantics by emitting the
        corresponding native MultiGrid object types rather than degrading them.

        Args:
            spec: TaskSpecification from the minigrid module (standard format)

        Returns:
            Dictionary in multigrid format ready for MultiGridEnv initialization

        Limitations:
            - Border cells are represented as explicit wall objects so square-grid
              semantics match the MiniGrid backend.
        """
        width, height = spec.maze.dimensions

        coordinate_tiling = None
        cell_ids_by_grid_pos = {}
        if self.tiling_type in {"square", "hex"}:
            from multigrid.env import TilingRegistry

            coordinate_tiling = TilingRegistry.get(self.tiling_type)
            coordinate_tiling.generate_graph(width, height, spec.seed)
            cell_ids_by_grid_pos = {
                (cell.col, cell.row): cell_id
                for cell_id, cell in coordinate_tiling.cells.items()
            }

        def canonical_pos(x: int, y: int) -> dict:
            if coordinate_tiling is not None and (x, y) in cell_ids_by_grid_pos:
                cell_id = cell_ids_by_grid_pos[(x, y)]
                cx, cy = coordinate_tiling.cell_to_canonical(cell_id)
                return {"x": cx, "y": cy}
            return {
                "x": (x + 0.5) / width,
                "y": (y + 0.5) / height,
            }

        # Build scene objects list
        objects = []

        wall_positions = {(w.x, w.y) for w in spec.maze.walls}
        for x in range(width):
            wall_positions.add((x, 0))
            wall_positions.add((x, height - 1))
        for y in range(height):
            wall_positions.add((0, y))
            wall_positions.add((width - 1, y))

        barrier_positions = {
            (door.position.x, door.position.y)
            for door in spec.mechanisms.doors
        } | {
            (gate.position.x, gate.position.y)
            for gate in spec.mechanisms.gates
        }
        wall_positions -= barrier_positions

        for x, y in sorted(wall_positions):
            objects.append({
                "id": f"wall_{x}_{y}",
                "type": "wall",
                "color": "grey",
                "position": canonical_pos(x, y),
            })

        for key in spec.mechanisms.keys:
            objects.append({
                "id": key.id,
                "type": "key",
                "color": key.color,
                "position": canonical_pos(key.position.x, key.position.y),
            })

        for door in spec.mechanisms.doors:
            objects.append({
                "id": door.id,
                "type": "door",
                "color": door.requires_key,
                "position": canonical_pos(door.position.x, door.position.y),
                "is_locked": door.initial_state == "locked",
            })

        for switch in spec.mechanisms.switches:
            objects.append({
                "id": switch.id,
                "type": "switch",
                "color": switch.color,
                "position": canonical_pos(switch.position.x, switch.position.y),
                "controls": switch.controls,
                "switch_type": switch.switch_type,
                "initial_state": switch.initial_state == "on",
            })

        for gate in spec.mechanisms.gates:
            controlled_by = [
                switch.id for switch in spec.mechanisms.switches if gate.id in switch.controls
            ]
            objects.append({
                "id": gate.id,
                "type": "gate",
                "color": gate.color,
                "position": canonical_pos(gate.position.x, gate.position.y),
                "is_open": gate.initial_state == "open",
                "controlled_by": controlled_by,
            })

        for block in spec.mechanisms.blocks:
            objects.append({
                "id": block.id,
                "type": "movable",
                "color": block.color,
                "position": canonical_pos(block.position.x, block.position.y),
            })

        for hazard in spec.mechanisms.hazards:
            objects.append({
                "id": hazard.id,
                "type": "hazard",
                "color": "red",
                "position": canonical_pos(hazard.position.x, hazard.position.y),
                "hazard_type": hazard.hazard_type,
            })

        for teleporter in spec.mechanisms.teleporters:
            a_id = f"{teleporter.id}_a"
            b_id = f"{teleporter.id}_b"
            objects.append({
                "id": a_id,
                "type": "teleporter",
                "color": teleporter.color,
                "position": canonical_pos(teleporter.position_a.x, teleporter.position_a.y),
                "linked_to": b_id,
            })
            objects.append({
                "id": b_id,
                "type": "teleporter",
                "color": teleporter.color,
                "position": canonical_pos(teleporter.position_b.x, teleporter.position_b.y),
                "linked_to": a_id if teleporter.bidirectional else None,
            })

        # PR #57 tiles: storage for the rulebook (configure() keeps them off
        # the legacy tilings, whose native rules would ignore them).
        for i, cell in enumerate(spec.mechanisms.kill_cells):
            objects.append({
                "id": f"death_{i}",
                "type": "death",
                "position": canonical_pos(cell.x, cell.y),
            })
        for i, cell in enumerate(spec.mechanisms.frozen_tiles):
            objects.append({
                "id": f"frozen_{i}",
                "type": "frozen",
                "position": canonical_pos(cell.x, cell.y),
            })
        for i, (cell, direction) in enumerate(zip(
            spec.mechanisms.rotating_tiles, spec.mechanisms.rotating_initial_directions
        )):
            objects.append({
                "id": f"rotating_{i}",
                "type": "rotating",
                "position": canonical_pos(cell.x, cell.y),
                "direction": direction,
            })

        goal_spec = {}
        if spec.goal:
            if spec.goal.goal_type == "reach_position":
                goal_target = spec.goal.target or spec.maze.goal
                goal_spec = {
                    "type": "reach_position",
                    "target": canonical_pos(goal_target.x, goal_target.y),
                }
            elif spec.goal.goal_type == "collect_all":
                goal_spec = {
                    "type": "collect_all",
                    "target_ids": spec.goal.target_ids
                }
            elif spec.goal.goal_type == "push_block_to":
                goal_spec = {
                    "type": "push_block_to",
                    "target_ids": spec.goal.target_ids,
                    "target_positions": [
                        canonical_pos(p.x, p.y)
                        for p in spec.goal.target_positions
                    ] if spec.goal.target_positions else []
                }

        # Construct complete MultiGrid task specification
        return {
            "task_id": spec.task_id,
            "seed": spec.seed,
            "tiling": {
                "type": self.tiling_type,  # square, hex, or triangle
                "grid_size": {
                    "width": spec.maze.dimensions[0],
                    "height": spec.maze.dimensions[1]
                }
            },
            "scene": {
                "agent": {
                    "position": canonical_pos(spec.maze.start.x, spec.maze.start.y),
                    # Native index 0 (north on square); reset() turns a
                    # rulebook task east, as MiniGrid starts.
                    "facing": 0
                },
                "objects": objects,
            },
            "goal": goal_spec,
            "rules": {
                "key_consumption": spec.rules.key_consumption,
                "switch_type": spec.rules.switch_type,
            },
            "limits": {
                "max_steps": spec.max_steps
            },
            "metadata": spec.metadata or {},
        }

    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, GridState, dict]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility

        Returns:
            observation: The initial observation (RGB image)
            state: The initial GridState
            info: Additional information dictionary
        """
        if not self._configured or self.env is None:
            raise RuntimeError("Backend must be configured before reset")

        obs, info = self.env.reset(seed=seed)
        self._step_count = 0

        if self.uses_rulebook:
            tiling = self.env.tiling
            self._xy_by_cell = {cell_id: (cell.col, cell.row) for cell_id, cell in tiling.cells.items()}
            self._cell_by_xy = {xy: cell_id for cell_id, xy in self._xy_by_cell.items()}
            self._key_objects = {
                obj.id: obj for obj in self.env.state.objects.values() if obj.obj_type == "key"
            }
            # The rulebook's start state is authoritative (facing east, gates
            # opened by initially-on switches, one-shot switches spent), so
            # visibility starts over from it.
            self._write_planner_state(self._planning_ctx.initial_state())
            self.env.state.explored_cells = set()
            self.env.state.update_visibility()
            obs = self.render()
            info = {}  # as MiniGridBackend.reset under full observability

        state = self._build_grid_state()

        return obs, state, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, GridState, dict]:
        """
        Execute one action in the environment.

        Square and hex tasks apply the R1 rulebook (see ``_rulebook_step``).
        The legacy tilings translate MiniGrid actions onto multigrid's native
        action enumeration:

        MiniGrid Action → MultiGrid Action
        0: turn_left     → 2: TURN_LEFT
        1: turn_right    → 3: TURN_RIGHT
        2: forward       → 0: FORWARD
        3: pickup        → 4: PICKUP
        4: drop          → 5: DROP
        5: toggle        → 6: TOGGLE
        6: done/wait     → 8: WAIT

        Args:
            action: The action to execute (0-6, standard MiniGrid action space)

        Returns:
            observation: RGB image of the new state
            reward: Reward for this step
            terminated: Whether the episode ended (goal reached or failure)
            truncated: Whether the episode was cut short (max steps reached)
            state: GridState representing the new environment state
            info: Additional information dictionary from the environment

        Raises:
            RuntimeError: If the backend has not been configured or reset
        """
        if not self._configured or self.env is None:
            raise RuntimeError("Backend must be configured before step")

        if self.uses_rulebook:
            return self._rulebook_step(action)

        # Map MiniGrid action to MultiGrid action
        # This translation ensures compatibility between backends
        action_map = {
            0: 2,  # turn_left -> TURN_LEFT
            1: 3,  # turn_right -> TURN_RIGHT
            2: 0,  # forward -> FORWARD
            3: 4,  # pickup -> PICKUP
            4: 5,  # drop -> DROP
            5: 6,  # toggle -> TOGGLE
            6: 8,  # done -> WAIT
        }

        # Get MultiGrid action index, default to WAIT if action invalid
        multigrid_action = action_map.get(action, 8)

        # Execute action in MultiGrid environment
        obs, reward, terminated, truncated, info = self.env.step(multigrid_action)

        # Track step count (MultiGrid doesn't track this internally)
        self._step_count += 1

        # Build GridState for backend-agnostic representation
        state = self._build_grid_state()
        # Update state with step results
        state.terminated = terminated
        state.truncated = truncated
        state.reward = reward
        state.step_count = self._step_count

        return obs, reward, terminated, truncated, state, info

    def _rulebook_step(self, action: int) -> tuple[np.ndarray, float, bool, bool, GridState, dict]:
        """CustomMiniGridEnv.step on the multigrid WorldState: same rulebook,
        same invalid-action and mechanic info, same reward and termination."""
        action = int(action)
        ctx = self._planning_ctx
        before = self._read_planner_state()
        after = apply(ctx, before, action)
        transition = next((t for t in successors(ctx, before) if t.action == action), None)
        self._write_planner_state(after)

        info: dict = {}
        if transition is None and action != int(MiniGridActions.DONE):
            # Turns are always legal. Everything else that apply treats as a
            # no-op is an invalid action (bump wall, empty pickup, gated DROP).
            if action not in (int(MiniGridActions.TURN_LEFT), int(MiniGridActions.TURN_RIGHT)):
                info["invalid_action"] = True
        if transition is not None:
            mechanic = classify_mechanic(ctx, before, action, transition.label, after)
            if mechanic is not None:
                info["mechanic"] = mechanic

        self._step_count += 1
        truncated = self._step_count >= self._max_steps
        reward, terminated = 0.0, False
        if self._goal_reached(after):
            # MiniGridEnv._reward()
            reward = 1 - 0.9 * (self._step_count / self._max_steps)
            terminated = True

        self.env.state.update_visibility()
        obs = self.render()
        state = self._build_grid_state()
        state.terminated = terminated
        state.truncated = truncated
        state.reward = reward
        state.goal_reached = terminated and reward > 0
        return obs, reward, terminated, truncated, state, info

    def _read_planner_state(self) -> PlannerState:
        ws = self.env.state
        holding = ws.agent.holding
        carrying_key = holding.id if holding is not None and holding.obj_type == "key" else None
        key_positions = frozenset(
            (obj.id, *self._xy_by_cell[obj.cell_id])
            for obj in ws.objects.values()
            if obj.obj_type == "key" and obj.cell_id is not None
        )
        by_type = self._objects_by_type()
        return PlannerState(
            agent_pos=self._xy_by_cell[ws.agent.cell_id],
            agent_dir=self._native_facing.index(ws.agent.facing),
            carrying_key=carrying_key,
            collected_keys=frozenset(ws.collected_keys),
            active_switches=frozenset(o.id for o in by_type["switch"] if o.is_active),
            used_switches=frozenset(o.id for o in by_type["switch"] if o.used),
            open_gates=frozenset(o.id for o in by_type["gate"] if o.is_open),
            open_doors=frozenset(o.id for o in by_type["door"] if o.is_open or not o.is_locked),
            key_positions=key_positions,
            freeze_remaining=int(ws.freeze_remaining),
            rotator_dirs=tuple(int(o.direction) for o in by_type["rotating"]),
        )

    def _write_planner_state(self, state: PlannerState) -> None:
        ws = self.env.state
        ws.agent.cell_id = self._cell_by_xy[tuple(state.agent_pos)]
        ws.agent.facing = self._native_facing[state.agent_dir]
        ws.collected_keys = set(state.collected_keys)
        ws.freeze_remaining = int(state.freeze_remaining)

        by_type = self._objects_by_type()
        for tile, direction in zip(by_type["rotating"], state.rotator_dirs):
            tile.direction = direction
        for switch in by_type["switch"]:
            switch.is_active = switch.id in state.active_switches
            switch.used = switch.id in state.used_switches
        for gate in by_type["gate"]:
            gate.set_open(gate.id in state.open_gates)
        for door in by_type["door"]:
            is_open = door.id in state.open_doors
            door.is_open = is_open
            door.is_locked = not is_open

        # Keys on the ground live in ws.objects; a held or consumed key does
        # not (multigrid's own PICKUP convention).
        for key_id in self._key_objects:
            ws.objects.pop(key_id, None)
        for key_id, x, y in state.key_positions:
            key = self._key_objects[key_id]
            key.cell_id = self._cell_by_xy[(x, y)]
            ws.objects[key_id] = key
        ws.agent.holding = None
        if state.carrying_key is not None:
            ws.agent.holding = self._key_objects[state.carrying_key]
            ws.agent.holding.cell_id = None

    def _objects_by_type(self) -> dict[str, list]:
        """Rulebook objects by type, each list in spec order (ids sort as
        written by _convert_task_spec, so rotating_i is the spec's i-th tile)."""
        grouped: dict[str, list] = {t: [] for t in ("switch", "gate", "door", "rotating")}
        for obj in self.env.state.objects.values():
            if obj.obj_type in grouped:
                grouped[obj.obj_type].append(obj)
        grouped["rotating"].sort(key=lambda o: int(o.id.rsplit("_", 1)[1]))
        return grouped

    def _goal_reached(self, state: PlannerState) -> bool:
        """CustomMiniGridEnv._check_goal_completion over a PlannerState."""
        goal = self.task_spec.goal
        if goal is None or goal.goal_type == "reach_position":
            return tuple(state.agent_pos) == tuple(self._planning_ctx.goal)
        on_ground = {key_id for key_id, _x, _y in state.key_positions}
        if goal.goal_type in {"pickup_key", "collect_key"}:
            return bool(goal.target_ids) and all(k not in on_ground for k in goal.target_ids)
        if goal.goal_type == "activate_switch":
            return bool(goal.target_ids) and all(s in state.active_switches for s in goal.target_ids)
        if goal.goal_type == "collect_all":
            key_ids = {key.id for key in self.task_spec.mechanisms.keys}
            switch_ids = {switch.id for switch in self.task_spec.mechanisms.switches}
            gate_ids = {gate.id for gate in self.task_spec.mechanisms.gates}

            def done(target_id: str) -> bool:
                if target_id in key_ids:
                    return target_id not in on_ground
                if target_id in switch_ids:
                    return target_id in state.active_switches
                if target_id in gate_ids:
                    return target_id in state.open_gates
                return False  # blocks never leave the grid under the rulebook

            return bool(goal.target_ids) and all(done(t) for t in goal.target_ids)
        if goal.goal_type == "push_block_to":
            # Blocks are static obstacles under the rulebook (as in MiniGrid's).
            blocks = {block.id: block.position.to_tuple() for block in self.task_spec.mechanisms.blocks}
            return len(goal.target_ids) == len(goal.target_positions) and all(
                blocks.get(block_id) == target.to_tuple()
                for block_id, target in zip(goal.target_ids, goal.target_positions)
            )
        if goal.goal_type == "survive_steps":
            return self._step_count >= self._max_steps
        return False

    def render(self) -> np.ndarray:
        """
        Render the current environment state.

        Returns:
            RGB image array of shape (H, W, 3)
        """
        if self.env is None:
            return np.zeros((self.render_height, self.render_width, 3), dtype=np.uint8)

        if self.uses_rulebook:
            visible = explored = None
            if self.task_spec.rules.observability != "full":
                visible = {self._xy_by_cell[c] for c in self.env.state.visible_cells}
                explored = {self._xy_by_cell[c] for c in self.env.state.explored_cells}
            return self._frame_renderer.render(
                self._read_planner_state(), visible_cells=visible, explored_cells=explored
            )

        return self.env.render()

    def get_mission_text(self) -> str:
        """
        Get the mission/goal description text.

        Returns:
            Human-readable mission description
        """
        if self.task_spec is None:
            return "No mission"

        if self.uses_rulebook:
            return self.task_spec.get_mission_text()  # as MiniGrid's env.mission

        # Use task description or generate from goal
        if self.task_spec.description:
            return self.task_spec.description

        if self.task_spec.goal:
            goal_type = self.task_spec.goal.goal_type
            if goal_type == "reach_position":
                return f"Navigate to position ({self.task_spec.goal.target.x}, {self.task_spec.goal.target.y})"
            elif goal_type == "collect_all":
                return f"Collect all items: {', '.join(self.task_spec.goal.target_ids)}"
            elif goal_type == "push_block_to":
                return "Push blocks to target positions"

        return "Complete the task"

    def get_state(self) -> GridState:
        """
        Get the current environment state.

        Returns:
            Current GridState
        """
        return self._build_grid_state()

    def rotator_directions(self) -> tuple[int, ...]:
        if not self.uses_rulebook or self.env is None or self.env.state is None:
            return ()
        return tuple(int(o.direction) for o in self._objects_by_type()["rotating"])

    def freeze_remaining(self) -> int:
        if not self.uses_rulebook or self.env is None or self.env.state is None:
            return 0
        return int(self.env.state.freeze_remaining)

    def _build_grid_state(self) -> GridState:
        """
        Build a GridState from the current MultiGrid state.

        Returns:
            GridState representing current environment
        """
        if self.env is None or self.env.state is None:
            return GridState(
                agent_position=(0, 0),
                agent_direction=0,
                step_count=self._step_count,
                max_steps=self._max_steps,
            )
        if self.uses_rulebook and self._xy_by_cell:
            return self._rulebook_grid_state()

        state = self.env.state
        tiling = self.env.tiling

        def cell_to_grid_pos(cell_id: str) -> tuple[int, int]:
            cell = tiling.cells.get(cell_id)
            if cell is not None:
                return (cell.col, cell.row)
            pos = tiling.cell_to_canonical(cell_id)
            return (
                int(pos[0] * self.task_spec.maze.dimensions[0]),
                int(pos[1] * self.task_spec.maze.dimensions[1])
            )

        # Get agent position in grid coordinates
        grid_pos = cell_to_grid_pos(state.agent.cell_id)

        # Get carrying object
        carrying = None
        if state.agent.holding is not None:
            carrying = state.agent.holding.id

        block_ids = {block.id for block in self.task_spec.mechanisms.blocks}
        key_ids = {key.id for key in self.task_spec.mechanisms.keys}

        # Build block positions
        block_positions = {}
        for obj_id, obj in state.objects.items():
            if obj_id in block_ids and obj.obj_type == "movable" and obj.cell_id is not None:
                block_positions[obj_id] = cell_to_grid_pos(obj.cell_id)

        # Convert visibility sets from cell_id strings to (x,y) grid coords
        obs_mode = getattr(state, 'observability_mode', 'full')
        visible_xy = set()
        explored_xy = set()

        if obs_mode != "full":
            for cell_id in state.visible_cells:
                visible_xy.add(cell_to_grid_pos(cell_id))
            for cell_id in state.explored_cells:
                explored_xy.add(cell_to_grid_pos(cell_id))

        open_doors = {
            obj.id for obj in state.objects.values()
            if obj.obj_type == "door" and getattr(obj, "is_open", False)
        }
        active_switches = {
            obj.id for obj in state.objects.values()
            if obj.obj_type == "switch" and getattr(obj, "is_active", False)
        }
        open_gates = {
            obj.id for obj in state.objects.values()
            if obj.obj_type == "gate" and getattr(obj, "is_open", False)
        }
        collected_keys = key_ids - {
            obj.id for obj in state.objects.values()
            if obj.obj_type == "key"
        }
        teleporter_cooldowns = {
            obj.id: getattr(obj, "current_cooldown", 0)
            for obj in state.objects.values()
            if obj.obj_type == "teleporter"
        }

        return GridState(
            agent_position=grid_pos,
            agent_direction=state.agent.facing,
            agent_carrying=carrying,
            step_count=self._step_count,
            max_steps=self._max_steps,
            open_doors=open_doors,
            collected_keys=collected_keys,
            active_switches=active_switches,
            open_gates=open_gates,
            block_positions=block_positions,
            teleporter_cooldowns=teleporter_cooldowns,
            goal_reached=state.check_goal(),
            observability_mode=obs_mode,
            visible_cells=visible_xy,
            explored_cells=explored_xy,
        )

    def _rulebook_grid_state(self) -> GridState:
        """MiniGridBackend._get_grid_state's shape, read off the planner state."""
        planner = self._read_planner_state()
        spec = self.task_spec
        carrying = None
        if planner.carrying_key is not None:
            # The key's spec colour. MiniGrid reports its own palette name,
            # which folds cyan into blue and brown into grey.
            carrying = self._planning_ctx.keys_by_id[planner.carrying_key]["color"]
        obs_mode = spec.rules.observability
        visible_cells: set[tuple[int, int]] = set()
        explored_cells: set[tuple[int, int]] = set()
        if obs_mode != "full":
            visible_cells = {self._xy_by_cell[c] for c in self.env.state.visible_cells}
            explored_cells = {self._xy_by_cell[c] for c in self.env.state.explored_cells}
        return GridState(
            agent_position=tuple(planner.agent_pos),
            agent_direction=int(planner.agent_dir),
            agent_carrying=carrying,
            step_count=self._step_count,
            max_steps=self._max_steps,
            open_doors=set(planner.open_doors),
            collected_keys=set(planner.collected_keys),
            active_switches=set(planner.active_switches),
            open_gates=set(planner.open_gates),
            # Blocks are static under the rulebook.
            block_positions={block.id: block.position.to_tuple() for block in spec.mechanisms.blocks},
            # The rulebook has no teleporter cooldown (MiniGrid reports 0 too).
            teleporter_cooldowns={
                f"{tp.id}_{end}": 0 for tp in spec.mechanisms.teleporters for end in ("a", "b")
            },
            key_positions={key_id: (x, y) for key_id, x, y in planner.key_positions},
            freeze_remaining=int(planner.freeze_remaining),
            rotator_dirs=tuple(planner.rotator_dirs),
            goal_reached=self._goal_reached(planner),
            observability_mode=obs_mode,
            visible_cells=visible_cells,
            explored_cells=explored_cells,
            tiling=self.tiling_type,
        )

    def close(self) -> None:
        """Clean up resources."""
        if self.env is not None:
            # MultiGridEnv doesn't have explicit close
            self.env = None
        self._configured = False

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        """Shape of observations (H, W, C)."""
        if self.uses_rulebook:
            return self._frame_renderer.frame_shape
        return (64, 64, 3)
