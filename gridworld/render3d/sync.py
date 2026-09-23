"""Push a GridState into a compiled scene: show/hide geoms, move mocap bodies.

Hidden geoms move to HIDDEN_GROUP, which the renderer never draws. (Sinking
them below the floor instead left them visible past the floor's edge.)
Nothing is recompiled per frame.
"""

from __future__ import annotations

import math

import mujoco

from gridworld.backends.base import GridState

from . import palette
from .cameras import DIRECTION_YAW, cell_center
from .scene import HIDDEN_GROUP, KEY_HEIGHT, ROTATOR_ARROW_LIFT, SKULL_TILT, SceneIndex

LIFTED_ARROW_ALPHA = 0.55


def _yaw_pitch_quat(yaw: float, pitch_up: float) -> tuple[float, float, float, float]:
    """Quaternion (w, x, y, z) turning +x to ``yaw`` about z, then raising it by ``pitch_up``."""
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(-pitch_up / 2), math.sin(-pitch_up / 2)  # negative y-rotation lifts +x
    # q = qz(yaw) * qy(-pitch)
    return (cy * cp, -sy * sp, cy * sp, sy * cp)


class SceneState:
    """Resolves SceneIndex names to ids once; applies a GridState per frame."""

    def __init__(self, model: mujoco.MjModel, index: SceneIndex):
        self.model = model
        self.index = index
        self._home_group = model.geom_group.copy()
        # Compiled positions: a mesh geom's pos carries the compiler's
        # centre-of-mass offset, so lifts add to it rather than replace it.
        self._home_pos = model.geom_pos.copy()

        def ids(names) -> tuple[int, ...]:
            out = []
            for name in names:
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                if gid < 0:
                    raise KeyError(f"geom {name!r} missing from scene")
                out.append(gid)
            return tuple(out)

        def mocap(body: str) -> int:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
            if body_id < 0 or model.body_mocapid[body_id] < 0:
                raise KeyError(f"mocap body {body!r} missing from scene")
            return int(model.body_mocapid[body_id])

        def body_geoms(body: str) -> tuple[int, ...]:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
            first = int(model.body_geomadr[body_id])
            return tuple(range(first, first + int(model.body_geomnum[body_id])))

        self._doors = {d: (ids(index.door_closed[d]), ids(index.door_open[d])) for d in index.door_closed}
        self._switches = {s: (ids(index.switch_off[s]), ids(index.switch_on[s])) for s in index.switch_off}
        self._gates = {g: (ids(index.gate_closed[g]), ids(index.gate_open[g])) for g in index.gate_closed}
        self._carried = {colour: ids(names) for colour, names in index.carried.items()}
        self._keys = {
            key_id: (mocap(body), body_geoms(body)) for key_id, body in index.key_bodies.items()
        }
        self._rotators = {
            i: tuple(ids(names) for names in arrows) for i, arrows in index.rotator_arrows.items()
        }
        self._skulls = {i: mocap(body) for i, body in index.kill_bodies.items()}
        self._agent = mocap(index.agent_body)

    def _show(self, geom_ids, visible: bool) -> None:
        for gid in geom_ids:
            self.model.geom_group[gid] = self._home_group[gid] if visible else HIDDEN_GROUP

    def apply(
        self,
        data: mujoco.MjData,
        state: GridState,
        door_states: dict[str, bool],
        *,
        rotators: tuple[int, ...] | None = None,
        yaw: float | None = None,
    ) -> None:
        """``rotators``: current direction of each rotating tile in spec order
        (None: the spec's initial directions). ``yaw`` (degrees) overrides the
        agent's heading yaw (turn animation)."""
        if rotators is None:
            rotators = self.index.rotator_initial
        agent_cell = tuple(int(v) for v in state.agent_position)
        for i, arrows in self._rotators.items():
            shown = int(rotators[i]) if i < len(rotators) else None
            lift = ROTATOR_ARROW_LIFT if self.index.rotator_cells[i] == agent_cell else 0.0
            for d, geom_ids in enumerate(arrows):
                self._show(geom_ids, d == shown)
                for gid in geom_ids:
                    self.model.geom_pos[gid][2] = self._home_pos[gid][2] + lift
                    # lifted over the wedge, the arrow goes translucent so the
                    # agent's heading stays readable underneath it
                    self.model.geom_rgba[gid][3] = LIFTED_ARROW_ALPHA if lift else 1.0
        for door_id, (closed, opened) in self._doors.items():
            is_open = bool(door_states.get(door_id, False))
            self._show(closed, not is_open)
            self._show(opened, is_open)
        for switch_id, (off, on) in self._switches.items():
            active = switch_id in state.active_switches
            self._show(off, not active)
            self._show(on, active)
        for gate_id, (closed, opened) in self._gates.items():
            is_open = gate_id in state.open_gates
            self._show(closed, not is_open)
            self._show(opened, is_open)
        for key_id, (mocap_id, geom_ids) in self._keys.items():
            position = state.key_positions.get(key_id)
            self._show(geom_ids, position is not None)  # held or consumed: hidden in place
            if position is not None:
                x, y, _ = cell_center(*position)
                data.mocap_pos[mocap_id] = (x, y, KEY_HEIGHT)
        carrying = palette.normalize(state.agent_carrying) if state.agent_carrying else None
        for colour, geom_ids in self._carried.items():
            self._show(geom_ids, colour == carrying)
        ax, ay, _ = cell_center(*state.agent_position)
        for i, mocap_id in self._skulls.items():
            sx, sy, _ = cell_center(*self.index.kill_cells[i])
            bearing = math.atan2(ay - sy, ax - sx) if (ax, ay) != (sx, sy) else 0.0
            data.mocap_quat[mocap_id] = _yaw_pitch_quat(bearing, math.radians(SKULL_TILT))
        data.mocap_pos[self._agent] = (ax, ay, 0.0)
        half = math.radians(DIRECTION_YAW[int(state.agent_direction)] if yaw is None else yaw) / 2.0
        data.mocap_quat[self._agent] = (math.cos(half), 0.0, 0.0, math.sin(half))
        mujoco.mj_kinematics(self.model, data)  # mocap edits -> world poses
