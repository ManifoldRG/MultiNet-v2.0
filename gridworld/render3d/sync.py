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
from .scene import HIDDEN_GROUP, KEY_HEIGHT, SceneIndex


class SceneState:
    """Resolves SceneIndex names to ids once; applies a GridState per frame."""

    def __init__(self, model: mujoco.MjModel, index: SceneIndex):
        self.model = model
        self.index = index
        self._home_group = model.geom_group.copy()

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
        self._agent = mocap(index.agent_body)

    def _show(self, geom_ids, visible: bool) -> None:
        for gid in geom_ids:
            self.model.geom_group[gid] = self._home_group[gid] if visible else HIDDEN_GROUP

    def apply(self, data: mujoco.MjData, state: GridState, door_states: dict[str, bool]) -> None:
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
        data.mocap_pos[self._agent] = (ax, ay, 0.0)
        yaw = math.radians(DIRECTION_YAW[int(state.agent_direction)])
        data.mocap_quat[self._agent] = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))
        mujoco.mj_kinematics(self.model, data)  # mocap edits -> world poses
