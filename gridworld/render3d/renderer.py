"""SceneRenderer: TaskSpecification + GridState -> RGB frame (MuJoCo, headless)."""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")  # must precede `import mujoco`

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from gridworld.backends.base import GridState  # noqa: E402
from gridworld.task_spec import TaskSpecification  # noqa: E402

from .cameras import (  # noqa: E402
    EYE_PRESETS,
    PRESETS,
    TILT_LEVELS,
    CameraPose,
    check_tilt,
    pose_for,
    tilt_pose,
    view_wall_height,
)
from .hud import NORTH, draw_compass, turns_with_agent  # noqa: E402
from .scene import AGENT_GROUP, HIDDEN_GROUP, build_scene  # noqa: E402
from .sync import SceneState  # noqa: E402


class SceneRenderer:
    def __init__(
        self,
        spec: TaskSpecification,
        *,
        camera: str = "top_down",
        resolution: int = 512,
        wall_height: float | None = None,
        tilt: int | None = None,
    ):
        if camera not in PRESETS:
            raise ValueError(f"unknown camera preset {camera!r}; choose from {PRESETS}")
        check_tilt(tilt)
        self.spec = spec
        self.resolution = int(resolution)
        self._wall_height = wall_height
        self._camera = camera
        self._tilt = tilt  # a demo tilt level overrides the preset while set
        self._renderer: mujoco.Renderer | None = None
        self.last_pose: CameraPose | None = None
        self._build()

    @property
    def camera(self) -> str:
        return self._camera

    @property
    def tilt(self) -> int | None:
        return self._tilt

    @property
    def view_turns_with_agent(self) -> bool:
        return turns_with_agent(self._camera, self._tilt)

    def _effective_wall_height(self) -> float:
        return view_wall_height(self._camera, self._tilt, self._wall_height)

    def _rebuild_if_wall_height_changed(self) -> None:
        if self._effective_wall_height() != self.index.wall_height:
            self.close()
            self._build()

    def _build(self) -> None:
        xml, self.index = build_scene(
            self.spec, wall_height=self._effective_wall_height(), resolution=self.resolution
        )
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self._sync = SceneState(self.model, self.index)
        self._renderer = mujoco.Renderer(self.model, self.resolution, self.resolution)
        self._option = mujoco.MjvOption()
        self._cam = mujoco.MjvCamera()
        self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    def set_camera(self, camera: str) -> None:
        if camera not in PRESETS:
            raise ValueError(f"unknown camera preset {camera!r}; choose from {PRESETS}")
        self._camera = camera
        self._tilt = None
        self._rebuild_if_wall_height_changed()

    def set_tilt(self, level: int | None) -> None:
        """Show a demo tilt level (None: back to the preset). Display-only."""
        check_tilt(level)
        self._tilt = level
        self._rebuild_if_wall_height_changed()

    def render(
        self,
        state: GridState,
        door_states: dict[str, bool],
        *,
        rotators: tuple[int, ...] | None = None,
        yaw: float | None = None,
    ) -> np.ndarray:
        """``rotators``: each rotating tile's current direction (spec order;
        None = initial). ``yaw`` (degrees) overrides the heading's yaw for the
        agent and the views that turn with it (frames partway through a turn)."""
        self._sync.apply(self.data, state, door_states, rotators=rotators, yaw=yaw)
        where = dict(
            agent_cell=tuple(state.agent_position),
            direction=int(state.agent_direction),
            maze_dims=(self.index.width, self.index.height),
            yaw=yaw,
        )
        if self._tilt is None:
            pose = pose_for(self._camera, wall_height=self.index.wall_height, **where)
            eye_view = self._camera in EYE_PRESETS
        else:
            pose = tilt_pose(self._tilt, **where)
            eye_view = TILT_LEVELS[self._tilt].preset in EYE_PRESETS
        self.last_pose = pose
        # Free-camera projection is model-global: set it per frame.
        self.model.vis.global_.orthographic = int(pose.orthographic)
        self.model.vis.global_.fovy = pose.fovy
        self._cam.lookat[:] = pose.lookat
        self._cam.distance = pose.distance
        self._cam.azimuth = pose.azimuth
        self._cam.elevation = pose.elevation
        # MjvOption hides geom groups 3-5 by default: switch the agent's group
        # on explicitly, and off for the first-person eye. The hidden group
        # stays off whatever the default.
        self._option.geomgroup[AGENT_GROUP] = 0 if eye_view else 1
        self._option.geomgroup[HIDDEN_GROUP] = 0
        self._renderer.update_scene(self.data, camera=self._cam, scene_option=self._option)
        frame = self._renderer.render().copy()
        # Every 3D view carries a compass, showing what is up in THIS frame: the
        # agent's heading where the view turns with it, north otherwise. Arms of
        # an ablation then differ only in viewpoint.
        up = int(state.agent_direction) if self.view_turns_with_agent else NORTH
        frame = draw_compass(frame, up)
        return frame

    def close(self) -> None:
        # Explicit close avoids EGL "Exception ignored" noise at interpreter exit.
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
