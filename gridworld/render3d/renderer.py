"""SceneRenderer: TaskSpecification + GridState -> RGB frame (MuJoCo, headless)."""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")  # must precede `import mujoco`

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from gridworld.backends.base import GridState  # noqa: E402
from gridworld.task_spec import TaskSpecification  # noqa: E402

from .cameras import DEFAULT_WALL_HEIGHT, PRESETS, CameraPose, pose_for  # noqa: E402
from .scene import AGENT_GROUP, build_scene  # noqa: E402
from .sync import SceneState  # noqa: E402


class SceneRenderer:
    def __init__(
        self,
        spec: TaskSpecification,
        *,
        camera: str = "top_down",
        resolution: int = 512,
        wall_height: float | None = None,
    ):
        if camera not in PRESETS:
            raise ValueError(f"unknown camera preset {camera!r}; choose from {PRESETS}")
        self.spec = spec
        self.resolution = int(resolution)
        self._wall_height = wall_height
        self._camera = camera
        self._renderer: mujoco.Renderer | None = None
        self.last_pose: CameraPose | None = None
        self._build()

    @property
    def camera(self) -> str:
        return self._camera

    def _effective_wall_height(self) -> float:
        if self._wall_height is not None:
            return float(self._wall_height)
        return DEFAULT_WALL_HEIGHT[self._camera]

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
        if self._effective_wall_height() != self.index.wall_height:
            self.close()
            self._build()

    def render(self, state: GridState, door_states: dict[str, bool]) -> np.ndarray:
        self._sync.apply(self.data, state, door_states)
        pose = pose_for(
            self._camera,
            agent_cell=tuple(state.agent_position),
            direction=int(state.agent_direction),
            maze_dims=(self.index.width, self.index.height),
            wall_height=self.index.wall_height,
        )
        self.last_pose = pose
        # Free-camera projection is model-global: set it per frame.
        self.model.vis.global_.orthographic = int(pose.orthographic)
        self.model.vis.global_.fovy = pose.fovy
        self._cam.lookat[:] = pose.lookat
        self._cam.distance = pose.distance
        self._cam.azimuth = pose.azimuth
        self._cam.elevation = pose.elevation
        # MjvOption hides geom groups 3-5 by default: switch the agent's group
        # on explicitly, and off for the first-person eye.
        self._option.geomgroup[AGENT_GROUP] = 0 if self._camera == "first_person" else 1
        self._renderer.update_scene(self.data, camera=self._cam, scene_option=self._option)
        return self._renderer.render().copy()

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
