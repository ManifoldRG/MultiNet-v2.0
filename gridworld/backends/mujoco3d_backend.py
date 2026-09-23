"""3D backend: any AbstractGridBackend supplies mechanics + GridState; MuJoCo
draws the frame.

Reward, termination, GridState and info pass through untouched, so a 3D
episode is action-for-action identical to one on the state backend. Imports
only the common layer; mujoco loads lazily in configure().
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np

from ..render3d.cameras import DIRECTION_YAW, PRESETS, TILT_LEVELS, check_tilt, view_wall_height
from ..render3d.hud import turns_with_agent
from ..render3d.scene import check_supported
from ..task_spec import TaskSpecification
from .base import AbstractGridBackend, GridState


class Mujoco3DBackend(AbstractGridBackend):
    def __init__(
        self,
        state_backend: AbstractGridBackend,
        *,
        camera: str = "top_down",
        resolution: int = 512,
        wall_height: Optional[float] = None,
    ):
        super().__init__()
        if camera not in PRESETS:
            raise ValueError(f"unknown camera preset {camera!r}; choose from {PRESETS}")
        self.state_backend = state_backend
        self.resolution = int(resolution)
        self.wall_height = wall_height
        self._camera = camera
        self._tilt: Optional[int] = None  # demo tilt level; overrides the preset while set
        self._renderer = None
        self._frame: Optional[np.ndarray] = None
        self._frame_key = None

    def configure(self, task_spec: TaskSpecification) -> None:
        check_supported(task_spec)  # before touching the state backend
        from ..render3d.renderer import SceneRenderer  # lazy: mujoco loads only when used

        self.state_backend.configure(task_spec)
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._renderer = SceneRenderer(
            task_spec,
            camera=self._camera,
            resolution=self.resolution,
            wall_height=self.wall_height,
            tilt=self._tilt,
        )
        self.task_spec = task_spec
        self._configured = True
        self._frame = None
        self._frame_key = None

    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, GridState, dict]:
        if self._renderer is None:
            raise RuntimeError("Backend must be configured before reset()/step()")
        _flat, state, info = self.state_backend.reset(seed=seed)
        return self._frame_for(state), state, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, GridState, dict]:
        if self._renderer is None:
            raise RuntimeError("Backend must be configured before reset()/step()")
        _flat, reward, terminated, truncated, state, info = self.state_backend.step(action)
        return self._frame_for(state), reward, terminated, truncated, state, info

    def render(self) -> np.ndarray:
        if self._renderer is None:
            raise RuntimeError("Backend must be configured before render()")
        return self._frame_for(self.state_backend.get_state())

    def _frame_for(self, state: GridState) -> np.ndarray:
        doors = self.state_backend.door_states()
        rotators = self.state_backend.rotator_directions()
        freeze = self.state_backend.freeze_remaining()
        # Everything the frame depends on; a matching key reuses the cached frame
        # (the pygame UI calls render() every tick).
        key = (
            tuple(int(v) for v in state.agent_position),
            int(state.agent_direction),
            state.agent_carrying,
            frozenset(state.active_switches),
            frozenset(state.open_gates),
            frozenset((k, tuple(int(c) for c in v)) for k, v in state.key_positions.items()),
            frozenset(doors.items()),
            rotators,
            freeze,
            self._camera,
            self._tilt,
        )
        if self._frame is None or key != self._frame_key:
            self._frame = self._renderer.render(state, doors, rotators=rotators, freeze=freeze)
            self._frame_key = key
        return self._frame

    def render_turn(self, from_direction: int, fraction: float) -> np.ndarray:
        """Display-only frame ``fraction`` of the way through a turn from
        ``from_direction`` to the current heading. State and the cached frame
        are untouched; the compass shows whichever heading is nearer."""
        if self._renderer is None:
            raise RuntimeError("Backend must be configured before render_turn()")
        fraction = min(max(float(fraction), 0.0), 1.0)
        if fraction >= 1.0:
            return self.render()
        state = self.state_backend.get_state()
        start = DIRECTION_YAW[int(from_direction)]
        delta = (DIRECTION_YAW[int(state.agent_direction)] - start + 180.0) % 360.0 - 180.0
        shown = state if fraction >= 0.5 else dataclasses.replace(state, agent_direction=int(from_direction))
        return self._renderer.render(
            shown,
            self.state_backend.door_states(),
            rotators=self.state_backend.rotator_directions(),
            freeze=self.state_backend.freeze_remaining(),
            yaw=start + fraction * delta,
        )

    @property
    def view_turns_with_agent(self) -> bool:
        return turns_with_agent(self._camera, self._tilt)

    @property
    def tilt(self) -> Optional[int]:
        return self._tilt

    @property
    def tilt_levels(self) -> int:
        return len(TILT_LEVELS)

    @property
    def wall_height_shown(self) -> float:
        return view_wall_height(self._camera, self._tilt, self.wall_height)

    def set_tilt(self, level: Optional[int]) -> None:
        """Show a demo tilt level (None: back to the camera preset).
        Display-only: state is untouched."""
        check_tilt(level)
        self._tilt = level
        if self._renderer is not None:
            self._renderer.set_tilt(level)
        self._frame = None
        self._frame_key = None

    def get_mission_text(self) -> str:
        return self.state_backend.get_mission_text()

    def get_state(self) -> GridState:
        return self.state_backend.get_state()

    def door_states(self) -> dict[str, bool]:
        return self.state_backend.door_states()

    def rotator_directions(self) -> tuple[int, ...]:
        return self.state_backend.rotator_directions()

    def freeze_remaining(self) -> int:
        return self.state_backend.freeze_remaining()

    @property
    def frame_is_grid_aligned(self) -> bool:
        return False

    @property
    def action_space_size(self) -> int:
        return self.state_backend.action_space_size

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return (self.resolution, self.resolution, 3)

    @property
    def camera(self) -> str:
        return self._camera

    @property
    def camera_names(self) -> tuple[str, ...]:
        return PRESETS

    def set_camera(self, camera: str) -> None:
        if camera not in PRESETS:
            raise ValueError(f"unknown camera preset {camera!r}; choose from {PRESETS}")
        self._camera = camera
        self._tilt = None
        if self._renderer is not None:
            self._renderer.set_camera(camera)
        self._frame = None
        self._frame_key = None

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._configured = False
        self.state_backend.close()
