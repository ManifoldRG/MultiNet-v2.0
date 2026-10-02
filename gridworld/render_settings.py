"""Which backend/camera/resolution an episode renders with (run-config ``render``).

The 3D backend is a render layer: mechanics, reward and termination come from
the same state backend either way, so only the frame differs. The settings ride
in the artifact/hash *label* rather than in a new hashed field, so 2D runs keep
byte-identical run hashes and every camera gets its own artifact directory.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Optional

from gridworld.render3d.cameras import PRESETS  # pure pose math; no mujoco import
from gridworld.task_spec import TaskSpecification

BACKENDS = ("minigrid", "mujoco3d")
GRID_PIXELS_PER_CELL = 32  # MiniGrid's tile size: a 3D frame then costs the same image tokens
_KEYS = {"backend", "camera", "resolution"}


@dataclass(frozen=True)
class RenderSettings:
    backend: str = "minigrid"
    camera: str | None = None
    resolution: int | str = "grid"  # "grid" = GRID_PIXELS_PER_CELL per cell

    @classmethod
    def from_run_config(cls, run_config: dict[str, Any]) -> RenderSettings:
        """Parse the optional top-level ``render`` block, failing fast on
        anything unknown (like the experiment_config overlay does)."""
        if "render" not in run_config:
            return cls()
        block = run_config["render"]
        if not isinstance(block, dict):  # a falsy [] / "" / 0 / null must not mean 2D
            raise ValueError(f"run-config 'render' must be an object, got {block!r}")
        unknown = sorted(set(block) - _KEYS)
        if unknown:
            raise ValueError(f"unknown render keys: {', '.join(unknown)}; known: {sorted(_KEYS)}")
        backend = block.get("backend", "minigrid")
        if backend not in BACKENDS:
            raise ValueError(f"unknown render backend {backend!r}; choose from {list(BACKENDS)}")
        camera = block.get("camera")
        if backend == "minigrid":
            # MiniGrid would silently ignore these, recording a setting the run never used.
            for key in ("camera", "resolution"):
                if key in block:
                    raise ValueError(f"{key} is a 3D setting; the minigrid backend has no {key}")
        if backend == "mujoco3d" and camera not in PRESETS:
            raise ValueError(f"3D runs need a camera from {list(PRESETS)}, got {camera!r}")
        resolution = block.get("resolution", "grid")
        if resolution != "grid" and (not isinstance(resolution, int) or isinstance(resolution, bool) or resolution <= 0):
            raise ValueError(f"render resolution must be a positive int or 'grid', got {resolution!r}")
        return cls(backend=backend, camera=camera, resolution=resolution)

    @property
    def label(self) -> str:
        """Artifact directory and run-hash label: one per rendered view."""
        if self.backend == "minigrid":
            return "minigrid"
        return f"{self.backend}_{self.camera}_{self.resolution}"

    def backend_kwargs(self, spec: TaskSpecification) -> dict[str, Any]:
        """Backend constructor arguments for this task."""
        if self.backend == "minigrid":
            return {}
        return {"camera": self.camera, "resolution": self.frame_pixels(spec)}

    def frame_pixels(self, spec: TaskSpecification) -> int:
        if self.resolution != "grid":
            return int(self.resolution)
        return GRID_PIXELS_PER_CELL * max(spec.maze.dimensions)

    def provenance(self, spec: TaskSpecification) -> Optional[dict[str, Any]]:
        """What draws this task's 3D frames, for the episode cache key and
        run_inputs.json: the settings, the resolved frame size, the render-layer
        version and the mujoco version. None for 2D, so 2D run hashes stay
        byte-identical."""
        if self.backend == "minigrid":
            return None
        from gridworld.render3d import RENDER3D_VERSION

        return {
            **dataclasses.asdict(self),
            "frame_pixels": self.frame_pixels(spec),
            "render3d_version": RENDER3D_VERSION,
            "mujoco_version": _mujoco_version(),
        }


def _mujoco_version() -> str:
    """Read lazily (3D runs only), after the MUJOCO_GL default is in place."""
    from gridworld.render3d.gl import default_mujoco_gl

    default_mujoco_gl()  # must precede the first `import mujoco`
    import mujoco

    return mujoco.__version__
