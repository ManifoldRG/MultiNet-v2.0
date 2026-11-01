"""Which backend/camera/resolution an episode renders with (run-config ``render``).

The 3D backend is a render layer: mechanics, reward and termination come from
the same state backend either way, so only the frame differs. The settings ride
in the artifact/hash *label* rather than in a new hashed field, so 2D runs keep
byte-identical run hashes and every camera gets its own artifact directory. A 3D
label also carries RENDER3D_VERSION, so a visual change writes new episodes
beside the old ones instead of over them.

A 2D block may also override every task's ``rules.observability`` (and
``rules.view_size``), so the same manifest mazes run under view_cone or
fog_of_war without new maze files. Observability changes only what the frame
shows (custom_env masks unseen cells), never the dynamics, so it rides the same
label/provenance path: its own artifact directory and run hash, with
FOG_RENDER_VERSION in the label like RENDER3D_VERSION. Without the key nothing
changes.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Optional

from gridworld.render3d.cameras import PRESETS  # pure pose math; no mujoco import
from gridworld.task_spec import TaskSpecification

BACKENDS = ("minigrid", "mujoco3d")
# MiniGrid's tile size. The 3D frame is square (this many px per cell of the
# longer side), so it matches MiniGrid's 32W x 32H frame, and its image-token
# cost, only for a square maze.
GRID_PIXELS_PER_CELL = 32
_KEYS = {"backend", "camera", "resolution", "start_map", "observability", "view_size"}
# The partial-observability modes a 2D render block may impose on every task.
PARTIAL_OBSERVABILITY = ("view_cone", "fog_of_war")
# Bumped when the 2D partial-observability frame changes (custom_env's unknown
# tile, REMEMBERED_DIM or what is masked), so a rerun writes beside episodes
# drawn by the old mask instead of reusing them. Full-observability 2D frames
# carry no version: their label stays "minigrid".
FOG_RENDER_VERSION = 1


@dataclass(frozen=True)
class RenderSettings:
    backend: str = "minigrid"
    camera: str | None = None
    resolution: int | str = "grid"  # "grid" = GRID_PIXELS_PER_CELL per cell
    # A top-down map of the reset state, shown to the model once per request
    # beside its 3D view (interface/episode_step.py). Off = today's prompts.
    start_map: bool = False
    # 2D only: impose this rules.observability / rules.view_size on every task
    # (apply_to_spec). None = each task file's own rules.
    observability: str | None = None
    view_size: int | None = None

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
        start_map = block.get("start_map", False)
        if not isinstance(start_map, bool):
            raise ValueError(f"render start_map must be a bool, got {start_map!r}")
        if start_map and backend != "mujoco3d":
            raise ValueError("start_map is a 3D setting; the minigrid frame already shows the whole maze")
        if start_map and camera == "top_down":
            raise ValueError("start_map adds a top-down map; the top_down camera already is one")
        observability, view_size = _parse_observability(block, backend)
        return cls(
            backend=backend,
            camera=camera,
            resolution=resolution,
            start_map=start_map,
            observability=observability,
            view_size=view_size,
        )

    @property
    def label(self) -> str:
        """Artifact directory and run-hash label: one per rendered view and
        render version (read at call time, like provenance)."""
        if self.backend == "minigrid":
            if self.observability is None:
                return "minigrid"
            return f"minigrid_{self.observability}_v{self.view_size}_r{FOG_RENDER_VERSION}"
        from gridworld import render3d

        suffix = "_map" if self.start_map else ""
        return f"{self.backend}_{self.camera}_{self.resolution}{suffix}_r{render3d.RENDER3D_VERSION}"

    def backend_kwargs(self, spec: TaskSpecification) -> dict[str, Any]:
        """Backend constructor arguments for this task."""
        if self.backend == "minigrid":
            return {}
        kwargs: dict[str, Any] = {"camera": self.camera, "resolution": self.frame_pixels(spec)}
        if self.start_map:
            kwargs["start_map"] = True
        return kwargs

    def frame_pixels(self, spec: TaskSpecification) -> int:
        if self.resolution != "grid":
            return int(self.resolution)
        return GRID_PIXELS_PER_CELL * max(spec.maze.dimensions)

    def apply_to_spec(self, spec: TaskSpecification) -> TaskSpecification:
        """``spec`` with this block's observability imposed (unchanged without one)."""
        if self.observability is None:
            return spec
        rules = dataclasses.replace(
            spec.rules, observability=self.observability, view_size=self.view_size
        )
        return dataclasses.replace(spec, rules=rules)

    def unit_block(self) -> Optional[dict[str, Any]]:
        """The run-config ``render`` block a distributed worker rebuilds these
        settings from (``from_run_config``); None for the 2D default, whose
        units carry no block."""
        if self.backend != "minigrid":
            return self._render3d_fields()
        if self.observability is None:
            return None
        return {"backend": "minigrid", "observability": self.observability, "view_size": self.view_size}

    def _render3d_fields(self) -> dict[str, Any]:
        # The 3D settings exactly as the unit block and provenance recorded them
        # before the 2D observability fields existed, so 3D hashes do not move.
        return {
            "backend": self.backend,
            "camera": self.camera,
            "resolution": self.resolution,
            "start_map": self.start_map,
        }

    def provenance(self, spec: TaskSpecification) -> Optional[dict[str, Any]]:
        """What draws this task's frames, for the episode cache key and
        run_inputs.json. 3D: the settings, the resolved frame size, the
        render-layer version and the mujoco version. 2D with an observability
        override: the mode, view size and FOG_RENDER_VERSION. None for plain 2D,
        so 2D run hashes stay byte-identical."""
        if self.backend == "minigrid":
            if self.observability is None:
                return None
            return {**self.unit_block(), "fog_render_version": FOG_RENDER_VERSION}
        from gridworld.render3d import RENDER3D_VERSION

        return {
            **self._render3d_fields(),
            "frame_pixels": self.frame_pixels(spec),
            "render3d_version": RENDER3D_VERSION,
            "mujoco_version": _mujoco_version(),
        }

    def check_experiment(self, cell: Any) -> None:
        """Fail fast, before any paid call, when an experiment cell (an
        ExperimentConfig) cannot show the model what these settings draw."""
        self.check_observation(cell.observation)
        if self.observability is not None and cell.prompting == "text_initial_maze":
            raise ValueError(
                f"render observability {self.observability!r} hides unseen cells, but prompting "
                "'text_initial_maze' prints the whole maze in the system prompt"
            )

    def check_observation(self, observation: str) -> None:
        """Fail fast when a 3D render meets an observation with no images: the
        model would never see a frame, so the run would repeat the 2D run under
        another label and pay for GL work. A 2D observability override needs
        image_only: the text observations describe the whole maze."""
        if self.observability is not None and observation != "image_only":
            raise ValueError(
                f"render observability {self.observability!r} masks the frame only; observation "
                f"{observation!r} describes the whole maze in text, so run it with image_only"
            )
        if self.backend == "minigrid" or observation != "text_only":
            return
        if self.start_map:
            raise ValueError(
                "render start_map shows the model an image, but observation 'text_only' carries none"
            )
        raise ValueError(
            f"render {self.backend!r} draws frames, but observation 'text_only' shows the model none; "
            "run text_only on the 2D backend"
        )


def _parse_observability(block: dict[str, Any], backend: str) -> tuple[str | None, int | None]:
    """(observability, view_size) from a render block; (None, None) without one."""
    if "observability" not in block:
        if "view_size" in block:
            raise ValueError("render view_size needs render observability (view_cone or fog_of_war)")
        return None, None
    observability = block["observability"]
    if observability not in PARTIAL_OBSERVABILITY:
        raise ValueError(
            f"render observability must be one of {list(PARTIAL_OBSERVABILITY)}, got {observability!r}; "
            "omit it to keep each task's own rules"
        )
    if backend != "minigrid":
        raise ValueError("observability is a 2D setting; the 3D scene renders full observability only")
    view_size = block.get("view_size", 7)  # MiniGrid's default agent_view_size
    if isinstance(view_size, bool) or not isinstance(view_size, int) or view_size < 3 or view_size % 2 == 0:
        raise ValueError(f"render view_size must be an odd int >= 3, got {view_size!r}")
    return observability, view_size


def _mujoco_version() -> str:
    """Read lazily (3D runs only), after the MUJOCO_GL default is in place."""
    from gridworld.render3d.gl import default_mujoco_gl

    default_mujoco_gl()  # must precede the first `import mujoco`
    import mujoco

    return mujoco.__version__
