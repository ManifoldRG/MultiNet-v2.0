"""3D (MuJoCo) rendering of task-spec mazes from a GridState.

Imports only the common layer (task_spec, GridState) plus numpy/mujoco --
never minigrid or the MiniGrid backend (tests/test_backend_import_boundary.py).

On Linux MUJOCO_GL defaults to osmesa (software rendering, works without a
GPU) when unset (gl.py); renderer.py, the only module here that imports mujoco,
applies that default first.
"""

# Version of what a 3D frame looks like. BUMP IT ON ANY VISUAL CHANGE (scene,
# palette, materials, lighting, cameras, HUD): it is folded into the 3D episode
# cache key and run_inputs.json, so a bump re-runs episodes whose frames were
# drawn by older renderer code instead of silently reusing them.
RENDER3D_VERSION = "1"
