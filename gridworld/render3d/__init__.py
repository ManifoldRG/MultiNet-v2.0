"""3D (MuJoCo) rendering of task-spec mazes from a GridState.

Imports only the common layer (task_spec, GridState) plus numpy/mujoco --
never minigrid or the MiniGrid backend (tests/test_backend_import_boundary.py).

MUJOCO_GL defaults to osmesa (software rendering, works without a GPU); the
default is set in renderer.py, the only module here that imports mujoco.
"""
