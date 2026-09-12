"""3D (MuJoCo) rendering of task-spec mazes from a GridState.

Imports only the common layer (task_spec, GridState) plus numpy/mujoco --
never minigrid or the MiniGrid backend (tests/test_backend_import_boundary.py).
"""

import os

# Must precede the first `import mujoco` anywhere in the process. OSMesa is
# software rendering (works without a GPU); set MUJOCO_GL=egl for speed.
os.environ.setdefault("MUJOCO_GL", "osmesa")
