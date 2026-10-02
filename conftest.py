"""Pytest bootstrap: make repo-root packages importable in any invocation.

Several test modules import repo-root packages that are deliberately not
packaged/installed (e.g. `deploy` — operator tooling excluded from the wheel
in v2.0.0). Bare `pytest` does not put the rootdir on sys.path, so without
this the suite only collects when an alphabetically-early test module happens
to insert it first.
"""
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from gridworld.render3d.gl import default_mujoco_gl  # noqa: E402

# Headless MuJoCo rendering. Must be set before the first `import mujoco` in
# the test process; on Linux it defaults to OSMesa, which works on CPU-only
# machines (CI). Export MUJOCO_GL=egl yourself for GPU speed.
default_mujoco_gl()
