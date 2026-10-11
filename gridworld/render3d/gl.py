"""The MUJOCO_GL default, shared by the renderer and the test bootstrap.

No mujoco import here: the default must be in place before the first
``import mujoco``, which reads MUJOCO_GL once.
"""

from __future__ import annotations

import os
import sys


def default_mujoco_gl() -> None:
    """On Linux with MUJOCO_GL unset, use osmesa (software rendering, works
    without a GPU or display). Anywhere else leave it alone: mujoco accepts
    osmesa/egl only on Linux, and an unset value lets it pick the platform's
    native GL. An explicit MUJOCO_GL (e.g. egl for GPU speed) always wins."""
    if sys.platform.startswith("linux") and "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "osmesa"
