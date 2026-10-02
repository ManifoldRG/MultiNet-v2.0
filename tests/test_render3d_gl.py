"""MUJOCO_GL defaults: osmesa only on Linux and only when unset.

mujoco accepts osmesa/egl only on Linux; anywhere else that value makes
``import mujoco`` raise RuntimeError (which pytest.importorskip does not catch).
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from render3d_test_utils import REPO_ROOT


def _helper():
    from gridworld.render3d.gl import default_mujoco_gl

    return default_mujoco_gl


@pytest.mark.parametrize("platform", ["linux", "linux2"])
def test_linux_without_mujoco_gl_defaults_to_osmesa(monkeypatch, platform):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    _helper()()
    assert os.environ["MUJOCO_GL"] == "osmesa"


def test_an_explicit_mujoco_gl_is_never_overridden(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("MUJOCO_GL", "egl")
    _helper()()
    assert os.environ["MUJOCO_GL"] == "egl"


@pytest.mark.parametrize("platform", ["darwin", "win32", "cygwin"])
def test_off_linux_mujoco_gl_stays_unset(monkeypatch, platform):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    _helper()()
    assert "MUJOCO_GL" not in os.environ


def test_importing_the_renderer_off_linux_leaves_mujoco_gl_unset():
    """The renderer's import-time default goes through the helper: on macOS it
    must not plant a value mujoco rejects. Everything else is imported first so
    only the renderer's own top-level code sees the faked platform."""
    pytest.importorskip("mujoco")
    code = textwrap.dedent(
        """
        import os, sys
        import numpy, PIL.Image
        import gridworld.backends.base, gridworld.task_spec
        import gridworld.render3d.cameras, gridworld.render3d.hud
        import gridworld.render3d.scene, gridworld.render3d.sync
        import mujoco
        sys.platform = "darwin"
        import gridworld.render3d.renderer
        print(repr(os.environ.get("MUJOCO_GL")))
        """
    )
    env = {k: v for k, v in os.environ.items() if k != "MUJOCO_GL"}
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert result.stdout.strip() == "None"


def test_renderer_and_conftest_share_the_one_default():
    """Two copies of the default could disagree; only the helper sets it."""
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in [REPO_ROOT / "conftest.py", *sorted((REPO_ROOT / "gridworld").rglob("*.py"))]
        if path.name != "gl.py" and 'setdefault("MUJOCO_GL"' in path.read_text()
    ]
    assert not offenders, offenders
    assert "default_mujoco_gl()" in (REPO_ROOT / "conftest.py").read_text()
    assert "default_mujoco_gl()" in (REPO_ROOT / "gridworld" / "render3d" / "renderer.py").read_text()
