"""3D and common code must not depend on minigrid (spec §2 import rule).

Runtime checks run in a subprocess with every ``minigrid`` import blocked.
"""

from __future__ import annotations

import pytest

from render3d_test_utils import run_with_minigrid_blocked

COMMON_MODULES = [
    "gridworld.backends",
    "gridworld.backends.base",
    "interface.loader",
    "interface.runner",
    "interface.episode_step",
    "interface.episode_checkpoint",
]


def test_blocker_actually_blocks_minigrid():
    result = run_with_minigrid_blocked(
        "from gridworld.backends import get_backend\nget_backend('minigrid')"
    )
    assert result.returncode != 0
    assert "minigrid blocked" in result.stderr


@pytest.mark.parametrize("module", COMMON_MODULES)
def test_common_module_imports_without_minigrid(module):
    result = run_with_minigrid_blocked(f"import {module}")
    assert result.returncode == 0, result.stderr[-3000:]


def test_lazy_minigrid_backend_name_still_resolves():
    from gridworld.backends import MiniGridBackend
    from gridworld.backends.minigrid_backend import MiniGridBackend as Direct

    assert MiniGridBackend is Direct


def test_load_task_default_backend_is_minigrid(tmp_path):
    import json

    from gridworld.backends.minigrid_backend import MiniGridBackend
    from interface.loader import load_task
    from render3d_test_utils import CORRIDOR

    path = tmp_path / "corridor.json"
    path.write_text(json.dumps(CORRIDOR))
    backend, spec = load_task(path)
    assert isinstance(backend, MiniGridBackend)
    assert backend.is_configured and spec.task_id == "render3d_corridor"


import importlib.util

DEMO_MODULES = ["demo.theme", "demo.session", "demo.fx", "demo.api.view"]


@pytest.mark.parametrize("module", DEMO_MODULES)
def test_demo_module_imports_without_minigrid(module):
    result = run_with_minigrid_blocked(f"import {module}")
    assert result.returncode == 0, result.stderr[-3000:]


@pytest.mark.skipif(importlib.util.find_spec("fastapi") is None, reason="web extra not installed")
def test_demo_web_app_imports_without_minigrid():
    result = run_with_minigrid_blocked("import demo.api.app")
    assert result.returncode == 0, result.stderr[-3000:]
