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


import ast

from render3d_test_utils import REPO_ROOT

FORBIDDEN = (
    "minigrid",
    "gridworld.custom_env",
    "gridworld.task_parser",
    "gridworld.backends.minigrid_backend",
)
THREE_D_FILES = sorted((REPO_ROOT / "gridworld" / "render3d").glob("*.py")) + [
    REPO_ROOT / "gridworld" / "backends" / "mujoco3d_backend.py"
]


def _imported_modules(path):
    package = ".".join(path.relative_to(REPO_ROOT).with_suffix("").parts[:-1])
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = importlib.util.resolve_name("." * node.level + (node.module or ""), package)
            else:
                base = node.module
            yield base
            for alias in node.names:
                yield f"{base}.{alias.name}"


def test_all_3d_files_are_scanned():
    names = {p.name for p in THREE_D_FILES}
    assert {"palette.py", "cameras.py", "scene.py", "sync.py", "renderer.py", "mujoco3d_backend.py"} <= names


@pytest.mark.parametrize("path", THREE_D_FILES, ids=lambda p: p.name)
def test_3d_code_never_imports_minigrid(path):
    bad = [m for m in _imported_modules(path) if any(m == f or m.startswith(f + ".") for f in FORBIDDEN)]
    assert not bad, f"{path.name} imports {bad}"


@pytest.mark.skipif(importlib.util.find_spec("mujoco") is None, reason="mujoco3d extra not installed")
def test_3d_backend_renders_with_minigrid_blocked_on_a_stub_state_backend():
    result = run_with_minigrid_blocked(
        '''
        import sys
        sys.path.insert(0, "tests")
        from gridworld.backends.base import AbstractGridBackend, GridState
        from gridworld.backends.mujoco3d_backend import Mujoco3DBackend
        from gridworld.task_spec import TaskSpecification
        from render3d_test_utils import CORRIDOR

        class StubStateBackend(AbstractGridBackend):
            def configure(self, task_spec):
                self.task_spec = task_spec
                self._configured = True
            def _state(self):
                return GridState(agent_position=(1, 1), agent_direction=0, key_positions={"k1": (2, 1)})
            def reset(self, seed=None):
                return None, self._state(), {}
            def step(self, action):
                return None, 0.0, False, False, self._state(), {}
            def render(self):
                raise NotImplementedError
            def get_mission_text(self):
                return ""
            def get_state(self):
                return self._state()

        backend = Mujoco3DBackend(StubStateBackend(), camera="chase", resolution=64)
        backend.configure(TaskSpecification.from_dict(CORRIDOR))
        frame, _state, _info = backend.reset(seed=0)
        assert frame.shape == (64, 64, 3), frame.shape
        backend.close()
        '''
    )
    assert result.returncode == 0, result.stderr[-3000:]
