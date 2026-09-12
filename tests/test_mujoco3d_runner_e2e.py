"""A scripted agent drives the real runner over the 3D backend: same executed
actions and step count as 2D, and the model is shown the 3D frame."""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("minigrid")

from gridworld.backends import get_backend  # noqa: E402
from gridworld.backends.minigrid_backend import MiniGridBackend  # noqa: E402
from gridworld.baselines import plan_bfs_path  # noqa: E402
from gridworld.task_spec import TaskSpecification  # noqa: E402
from interface.config import ExperimentConfig  # noqa: E402
from interface.parser import ACTION_ORDER  # noqa: E402
from interface.runner import build_runner  # noqa: E402
from render3d_test_utils import MECHANISMS  # noqa: E402


class ScriptedAgent:
    """Returns one scripted action per query (step_by_step)."""

    def __init__(self, actions):
        self._actions = list(actions)
        self._i = 0
        self.last_usage = {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}

    def __call__(self, messages):
        action = self._actions[self._i] if self._i < len(self._actions) else "DONE"
        self._i += 1
        return f"FINAL_OUTPUT: {action}"


def _run(backend, spec, tokens):
    backend.configure(spec)
    runner = build_runner(ExperimentConfig(observation="image_only"), backend, spec)
    return runner, runner.run(ScriptedAgent(tokens), verbose=False)


def _executed(result):
    return [e["action"] for e in result["transcript"] if e.get("kind") == "step"]


def test_scripted_agent_plays_identically_on_3d_and_2d():
    spec = TaskSpecification.from_dict(MECHANISMS)
    plan = plan_bfs_path(spec)
    assert plan.success
    tokens = [ACTION_ORDER[a] for a in plan.actions]

    runner3d, result3d = _run(get_backend("mujoco3d", camera="chase", resolution=128), spec, tokens)
    _, result2d = _run(MiniGridBackend(render_mode="rgb_array"), spec, tokens)

    assert result3d["success"] is True
    assert result3d["steps_used"] == result2d["steps_used"] == len(plan.actions)
    assert _executed(result3d) == _executed(result2d) == tokens
    assert runner3d.last_rgb.shape == (128, 128, 3)  # the frame the model saw was the 3D render
