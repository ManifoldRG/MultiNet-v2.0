"""Natural-language wrapper around the standard MiniGrid backend."""

from __future__ import annotations

from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.task_spec import TaskSpecification

from .nl_action_parser import NLActionParser


class NLGridWorldEnv:
    """Execute short text commands against a gridworld task."""

    def __init__(self, task_spec: TaskSpecification, render_mode: str = "rgb_array"):
        self.task_spec = task_spec
        self.backend = MiniGridBackend(render_mode=render_mode)
        self.parser = NLActionParser()
        self._last_obs = None
        self._last_state = None
        self._last_info = {}

    def reset(self, seed: int | None = None):
        """Reset the wrapped environment and return observation plus metadata."""
        self.backend.configure(self.task_spec)
        obs, state, info = self.backend.reset(seed=seed)
        self._last_obs = obs
        self._last_state = state
        self._last_info = info
        return obs, {"mission": self.backend.get_mission_text(), **info}

    def step(self, command: str):
        """Parse and execute one natural-language command."""
        if self._last_state is None:
            self.reset(seed=self.task_spec.seed)

        parsed_actions = self.parser.parse(command, agent_facing=self._last_state.agent_direction)
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for action in parsed_actions:
            obs, step_reward, terminated, truncated, state, info = self.backend.step(action)
            self._last_obs = obs
            self._last_state = state
            reward += step_reward
            if terminated or truncated:
                break

        return self._last_obs, reward, terminated, truncated, {
            **info,
            "parsed_actions": parsed_actions,
        }

    def close(self) -> None:
        """Release backend resources."""
        self.backend.close()
