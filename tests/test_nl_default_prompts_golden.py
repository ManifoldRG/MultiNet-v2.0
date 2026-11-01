"""Default NL prompts stay byte-identical to what R2 sent.

``tests/fixtures/nl_default_prompts_M10.json`` was captured from origin/main
4b8b54d, before the ``tile_labels`` option existed. The scripted route walks
M10 onto a death tile and then solves it, so every tile mechanic fires (two
freezes, a death reset, both teleporters, turntable landings and rides). Each
query's messages are hashed under four text cells: the R2 NL cell, plus json,
ascii, standard/causal feedback and the text summary, which share the renderer
and feedback code the option touches. The R2 cell also keeps its system prompt
and first and last user messages verbatim, so a failure shows a readable diff.

Regenerate only for an intentional prompt change:
    python -m tests.test_nl_default_prompts_golden --write
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import deque
from pathlib import Path

from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.task_spec import TaskSpecification
from gridworld.world_model import TaskPlanningContext, successors
from interface.config import ExperimentConfig
from interface.runner import build_runner

_HERE = Path(__file__).resolve().parent
MAZE = _HERE / "fixtures" / "mechanics" / "M10_18x18_dense_kr_sg_kb_portals_death_rotating_frozen_0.json"
GOLDEN = _HERE / "fixtures" / "nl_default_prompts_M10.json"
ACTION_NAMES = ["TURN_LEFT", "TURN_RIGHT", "MOVE_FORWARD", "PICKUP", "DROP", "TOGGLE", "DONE"]

# The R2 NL cell (run_config.mechanics_15_nl.json) and three cells that cover
# the other text formats, feedback levels and history modes.
CELLS: dict[str, dict] = {
    "r2_nl": dict(
        prompting="minimal", observation="text_only", context_window="full",
        chat_history="stateless", in_context_learning="zero_shot", action_space="egocentric",
        querying="step_by_step", progress_stall_k=30, prompt_cache=True,
    ),
    "json_last_n_standard": dict(
        prompting="standard", observation="text_only", observation_text_format="json",
        context_window="last_n", in_context_learning="zero_shot", feedback="standard",
    ),
    "ascii_full_causal": dict(
        prompting="verbose", observation="text_only", observation_text_format="ascii",
        context_window="full", in_context_learning="zero_shot", feedback="causal",
        progress_stall_k=30,
    ),
    "coords_summary_minimal": dict(
        prompting="standard", observation="text_only", context_window="text_summary_and_last_n",
        in_context_learning="zero_shot",
    ),
}


def _bfs(ctx, start, done) -> list[int]:
    prev = {start: None}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        for t in successors(ctx, state):
            if done(t):
                path, cur = [t.action], state
                while prev[cur] is not None:
                    cur, action = prev[cur]
                    path.append(action)
                return path[::-1]
            if t.next_state not in prev:
                prev[t.next_state] = (state, t.action)
                queue.append(t.next_state)
    raise AssertionError("no route")


def death_detour_route(spec: TaskSpecification) -> list[str]:
    """BFS onto the nearest death tile, then BFS to the goal (the M10 trace route)."""
    ctx = TaskPlanningContext(spec)
    start = ctx.initial_state()
    goal = tuple(spec.resolved_goal())
    plan = _bfs(ctx, start, lambda t: t.label == "kill_reset")
    plan += _bfs(ctx, start, lambda t: t.next_state.agent_pos == goal and t.action == 2)
    return [ACTION_NAMES[a] for a in plan]


class Scripted:
    """Replays a fixed action list and keeps every request it was sent."""

    def __init__(self, actions):
        self._actions, self._i = list(actions), 0
        self.requests: list[list[dict]] = []
        self.last_usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}

    def __call__(self, messages):
        self.requests.append(json.loads(json.dumps(messages)))
        action = self._actions[self._i] if self._i < len(self._actions) else "DONE"
        self._i += 1
        return f"FINAL_OUTPUT: {action}"


def run_scripted(spec: TaskSpecification, config: ExperimentConfig, actions: list[str]):
    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(spec)
    agent = Scripted(actions)
    result = build_runner(config, backend, spec).run(agent, verbose=False)
    return result, agent.requests


def _digest(messages) -> str:
    blob = json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def capture() -> dict:
    spec = TaskSpecification.from_json(str(MAZE))
    route = death_detour_route(spec)
    out: dict = {"captured_from": "origin/main 4b8b54d", "route_length": len(route), "cells": {}}
    for name, overrides in CELLS.items():
        result, requests = run_scripted(spec, ExperimentConfig(**overrides), route)
        steps = [r for r in result["transcript"] if r.get("kind") == "step"]
        cell = {
            "end_reason": result["end_reason"],
            "query_sha256": [_digest(r) for r in requests],
            "prompt_feedback": [s["prompt_feedback"] for s in steps],
        }
        if name == "r2_nl":
            cell["system_prompt"] = requests[0][0]["content"]
            cell["first_user_message"] = requests[0][1]["content"]
            cell["last_user_message"] = requests[-1][1]["content"]
        out["cells"][name] = cell
    return out


def test_default_nl_prompts_match_main_byte_for_byte():
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    got = capture()
    assert got["route_length"] == golden["route_length"] == 90
    for name, want in golden["cells"].items():
        have = got["cells"][name]
        assert have["end_reason"] == want["end_reason"] == "success", name
        for key in ("system_prompt", "first_user_message", "last_user_message"):
            if key in want:
                assert have[key] == want[key], (name, key)
        assert have["prompt_feedback"] == want["prompt_feedback"], name
        first_diff = next(
            (i for i, (a, b) in enumerate(zip(have["query_sha256"], want["query_sha256"])) if a != b),
            None,
        )
        assert first_diff is None, f"{name}: query {first_diff + 1} differs from main"
        assert len(have["query_sha256"]) == len(want["query_sha256"]), name


if __name__ == "__main__":
    if "--write" not in sys.argv:
        raise SystemExit("usage: python -m tests.test_nl_default_prompts_golden --write")
    GOLDEN.write_text(json.dumps(capture(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {GOLDEN}")
