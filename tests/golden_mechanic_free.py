"""Golden prompts/feedback for mazes WITHOUT teleporters, death tiles or ice.

Mechanic feedback (TELEPORTED / DIED / FROZEN) must leave every mechanic-free
episode byte-identical: same step event types, same feedback strings, same
prompt text. This module plays four R1-panel mazes (vendored under
``tests/fixtures/golden_mechanic_free/mazes``; plain, key/door, switch/gate and
a distractor-heavy mix) through the real runner with a scripted agent and
records the text the model would have seen.

Each episode opens with a few deliberate mistakes (PICKUP / TOGGLE / DONE on
an empty cell, a MOVE_FORWARD into the north border wall, an unparseable
reply) so failure counting (queue clears, WRONG_DONE, BLOCKED) is pinned too,
then follows the BFS solution (``plan_bfs_path``) to the goal.

Regenerate (only from code whose mechanic-free behaviour is known-good):

    python tests/golden_mechanic_free.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridworld.actions import MiniGridActions
from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.baselines import plan_bfs_path
from gridworld.task_spec import TaskSpecification
from interface.config import ExperimentConfig
from interface.runner import build_runner

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "golden_mechanic_free"
MAZE_DIR = FIXTURE_DIR / "mazes"
GOLDEN_PATH = FIXTURE_DIR / "golden.json"

MAZES = (
    "S4_10x10_dense_1",
    "M1_8x8_corridor_kr_0",
    "M2_8x8_corridor_sg_0",
    "D2_8x8_corridor_wrong_ky_inactive_sb_sg_kr_1",
)

# The R1 fixed cell (gridworld/fixtures/run_config.r1.json) plus two variants
# that put step feedback text into the prompt at the other feedback levels.
_R1_CELL = {
    "prompting": "minimal",
    "observation": "image_only",
    "context_window": "text_summary_and_last_n",
    "chat_history": "stateless",
    "in_context_learning": "zero_shot",
    "action_space": "egocentric",
    "querying": "step_by_step",
    "progress_stall_k": 30,
}
CONFIGS = {
    "r1_cell": dict(_R1_CELL),
    "image_text_standard": {**_R1_CELL, "observation": "image_text", "feedback": "standard"},
    "text_only_causal": {**_R1_CELL, "observation": "text_only", "feedback": "causal"},
}

# Every maze here starts at (x, y) = (1, 1) facing EAST, so TURN_LEFT faces
# the north border wall. ``None`` is an unparseable reply.
MISTAKES = ("PICKUP", "TOGGLE", "DONE", "TURN_LEFT", "MOVE_FORWARD", "TURN_RIGHT", None)

_ACTION_NAMES = {int(a): a.name for a in MiniGridActions}


class _ScriptedAgent:
    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self._i = 0
        self.last_usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}

    def __call__(self, messages):
        reply = self._replies[self._i] if self._i < len(self._replies) else "FINAL_OUTPUT: DONE"
        self._i += 1
        return reply


def _replies(spec: TaskSpecification) -> list[str]:
    plan = plan_bfs_path(spec)
    assert plan.success, spec.task_id
    actions = list(MISTAKES) + [_ACTION_NAMES[a] for a in plan.actions]
    return [
        "I am not sure what to do." if a is None else f"FINAL_OUTPUT: {a}"
        for a in actions
    ]


def _text_parts(content) -> list[str]:
    if isinstance(content, str):
        return [content]
    parts = []
    for block in content:
        if block.get("type") == "text":
            parts.append(block["text"])
        else:
            parts.append(f"<{block.get('type')}>")
    return parts


def play(maze: str, config_name: str) -> dict:
    spec = TaskSpecification.from_json(str(MAZE_DIR / f"{maze}.json"))
    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(spec)
    runner = build_runner(ExperimentConfig(**CONFIGS[config_name]), backend, spec)
    result = runner.run(_ScriptedAgent(_replies(spec)), verbose=False)

    # Stateless chat: every query is [system, user] and the system prompt never
    # changes, so it is stored once per episode.
    system = None
    user_turns = []
    for rec in result["transcript"]:
        if rec.get("kind") != "query":
            continue
        sys_msg, user_msg = rec["agent_messages"]
        assert sys_msg["role"] == "system" and user_msg["role"] == "user"
        sys_parts = _text_parts(sys_msg["content"])
        assert system in (None, sys_parts), "system prompt changed mid-episode"
        system = sys_parts
        user_turns.append(_text_parts(user_msg["content"]))
    steps = [
        {
            "action": rec["action"],
            "event_type": rec["event_type"],
            "feedback": rec["feedback"],
            "consecutive_failures_after": rec["consecutive_failures_after"],
        }
        for rec in result["transcript"]
        if rec.get("kind") == "step"
    ]
    return {
        "end_reason": result["end_reason"],
        "success": result["success"],
        "steps_used": result["steps_used"],
        "query_count": result["query_count"],
        "steps": steps,
        "system": system,
        "user_turns": user_turns,
    }


def capture() -> dict:
    return {
        f"{maze}/{config_name}": play(maze, config_name)
        for maze in MAZES
        for config_name in CONFIGS
    }


def main() -> None:
    GOLDEN_PATH.write_text(
        json.dumps(capture(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {GOLDEN_PATH}")


if __name__ == "__main__":
    main()
