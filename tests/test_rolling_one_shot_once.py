"""Rolling multi-turn history must carry the one-shot ICL example exactly once.

Regression: the rolling/full history builder prepended the one-shot example to
every user turn; as turns accumulated the example was re-sent every step, bloating
tokens ~4-5x and swamping the observation (Claude collapsed into loops, failing
even empty_room). The example must appear on the current turn only, with lean
user turns stored in history.
"""

from __future__ import annotations

from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.task_spec import TaskSpecification
from interface.config import ExperimentConfig
from interface.runner import build_runner
from prompting_experiments.prompt_templates import user as user_templates

MARKER = user_templates.ONE_SHOT_EXAMPLE_INTRO


class ScriptedAgent:
    def __init__(self, actions):
        self._actions = list(actions)
        self._i = 0
        self.last_usage = {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}

    def __call__(self, messages):
        action = self._actions[self._i] if self._i < len(self._actions) else "DONE"
        self._i += 1
        return f"FINAL_OUTPUT: {action}"


def _user_turns_with_one_shot(agent_messages) -> int:
    n = 0
    for m in agent_messages:
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, list):
            if any(
                isinstance(b, dict)
                and b.get("type") == "text"
                and MARKER in b.get("text", "")
                for b in content
            ):
                n += 1
        elif isinstance(content, str) and MARKER in content:
            n += 1
    return n


SPEC = {
    "task_id": "empty_room_rolling",
    "seed": 0,
    "difficulty_tier": 1,
    "maze": {"dimensions": [6, 6], "walls": [], "start": [1, 1], "goal": [4, 4]},
    "mechanisms": {},
    "goal": {"type": "reach_position", "target": [4, 4]},
    "max_steps": 20,
}


def _run(**cfg):
    spec = TaskSpecification.from_dict(SPEC)
    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(spec)
    runner = build_runner(ExperimentConfig(**cfg), backend, spec)
    # Several non-terminal actions so multi-turn history accumulates past 1 turn.
    return runner.run(ScriptedAgent(["MOVE_FORWARD"] * 6), verbose=False)


def _queries(result):
    return [e for e in result["transcript"] if e.get("kind") == "query"]


def test_rolling_history_carries_one_shot_example_exactly_once():
    result = _run(
        observation="image_text",
        in_context_learning="one_shot",
        chat_history="rolling",
        context_window="current",
        chat_turns_max=3,
    )
    queries = _queries(result)
    assert len(queries) >= 4, "need several turns to exercise multi-turn history"
    for q in queries:
        count = _user_turns_with_one_shot(q["agent_messages"])
        assert count == 1, (
            f"query {q['query_index']}: one-shot example on {count} user turns (want 1)"
        )
    # History genuinely accumulated multiple user turns by the last query.
    last = queries[-1]["agent_messages"]
    assert len([m for m in last if m.get("role") == "user"]) >= 2


def test_stateless_history_still_has_one_shot_example():
    result = _run(
        observation="image_text",
        in_context_learning="one_shot",
        chat_history="stateless",
        context_window="last3",
    )
    for q in _queries(result):
        assert _user_turns_with_one_shot(q["agent_messages"]) == 1
