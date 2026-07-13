"""Multiturn chat must not ALSO embed the last-3 window into every stored turn.

With chat_history=rolling/full the prior turns ARE the history; building each
user turn with an embedded "Recent history" section (context_window=last3)
duplicates every observation 2-4x in context — the same failure family as the
one-shot ICL duplication that collapsed Claude to 7%. The runner comment
promised "lean user turns"; these tests make that true.
"""

from __future__ import annotations

from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.task_spec import TaskSpecification
from interface.config import ExperimentConfig
from interface.runner import build_runner
from prompting_experiments.prompt_templates.observation import RECENT_HISTORY_HEADER


class RecordingAgent:
    """Scripted step_by_step agent that keeps every message list it was sent."""

    def __init__(self, actions):
        self._actions = list(actions)
        self._i = 0
        self.calls: list[list[dict]] = []
        self.last_usage = {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}

    def __call__(self, messages):
        self.calls.append(messages)
        action = self._actions[self._i] if self._i < len(self._actions) else "DONE"
        self._i += 1
        return f"FINAL_OUTPUT: {action}"


SPEC = {
    "task_id": "history_dedup",
    "seed": 0,
    "difficulty_tier": 1,
    "maze": {"dimensions": [7, 7], "walls": [], "start": [1, 1], "goal": [5, 1]},
    "mechanisms": {},
    "goal": {"type": "reach_position", "target": [5, 1]},
    "max_steps": 20,
}


def _run(agent, **config_kwargs):
    spec = TaskSpecification.from_dict(SPEC)
    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(spec)
    runner = build_runner(ExperimentConfig(**config_kwargs), backend, spec)
    return runner.run(agent, verbose=False)


def _texts(messages):
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            yield content
        elif isinstance(content, list):
            for block in content:
                if block.get("type") == "text":
                    yield block.get("text", "")


def test_rolling_chat_turns_do_not_embed_last3_history():
    agent = RecordingAgent(["MOVE_FORWARD"] * 4)
    _run(
        agent,
        observation="text_only",
        context_window="last3",
        chat_history="rolling",
        chat_turns_max=3,
    )
    assert len(agent.calls) >= 4
    # In multiturn mode the chat carries the history; no turn may re-embed it.
    for call in agent.calls:
        for text in _texts(call):
            assert RECENT_HISTORY_HEADER not in text


def test_stateless_still_embeds_last3_history():
    agent = RecordingAgent(["MOVE_FORWARD"] * 4)
    _run(
        agent,
        observation="text_only",
        context_window="last3",
        chat_history="stateless",
    )
    late_call_texts = "\n".join(_texts(agent.calls[-1]))
    assert RECENT_HISTORY_HEADER in late_call_texts
