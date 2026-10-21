"""Every prompt states the moves remaining exactly once, in the current turn,
before the question and the FINAL_OUTPUT instruction: image_only included.

The number is the cap the environment enforces, state.max_steps -
state.step_count. With a stall limit the stall countdown appears once too.
"""

from __future__ import annotations

import dataclasses
import re

import pytest

from interface.config import ExperimentConfig
from interface.loader import default_maze_path, load_task
from interface.runner import build_runner
from prompting_experiments.prompt_templates import querying as querying_templates
from prompting_experiments.prompt_templates import user as user_templates

MAX_STEPS = 6
MOVES = re.compile(r'Moves remaining: (\d+)|"moves_remaining": (\d+)')
STALL = re.compile(r"Moves remaining until stall: (\d+)")

MODES = [
    ("text_only", "coords"), ("text_only", "ascii"), ("text_only", "json"),
    ("image_text", "coords"), ("image_text", "ascii"), ("image_text", "json"),
    ("image_only", "coords"),
]
WINDOWS = ["last_n", "text_summary_and_last_n", "full"]


class _Turner:
    """Turns in place every step and keeps every query it was sent."""

    def __init__(self) -> None:
        self.queries: list[list[dict]] = []

    def __call__(self, messages: list[dict]) -> str:
        self.queries.append(list(messages))  # rolling chat grows the list in place
        return "FINAL_OUTPUT: TURN_LEFT"


def _queries(cfg: ExperimentConfig) -> list[list[dict]]:
    backend, spec = load_task(default_maze_path())
    spec = dataclasses.replace(spec, max_steps=MAX_STEPS)
    backend.configure(spec)
    agent = _Turner()
    build_runner(cfg, backend, spec).run(agent, verbose=False)
    assert len(agent.queries) == MAX_STEPS
    return agent.queries


def _text(message: dict) -> str:
    content = message["content"]
    if isinstance(content, str):
        return content
    return "\n".join(b["text"] for b in content if b.get("type") == "text")


def _numbers(pattern: re.Pattern, text: str) -> list[int]:
    return [int(next(g for g in m.groups() if g is not None)) for m in pattern.finditer(text)]


def _cfg(observation, fmt, window, stall_k, **overrides) -> ExperimentConfig:
    return ExperimentConfig(
        observation=observation, observation_text_format=fmt, context_window=window,
        in_context_learning="zero_shot", progress_stall_k=stall_k, **overrides,
    )


def _check_current_turn(text: str, step: int, stall_k: int | None) -> int | None:
    assert _numbers(MOVES, text) == [MAX_STEPS - step]
    # No second budget line in other wording ("Step X of Y (N remaining).").
    assert text.count("remaining") == 1 + (stall_k is not None)
    question = text.index(user_templates.NEXT_ACTION_QUESTION)
    instruction = text.index(querying_templates.SINGLE_ACTION_FINAL_OUTPUT_INSTRUCTION)
    assert MOVES.search(text).start() < question < instruction
    stall = _numbers(STALL, text)
    if stall_k is None:
        assert stall == []
        return None
    assert len(stall) == 1 and 0 < stall[0] <= stall_k
    assert STALL.search(text).start() < question
    return stall[0]


@pytest.mark.parametrize("stall_k", [None, 30])
@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("observation,fmt", MODES)
def test_one_moves_remaining_line_in_the_current_turn(observation, fmt, window, stall_k):
    current = _queries(_cfg(observation, fmt, "current", stall_k))
    windowed = _queries(_cfg(observation, fmt, window, stall_k))
    for step, (cur, win) in enumerate(zip(current, windowed)):
        cur_text, win_text = _text(cur[-1]), _text(win[-1])
        _check_current_turn(cur_text, step, stall_k)
        # The history window only prepends; the current turn is unchanged.
        assert win_text.endswith(cur_text)
        if fmt == "coords":
            # History steps never repeat the budget (ascii/json history snapshots
            # carry their own past state, budget included, by design).
            assert len(_numbers(MOVES, win_text)) == 1


def test_image_only_matches_the_text_modes_step_for_step():
    budgets = {}
    for observation in ("text_only", "image_text", "image_only"):
        queries = _queries(_cfg(observation, "coords", "full", 30))
        budgets[observation] = [
            (_numbers(MOVES, _text(q[-1])), _numbers(STALL, _text(q[-1]))) for q in queries
        ]
    assert budgets["image_only"] == budgets["text_only"] == budgets["image_text"]


def test_rolling_chat_current_turn_has_one_line():
    cfg = _cfg("image_only", "coords", "current", 30, chat_history="rolling")
    for step, messages in enumerate(_queries(cfg)):
        _check_current_turn(_text(messages[-1]), step, 30)


def test_no_step_x_of_y_line():
    for messages in _queries(_cfg("text_only", "coords", "full", None)):
        assert not re.search(r"Step \d+ of \d+", _text(messages[-1]))
