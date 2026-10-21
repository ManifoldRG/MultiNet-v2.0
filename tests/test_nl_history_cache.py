"""text_only full history goes out one text block per step so cache breakpoints
can sit on the newest steps, while every provider still reads the same text."""
from __future__ import annotations

from unittest import mock

import pytest

from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.task_spec import TaskSpecification
from interface.agents.claude import _resolve_cache_markers, _to_anthropic_turns
from interface.agents.kimi_k26 import _to_openai_messages as kimi_messages
from interface.agents.openai_agent import _to_openai_messages as openai_messages
from interface.agents.qwen35_vl import _to_qwen_content
from interface.agents.qwen_vllm import _to_openai_content as qwen_vllm_content
from interface.agents.reply import Reply
from interface.agents.runner_messages import join_split_text, strip_cache_markers
from interface.config import ExperimentConfig
from interface.episode_checkpoint import resume_stepper, save_checkpoint
from interface.episode_step import EpisodeStepper
from interface.observation import history_text, history_text_blocks
from interface.runner import _split_off_history, build_runner

# Turns, moves, a wall bump and an unparseable reply, so the history carries
# every kind of step feedback.
_REPLIES = [
    "FINAL_OUTPUT: MOVE_FORWARD",
    "FINAL_OUTPUT: TURN_RIGHT",
    "FINAL_OUTPUT: MOVE_FORWARD",
    "no action here",
    "FINAL_OUTPUT: MOVE_FORWARD",
    "FINAL_OUTPUT: MOVE_FORWARD",
    "FINAL_OUTPUT: MOVE_FORWARD",
    "FINAL_OUTPUT: TURN_LEFT",
    "FINAL_OUTPUT: PICKUP",
    "FINAL_OUTPUT: MOVE_FORWARD",
    "FINAL_OUTPUT: TURN_LEFT",
    "FINAL_OUTPUT: MOVE_FORWARD",
]


def _spec(max_steps: int = 40) -> TaskSpecification:
    return TaskSpecification.from_dict(
        {
            "task_id": "nl_history_cache",
            "seed": 0,
            "difficulty_tier": 1,
            "maze": {"dimensions": [7, 7], "walls": [[3, 2]], "start": [1, 1], "goal": [5, 5]},
            "mechanisms": {},
            "goal": {"type": "reach_position", "target": [5, 5]},
            "max_steps": max_steps,
        }
    )


def _runner(max_steps: int = 40, **overrides):
    spec = _spec(max_steps)
    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(spec)
    fields = {
        "observation": "text_only",
        "context_window": "full",
        "chat_history": "stateless",
        "in_context_learning": "zero_shot",
        "prompt_cache": True,
        "progress_stall_k": 30,
        **overrides,
    }
    return build_runner(ExperimentConfig(**fields), backend, spec)


def _unsplit_user_message(runner, stepper) -> dict:
    """The user message as main built it: text_only history inside one text."""
    with mock.patch("interface.runner.history_text_blocks", return_value=[]):
        return runner._build_message(
            stepper.state,
            stepper.last_feedback,
            stepper.transcript,
            stall_remaining=(
                stepper._stall_watchdog.remaining if stepper._stall_watchdog else None
            ),
        )


def _drive(runner, replies=_REPLIES, *, checkpoint_at=None, tmp_path=None):
    """[(request messages, unsplit user message)] per query; with
    ``checkpoint_at`` also the checkpoint path written before that query's reply."""
    stepper = EpisodeStepper(runner, verbose=False)
    stepper.start()
    queries = []
    checkpoint = None
    for index, reply in enumerate(replies):
        messages = stepper.next_query()
        if messages is None:
            break
        queries.append((messages, _unsplit_user_message(runner, stepper)))
        if index == checkpoint_at:
            checkpoint = tmp_path / "episode.ckpt.json"
            save_checkpoint(checkpoint, stepper)
        stepper.apply_reply(Reply(text=reply))
    return queries, checkpoint


def _user(messages: list[dict]) -> dict:
    return next(message for message in messages if message["role"] == "user")


@pytest.mark.parametrize("text_format", ["coords", "ascii", "json"])
@pytest.mark.parametrize("icl", ["zero_shot", "one_shot"])
@pytest.mark.parametrize("prompt_cache", [True, False])
def test_split_history_joins_to_the_unsplit_message(text_format, icl, prompt_cache):
    queries, _ = _drive(
        _runner(
            observation_text_format=text_format,
            in_context_learning=icl,
            prompt_cache=prompt_cache,
        )
    )
    assert len(queries) == len(_REPLIES)
    for messages, unsplit in queries[1:]:
        content = _user(messages)["content"]
        assert isinstance(content, list) and len(content) > 2
        assert join_split_text(content) == unsplit["content"]
        assert strip_cache_markers(content) == unsplit["content"]
    # The first query has no history and is unchanged.
    assert _user(queries[0][0]) == queries[0][1]


def test_caching_on_and_off_send_the_same_blocks_but_for_markers():
    on, _ = _drive(_runner(prompt_cache=True))
    off, _ = _drive(_runner(prompt_cache=False))
    for (messages_on, _), (messages_off, _) in zip(on, off):
        on_blocks = _user(messages_on)["content"]
        off_blocks = _user(messages_off)["content"]
        if isinstance(on_blocks, str):
            assert on_blocks == off_blocks
            continue
        assert [
            {k: v for k, v in block.items() if k != "cache_breakpoint"} for block in on_blocks
        ] == off_blocks
        assert not any("cache_breakpoint" in block for block in off_blocks)


def test_providers_without_block_breakpoints_send_the_unsplit_text():
    for icl in ("zero_shot", "one_shot"):
        queries, _ = _drive(_runner(in_context_learning=icl))
        for messages, unsplit in queries:
            expected = unsplit["content"]
            assert _user(kimi_messages(messages))["content"] == expected
            content = _user(messages)["content"]
            assert strip_cache_markers(content) == expected  # Qwen served-vLLM API
            if isinstance(expected, str):
                assert qwen_vllm_content(content) == expected
                assert _to_qwen_content(content) == expected


def _anthropic_user_blocks(messages: list[dict]) -> tuple[object, list[dict]]:
    system, turns = _to_anthropic_turns(messages)
    system_out, turns_out = _resolve_cache_markers(system, turns, ttl="5m", enabled=True)
    return system_out, turns_out[-1]["content"]


def _cached_prefix_end(blocks: list[dict]) -> int:
    return max(index for index, block in enumerate(blocks) if "cache_control" in block)


def _without_cache_control(block: dict) -> dict:
    return {k: v for k, v in block.items() if k != "cache_control"}


def test_claude_caches_every_step_and_reads_the_previous_prefix():
    queries, _ = _drive(_runner())
    previous = None
    for messages, unsplit in queries[1:]:
        system, blocks = _anthropic_user_blocks(messages)
        assert "cache_control" in system[0]
        assert "".join(block["text"] for block in blocks) == unsplit["content"]
        assert not any("text_continues" in block for block in blocks)
        marks = sum("cache_control" in block for block in blocks)
        assert 1 + marks <= 4
        end = _cached_prefix_end(blocks)
        # Everything but the current turn is cached: the last block alone
        # carries the state, budget/stall lines and the question.
        assert end == len(blocks) - 2
        assert "What is your next action?" in blocks[-1]["text"]
        assert "What is your next action?" not in "".join(b["text"] for b in blocks[:-1])
        if previous is not None:
            prev_end = _cached_prefix_end(previous)
            # This request repeats the previous cached prefix byte for byte and
            # marks its end, so the read is exact.
            assert [_without_cache_control(b) for b in blocks[: prev_end + 1]] == [
                _without_cache_control(b) for b in previous[: prev_end + 1]
            ]
            assert "cache_control" in blocks[prev_end]
        previous = blocks


def test_openai_gets_split_blocks_with_breakpoints_and_no_runner_keys():
    queries, _ = _drive(_runner())
    messages, unsplit = queries[-1]
    for prompt_cache in (True, False):
        out = openai_messages(messages, image_detail=None, prompt_cache=prompt_cache)
        blocks = _user(out)["content"]
        assert "".join(block["text"] for block in blocks) == unsplit["content"]
        assert not any("text_continues" in b or "cache_breakpoint" in b for b in blocks)
        marked = [i for i, b in enumerate(blocks) if "prompt_cache_breakpoint" in b]
        if prompt_cache:
            assert marked and max(marked) == len(blocks) - 2
        else:
            assert marked == []


def test_previous_request_end_is_marked_when_a_query_runs_many_steps():
    replies = ["SUB_GOAL: east\nFINAL_OUTPUT: " + ", ".join(["TURN_LEFT"] * 23)] * 3
    # No stall limit: turning in place makes no progress.
    queries, _ = _drive(
        _runner(max_steps=80, querying="subgoal", progress_stall_k=None), replies
    )
    assert len(queries) == 3
    _, first = _anthropic_user_blocks(queries[1][0])
    _, second = _anthropic_user_blocks(queries[2][0])
    # 23 new step blocks: beyond Anthropic's 20-block lookback from the newest
    # step, so the read relies on the previous request's end being marked.
    assert len(second) - len(first) == 23
    assert "cache_control" in second[_cached_prefix_end(first)]


def test_checkpoint_resume_reissues_the_identical_request(tmp_path):
    queries, checkpoint = _drive(_runner(), checkpoint_at=6, tmp_path=tmp_path)
    resumed = resume_stepper(checkpoint, runner=_runner())
    assert resumed.next_query() == queries[6][0]


def test_split_blocks_join_to_history_text():
    runner = _runner()
    stepper = EpisodeStepper(runner, verbose=False)
    stepper.start()
    for reply in _REPLIES:
        stepper.next_query()
        stepper.apply_reply(Reply(text=reply))
    for text_format in ("coords", "ascii", "json"):
        kwargs = dict(observation_text_format=text_format, include_facing=True)
        blocks = history_text_blocks(
            "text_only", "full", stepper.transcript, runner.task_spec, **kwargs
        )
        assert blocks and all(block["text_continues"] for block in blocks)
        assert "".join(block["text"] for block in blocks) == history_text(
            "text_only", "full", stepper.transcript, runner.task_spec, **kwargs
        )


@pytest.mark.parametrize(
    ("observation", "context_window"),
    [
        ("text_only", "last_n"),
        ("text_only", "text_summary_and_last_n"),
        ("text_only", "current"),
        ("image_text", "full"),
        ("image_only", "full"),
    ],
)
def test_only_text_only_full_history_is_split(observation, context_window):
    transcript = [
        {
            "kind": "step",
            "event_type": "VALID",
            "action": "MOVE_FORWARD",
            "position_before_row_col": (1, 1),
            "facing_before": "EAST",
            "prompt_feedback": "Moved.",
        }
    ]
    assert history_text_blocks(observation, context_window, transcript) == []


def test_prompt_not_opening_with_the_history_is_sent_whole():
    blocks = [{"type": "text", "text": "Full history (oldest first):", "text_continues": True}]
    assert _split_off_history("Something else first", blocks) is None
    assert _split_off_history("anything", []) is None
    split = _split_off_history("Full history (oldest first):\n\nQ?", blocks)
    assert split == blocks + [{"type": "text", "text": "\n\nQ?"}]


def test_prompt_preview_shows_the_text_as_sent():
    from prompting_experiments.preview_prompts import _content_to_text

    messages, unsplit = _drive(_runner())[0][-1]
    assert _content_to_text(_user(messages)["content"]) == unsplit["content"]


def test_join_split_text_keeps_other_blocks_in_place():
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    content = [
        {"type": "text", "text": "a", "text_continues": True},
        image,
        {"type": "text", "text": "b", "text_continues": True, "cache_breakpoint": True},
        {"type": "text", "text": "c"},
    ]
    assert join_split_text(content) == [
        {"type": "text", "text": "a"},
        image,
        {"type": "text", "text": "bc"},
    ]
    unflagged = [{"type": "text", "text": "x"}, {"type": "text", "text": "y"}]
    assert join_split_text(unflagged) is unflagged
    assert join_split_text("plain") == "plain"
