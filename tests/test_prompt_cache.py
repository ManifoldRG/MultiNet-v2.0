from __future__ import annotations

import json
import os

import numpy as np
import pytest

from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.task_spec import TaskSpecification
from interface.agents.claude import (
    ClaudeAnthropicAgent,
    ClaudeAnthropicConfig,
    _resolve_cache_markers,
)
from interface.agents.kimi_k26 import _build_body, _to_openai_messages
from interface.agents.qwen_vllm import _to_openai_content
from interface.agents.runner_messages import parse_runner_content, strip_cache_markers
from interface.config import ExperimentConfig
from interface.loader import default_maze_path, load_task
from interface.observation import history_content_blocks
from interface.renderer import rgb_to_image_block
from interface.runner import build_runner
from interface.telemetry import normalize_token_usage
from prompting_experiments.prompt_templates import feedback as feedback_templates


def test_markers_become_cache_control():
    turns = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "a", "_cache_breakpoint": True},
                {"type": "text", "text": "tail"},
            ],
        }
    ]
    system, out = _resolve_cache_markers("sys", turns, ttl="5m", enabled=True)
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    assert out[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in out[0]["content"][1]
    assert not any("_cache_breakpoint" in block for block in out[0]["content"])


def test_disabled_strips_markers_and_adds_nothing():
    turns = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "a", "_cache_breakpoint": True}],
        }
    ]
    system, out = _resolve_cache_markers("sys", turns, ttl="5m", enabled=False)
    assert system == "sys"
    assert out[0]["content"] == [{"type": "text", "text": "a"}]


def test_last_assistant_turn_gets_breakpoint():
    turns = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ]
    _, out = _resolve_cache_markers(None, turns, ttl="1h", enabled=True)
    assert out[1]["content"][0]["cache_control"] == {
        "type": "ephemeral",
        "ttl": "1h",
    }


def test_claude_sync_and_batch_cache_ttls_are_distinct(monkeypatch):
    import interface.agents.claude as claude_module

    captured = {}

    def _capture_post(_api_key, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(claude_module, "_post_messages", _capture_post)
    agent = ClaudeAnthropicAgent(
        config=ClaudeAnthropicConfig(
            enable_prompt_cache=True,
            cache_ttl="5m",
            batch_cache_ttl="1h",
        ),
        api_key="test-key",
    )
    messages = [
        {"role": "system", "content": "sys"},
        {
            "role": "user",
            "content": [{"type": "text", "text": "prefix", "cache_breakpoint": True}],
        },
    ]

    agent.generate(messages)
    assert captured["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert captured["messages"][0]["content"][0]["cache_control"] == {
        "type": "ephemeral"
    }
    assert "_cache_breakpoint" not in captured["messages"][0]["content"][0]
    batch_params = agent._params_for(messages)
    assert batch_params["system"][0]["cache_control"] == {
        "type": "ephemeral",
        "ttl": "1h",
    }
    assert batch_params["messages"][0]["content"][0]["cache_control"] == {
        "type": "ephemeral",
        "ttl": "1h",
    }


def test_cache_breakpoint_budget_is_enforced():
    turns = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": str(i), "_cache_breakpoint": True}
                for i in range(4)
            ],
        },
        {"role": "assistant", "content": "answer"},
    ]
    try:
        _resolve_cache_markers("sys", turns, ttl="5m", enabled=True)
    except ValueError as exc:
        assert "6 cache breakpoints requested" in str(exc)
    else:
        raise AssertionError("expected the Anthropic breakpoint limit to be enforced")


def test_nested_openai_cache_fields_are_not_folded_into_input_tokens():
    usage = normalize_token_usage(
        {
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "prompt_tokens_details": {
                "cached_tokens": 80,
                "cache_write_tokens": 10,
            },
        }
    )
    assert usage == {
        "input_tokens": 100,
        "output_tokens": 5,
        "total_tokens": 105,
        "cache_read_input_tokens": 80,
        "cache_creation_input_tokens": 10,
    }


def test_anthropic_cache_creation_ttl_breakdown_is_preserved():
    usage = normalize_token_usage(
        {
            "input_tokens": 10,
            "output_tokens": 2,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 7,
                "ephemeral_1h_input_tokens": 3,
            },
        }
    )
    assert usage["cache_creation_5m_input_tokens"] == 7
    assert usage["cache_creation_1h_input_tokens"] == 3


def test_runner_parsing_retains_markers_for_anthropic():
    parsed = parse_runner_content(
        [{"type": "text", "text": "stable", "cache_breakpoint": True}]
    )
    assert parsed[0].cache_breakpoint is True


def test_qwen_converters_do_not_forward_cache_markers():
    blocks = [
        {"type": "text", "text": "stable", "cache_breakpoint": True},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="},
         "cache_breakpoint": True},
    ]
    converted = _to_openai_content(blocks)
    assert converted == [
        {"type": "text", "text": "stable"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,AA=="},
        },
    ]
    assert _to_openai_messages(
        [{"role": "user", "content": strip_cache_markers(blocks)}]
    )[0]["content"] == converted


def test_kimi_cache_ttl_is_sync_only_and_markers_are_removed():
    messages = _to_openai_messages(
        [{"role": "user", "content": [{"type": "text", "text": "x",
                                       "cache_breakpoint": True}]}]
    )
    assert messages == [{"role": "user", "content": [{"type": "text", "text": "x"}]}]
    shared = {
        "model": "kimi-k2.6",
        "max_tokens": 100,
        "enable_thinking": False,
    }
    sync_body = _build_body(messages, **shared, cache_ttl="1h")
    batch_body = _build_body(messages, **shared, include_sampling=False)
    assert sync_body["prompt_cache_options"] == {"mode": "implicit", "ttl": "1h"}
    assert "prompt_cache_options" not in batch_body


def test_run_config_plumbs_provider_cache_options(monkeypatch):
    from scripts.run_pipeline import _build_agent_from_spec

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    claude, _ = _build_agent_from_spec(
        "claude",
        {
            "provider": "claude",
            "enable_prompt_cache": False,
            "cache_ttl": "1h",
            "batch_cache_ttl": "5m",
        },
    )
    kimi, _ = _build_agent_from_spec(
        "kimi",
        {"provider": "kimi", "prompt_cache_ttl": "1h"},
    )
    assert claude.config.enable_prompt_cache is False
    assert claude.config.cache_ttl == "1h"
    assert claude.config.batch_cache_ttl == "5m"
    assert kimi.config.prompt_cache_ttl == "1h"


def test_history_blocks_only_get_markers_when_enabled():
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    transcript = [
        {
            "kind": "step",
            "event_type": "VALID",
            "action": "MOVE_FORWARD",
            "position_after_row_col": (1, 2),
            "facing_after": "EAST",
            "prompt_feedback": "continue",
            "state_before": {"inventory": []},
            "_decision_frame_rgb": rgb,
        }
    ]
    disabled = history_content_blocks(
        "image_only", "full", transcript, cache_breakpoints=False
    )
    enabled = history_content_blocks(
        "image_only", "full", transcript, cache_breakpoints=True
    )
    assert not any("cache_breakpoint" in block for block in disabled)
    assert enabled[-1]["cache_breakpoint"] is True


def _simple_spec() -> TaskSpecification:
    return TaskSpecification.from_dict(
        {
            "task_id": "prompt_cache_stability",
            "seed": 0,
            "difficulty_tier": 1,
            "maze": {
                "dimensions": [6, 6],
                "walls": [],
                "start": [1, 1],
                "goal": [4, 4],
            },
            "mechanisms": {},
            "goal": {"type": "reach_position", "target": [4, 4]},
            "max_steps": 6,
        }
    )


class _ForwardAgent:
    def __call__(self, _messages):
        return "FINAL_OUTPUT: MOVE_FORWARD"


def _episode_messages(observation: str) -> list[list[dict]]:
    spec = _simple_spec()
    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(spec)
    runner = build_runner(
        ExperimentConfig(
            observation=observation,
            context_window="full",
            chat_history="stateless",
            prompt_cache=True,
        ),
        backend,
        spec,
    )
    result = runner.run(_ForwardAgent(), verbose=False)
    return [
        record["agent_messages"]
        for record in result["transcript"]
        if record.get("kind") == "query"
    ]


def _first_divergence(left: list[dict], right: list[dict]) -> int:
    for index, (left_block, right_block) in enumerate(zip(left, right)):
        if left_block != right_block:
            return index
    return min(len(left), len(right))


def test_history_prefix_stays_stable_for_cached_image_observations():
    for observation in ("image_only", "image_text"):
        queries = _episode_messages(observation)
        assert len(queries) >= 3
        user_n = next(message for message in queries[1] if message["role"] == "user")
        user_n1 = next(message for message in queries[2] if message["role"] == "user")
        blocks_n = user_n["content"]
        blocks_n1 = user_n1["content"]
        marked = [
            index
            for index, block in enumerate(blocks_n)
            if isinstance(block, dict) and block.get("cache_breakpoint")
        ]
        assert marked, f"{observation} history did not include a cache marker"
        stable_prefix_length = marked[-1] + 1
        stripped_n = strip_cache_markers(blocks_n)
        stripped_n1 = strip_cache_markers(blocks_n1)
        assert _first_divergence(stripped_n, stripped_n1) >= stable_prefix_length


def test_disabling_prompt_cache_preserves_the_unmarked_message_bytes():
    backend, spec = load_task(default_maze_path())
    backend_cached, spec_cached = load_task(default_maze_path())
    transcript = [
        {
            "kind": "step",
            "event_type": "VALID",
            "action": "MOVE_FORWARD",
            "position_after_row_col": (1, 2),
            "facing_after": "EAST",
            "prompt_feedback": "continue",
            "state_before": {"inventory": []},
            "_decision_frame_rgb": np.zeros((2, 2, 3), dtype=np.uint8),
        }
    ]
    disabled = build_runner(
        ExperimentConfig(
            observation="image_only",
            context_window="full",
            in_context_learning="zero_shot",
            prompt_cache=False,
        ),
        backend,
        spec,
    )
    enabled = build_runner(
        ExperimentConfig(
            observation="image_only",
            context_window="full",
            in_context_learning="zero_shot",
            prompt_cache=True,
        ),
        backend_cached,
        spec_cached,
    )
    disabled.last_rgb, state, _ = backend.reset(seed=spec.seed)
    enabled.last_rgb, state_cached, _ = backend_cached.reset(seed=spec_cached.seed)
    message_disabled = disabled._build_message(
        state, feedback_templates.INITIAL_FEEDBACK, transcript
    )
    message_enabled = enabled._build_message(
        state_cached, feedback_templates.INITIAL_FEEDBACK, transcript
    )
    assert json.dumps(message_disabled, separators=(",", ":")) == json.dumps(
        {"role": "user", "content": strip_cache_markers(message_enabled["content"])},
        separators=(",", ":"),
    )


def test_image_png_encoding_is_deterministic_and_non_mutating():
    rgb = np.arange(4 * 5 * 3, dtype=np.uint8).reshape((4, 5, 3))
    before = rgb.copy()
    first = rgb_to_image_block(rgb)
    second = rgb_to_image_block(rgb)
    assert first == second
    np.testing.assert_array_equal(rgb, before)


def test_stored_last_rgb_is_not_mutated_by_later_steps():
    spec = _simple_spec()
    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(spec)
    runner = build_runner(
        ExperimentConfig(
            observation="image_only",
            context_window="current",
            chat_history="stateless",
            in_context_learning="zero_shot",
        ),
        backend,
        spec,
    )
    previous_frame = None
    previous_snapshot = None
    calls = 0

    class _FrameCheckingAgent:
        def __call__(self, _messages):
            nonlocal previous_frame, previous_snapshot, calls
            if previous_frame is not None:
                np.testing.assert_array_equal(previous_frame, previous_snapshot)
            previous_frame = runner.last_rgb
            previous_snapshot = previous_frame.copy()
            calls += 1
            return "FINAL_OUTPUT: TURN_RIGHT"

    runner.run(_FrameCheckingAgent(), verbose=False)
    assert calls >= 2
    np.testing.assert_array_equal(previous_frame, previous_snapshot)


@pytest.mark.skipif(
    os.environ.get("RUN_ANTHROPIC_CACHE_SMOKE") != "1"
    or not os.environ.get("ANTHROPIC_API_KEY"),
    reason="set RUN_ANTHROPIC_CACHE_SMOKE=1 and ANTHROPIC_API_KEY for live billing smoke",
)
def test_live_sonnet_three_turn_cache_smoke():
    agent = ClaudeAnthropicAgent(
        config=ClaudeAnthropicConfig(
            model="claude-sonnet-4-6",
            max_tokens=32,
            cache_ttl="5m",
            enable_prompt_cache=True,
        )
    )
    stable_system = "Stable cache-prefix validation text. " * 2_000
    messages = [
        {"role": "system", "content": stable_system},
        {"role": "user", "content": "Turn one. Reply briefly."},
    ]
    first = agent.generate(messages)
    assert first.usage is not None
    messages.extend(
        [
            {"role": "assistant", "content": first.text},
            {"role": "user", "content": "Turn two. Reply briefly."},
        ]
    )
    second = agent.generate(messages)
    assert second.usage is not None
    assert second.usage.get("cache_read_input_tokens", 0) > 0
    messages.extend(
        [
            {"role": "assistant", "content": second.text},
            {"role": "user", "content": "Turn three. Reply briefly."},
        ]
    )
    third = agent.generate(messages)
    assert third.usage is not None
