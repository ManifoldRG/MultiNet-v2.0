from __future__ import annotations

import json
import logging
import os

import numpy as np
import pytest

from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.task_spec import TaskSpecification
from interface.agents.claude import (
    ClaudeAnthropicAgent,
    ClaudeAnthropicConfig,
    _cache_control,
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
from scripts.run_pipeline import (
    _NON_RUNTIME_MODEL_KEYS,
    _experiment_config_hash_payload,
    _experiment_config_payload,
    _expected_run_hash,
    _validate_cache_ttls,
)


@pytest.mark.parametrize("ttl", ["10m", ""])
def test_cache_control_rejects_unsupported_ttl(ttl):
    with pytest.raises(
        ValueError,
        match=r"cache ttl must be one of \('5m', '1h'\)",
    ):
        _cache_control(ttl)


@pytest.mark.parametrize(
    ("ttl", "expected"),
    [
        ("5m", {"type": "ephemeral"}),
        ("1h", {"type": "ephemeral", "ttl": "1h"}),
    ],
)
def test_cache_control_returns_supported_ttl(ttl, expected):
    assert _cache_control(ttl) == expected


@pytest.mark.parametrize("ttl_key", ["cache_ttl", "batch_cache_ttl"])
def test_claude_agent_rejects_unsupported_cache_ttl_at_construction(ttl_key):
    with pytest.raises(ValueError, match="cache ttl"):
        ClaudeAnthropicAgent(
            config=ClaudeAnthropicConfig(**{ttl_key: "10m"}),
            api_key="x",
        )


def test_claude_agent_warns_when_batch_deadline_exceeds_cache_ttl(caplog):
    with caplog.at_level(logging.WARNING, logger="interface.agents.claude"):
        ClaudeAnthropicAgent(api_key="x")

    assert "write premium without reads" in caplog.text
    assert "2.00x" in caplog.text and "1.25x" in caplog.text


@pytest.mark.parametrize(
    "config",
    [
        ClaudeAnthropicConfig(batch_deadline_s=3600),
        ClaudeAnthropicConfig(enable_prompt_cache=False),
    ],
)
def test_claude_agent_does_not_warn_when_cache_lifetime_is_adequate_or_disabled(
    config, caplog
):
    with caplog.at_level(logging.WARNING, logger="interface.agents.claude"):
        ClaudeAnthropicAgent(config=config, api_key="x")

    assert "write premium without reads" not in caplog.text


def test_claude_batch_round_cache_usage_warnings(monkeypatch, caplog):
    import interface.agents.claude as claude_module

    usage_by_round = [
        {"cache_creation_input_tokens": 12, "cache_read_input_tokens": 0},
        {"cache_creation_input_tokens": 8, "cache_read_input_tokens": 0},
        {"cache_creation_input_tokens": 5, "cache_read_input_tokens": 20},
    ]
    call_index = 0

    def fake_run_message_batch(_requests, **_kwargs):
        nonlocal call_index
        usage = usage_by_round[call_index]
        call_index += 1
        return {
            "i0": {
                "type": "succeeded",
                "message": {
                    "content": [{"type": "text", "text": "FINAL_OUTPUT: MOVE_FORWARD"}],
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 2,
                        **usage,
                    },
                    "stop_reason": "end_turn",
                },
            }
        }

    monkeypatch.setattr(claude_module, "run_message_batch", fake_run_message_batch)
    agent = ClaudeAnthropicAgent(
        config=ClaudeAnthropicConfig(batch_deadline_s=3600),
        api_key="x",
    )
    batch = [[{"role": "user", "content": "go"}]]
    with caplog.at_level(logging.WARNING, logger="interface.agents.claude"):
        agent.generate_batch(batch)
        assert "wrote prompt cache but read none" not in caplog.text
        caplog.clear()
        agent.generate_batch(batch)
        assert "wrote prompt cache but read none" in caplog.text
        caplog.clear()
        agent.generate_batch(batch)
        assert "wrote prompt cache but read none" not in caplog.text


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


@pytest.mark.parametrize(
    ("model", "enable_thinking"), [("kimi-k2.6", False), ("kimi-k3", True)]
)
def test_kimi_cache_ttl_is_sync_only_and_markers_are_removed(model, enable_thinking):
    messages = _to_openai_messages(
        [{"role": "user", "content": [{"type": "text", "text": "x",
                                       "cache_breakpoint": True}]}]
    )
    assert messages == [{"role": "user", "content": [{"type": "text", "text": "x"}]}]
    shared = {
        "model": model,
        "max_tokens": 100,
        "enable_thinking": enable_thinking,
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



@pytest.mark.parametrize(
    ("provider", "experiment", "entry", "expected"),
    [
        ("openai", {"prompt_cache": True}, {}, True),
        ("openai", {}, {}, False),
        ("openai", {"prompt_cache": True}, {"enable_prompt_cache": False}, False),
        ("claude", {"prompt_cache": False}, {}, False),
        ("kimi", {"prompt_cache": True}, {}, None),
    ],
)
def test_fleet_worker_takes_prompt_cache_from_the_experiment(
    monkeypatch, tmp_path, provider, experiment, entry, expected
):
    """Units ship the raw model entry; the sync worker must derive Claude/OpenAI
    enable_prompt_cache from the unit's experiment_config like the local path
    (an explicit entry wins), or Astra runs in OpenAI's write-only implicit mode."""
    import scripts.distributed_run_pipeline as drp

    monkeypatch.setattr(drp.pipeline, "_run_one_unit", lambda *a, **kw: ({"ok": True}, 1.0))
    monkeypatch.setattr(drp, "materialize_worker_inputs", lambda unit, root: {})
    monkeypatch.setattr(drp.ScorerConfig, "from_dict", staticmethod(lambda d: object()))
    built = []

    def factory(model_key, model_config):
        built.append(model_config.get("enable_prompt_cache"))
        return (lambda messages: "x"), "label"

    unit = {
        "unit_id": "u0", "model_key": "m", "model_id": "mid",
        "model_config": {"provider": provider, **entry},
        "task_id": "t", "seed": 0, "prompt_variant": "default", "scorer_config": {},
        "difficulty_max_static_score": 1000.0,
        "task_artifacts": {"scored_static": {}},
        "experiment_config": experiment,
    }
    drp.run_assigned_unit(unit, artifacts_root=tmp_path, agent_factory=factory)
    assert built == [expected]
    assert unit["model_config"] == {"provider": provider, **entry}  # not mutated


@pytest.mark.parametrize("context_window", ["last_n", "text_summary_and_last_n", "full"])
def test_history_breakpoints_only_apply_to_full_context(context_window):
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    transcript = [
        {"kind": "query"},
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
    blocks = history_content_blocks(
        "image_only",
        context_window,
        transcript,
        cache_breakpoints=True,
    )
    markers = [block for block in blocks if "cache_breakpoint" in block]
    if context_window == "full":
        assert markers
        assert all(block["type"] == "text" for block in markers)
        assert "Full history" not in markers[0]["text"]
    else:
        assert markers == []


def _history_step(index: int) -> dict:
    return {
        "kind": "step",
        "event_type": "VALID",
        "action": f"MOVE_{index}",
        "position_after_row_col": (1, index + 1),
        "facing_after": "EAST",
        "prompt_feedback": f"feedback-{index}",
        "state_before": {"inventory": []},
        "_decision_frame_rgb": np.full((2, 2, 3), index, dtype=np.uint8),
    }


def _marked_history_texts(blocks: list[dict]) -> list[dict]:
    return [
        block
        for block in blocks
        if isinstance(block, dict)
        and block.get("type") == "text"
        and block.get("cache_breakpoint")
    ]


def test_one_step_per_query_marks_last_two_history_steps():
    transcript = [{"kind": "query"}, _history_step(0)]
    transcript.extend(({"kind": "query"}, _history_step(1)))

    blocks = history_content_blocks(
        "image_only", "full", transcript, cache_breakpoints=True
    )

    marked_texts = _marked_history_texts(blocks)
    assert len(marked_texts) == 2
    assert "MOVE_0" in marked_texts[0]["text"]
    assert "MOVE_1" in marked_texts[1]["text"]




def test_subgoal_cache_markers_cover_previous_and_current_boundaries():
    transcript = [_history_step(0), {"kind": "query"}]
    transcript.extend(_history_step(index) for index in range(1, 13))

    blocks = history_content_blocks(
        "image_only", "full", transcript, cache_breakpoints=True
    )

    marked_texts = _marked_history_texts(blocks)
    marked_actions = [
        block["text"].split("FINAL_OUTPUT: MOVE_", 1)[1].splitlines()[0]
        for block in marked_texts
    ]
    marked_actions = [int(action) for action in marked_actions]
    assert marked_actions == [0, 12]
    assert marked_actions[-1] - marked_actions[0] > 10


@pytest.mark.parametrize("steps_per_query", range(1, 31))
def test_cache_prefix_reuses_a_previous_breakpoint(steps_per_query):
    transcript = [_history_step(0)]
    previous_last_prefix = None
    query_number = 0

    for _ in range(3):
        blocks = history_content_blocks(
            "image_only", "full", transcript, cache_breakpoints=True
        )
        marked_indices = [
            index
            for index, block in enumerate(blocks)
            if isinstance(block, dict) and block.get("cache_breakpoint")
        ]
        stripped = strip_cache_markers(blocks)
        prefixes = [stripped[: index + 1] for index in marked_indices]
        assert prefixes
        if previous_last_prefix is not None:
            assert previous_last_prefix in prefixes
        previous_last_prefix = prefixes[-1]

        transcript.append({"kind": "query", "query_index": query_number})
        transcript.extend(
            _history_step(1 + query_number * steps_per_query + offset)
            for offset in range(steps_per_query)
        )
        query_number += 1


@pytest.mark.parametrize(
    ("key", "value"),
    [
        (key, value)
        for key in ("cache_ttl", "batch_cache_ttl", "prompt_cache_ttl")
        for value in (None, "5m", "1h")
    ],
)
def test_validate_cache_ttls_accepts_supported_values(key, value):
    _validate_cache_ttls({key: value})


@pytest.mark.parametrize(
    ("key", "value"),
    [
        (key, value)
        for key in ("cache_ttl", "batch_cache_ttl", "prompt_cache_ttl")
        for value in ("2h", "30m", 5)
    ],
)
def test_validate_cache_ttls_rejects_unsupported_values(key, value):
    with pytest.raises(ValueError) as exc_info:
        _validate_cache_ttls({key: value})

    message = str(exc_info.value)
    assert key in message
    assert "5m" in message and "1h" in message


def test_cache_options_are_non_runtime_model_keys():
    assert {
        "enable_prompt_cache",
        "cache_ttl",
        "batch_cache_ttl",
        "prompt_cache_ttl",
    } <= _NON_RUNTIME_MODEL_KEYS


def test_prompt_cache_model_options_do_not_change_run_hash():
    common = {
        "provider": "claude",
        "model": "claude-test",
        "temperature": 0,
    }
    first = {
        **common,
        "enable_prompt_cache": False,
        "cache_ttl": "5m",
        "batch_cache_ttl": "1h",
        "prompt_cache_ttl": None,
    }
    second = {
        **common,
        "enable_prompt_cache": True,
        "cache_ttl": "1h",
        "batch_cache_ttl": "5m",
        "prompt_cache_ttl": "1h",
    }

    def expected_hash(model_config: dict) -> str:
        return _expected_run_hash(
            _simple_spec(),
            "claude-test",
            0,
            "minigrid",
            model_config=model_config,
        )

    assert expected_hash(first) == expected_hash(second)


def test_experiment_prompt_cache_is_not_part_of_run_hash_payload():
    enabled = ExperimentConfig(prompt_cache=True)
    disabled = ExperimentConfig(prompt_cache=False)
    hashed_payload = _experiment_config_hash_payload(enabled)

    assert "prompt_cache" not in hashed_payload
    assert _experiment_config_payload(enabled)["prompt_cache"] is True
    assert _expected_run_hash(
        _simple_spec(), "claude-test", 0, "minigrid", experiment_config=enabled
    ) == _expected_run_hash(
        _simple_spec(), "claude-test", 0, "minigrid", experiment_config=disabled
    )


def test_history_record_inventory_is_computed_once_per_step(monkeypatch):
    import interface.observation as observation_module

    calls = 0
    original = observation_module._history_record_inventory

    def counted_inventory(record):
        nonlocal calls
        calls += 1
        return original(record)

    monkeypatch.setattr(
        observation_module, "_history_record_inventory", counted_inventory
    )
    transcript = [
        {
            "kind": "step",
            "event_type": "VALID",
            "action": f"MOVE_{index}",
            "position_after_row_col": (1, index + 1),
            "facing_after": "EAST",
            "state_before": {"inventory": []},
            "_decision_frame_rgb": np.zeros((2, 2, 3), dtype=np.uint8),
        }
        for index in range(3)
    ]

    history_content_blocks("image_only", "last_n", transcript)

    assert calls == 3



@pytest.mark.parametrize("context_window", ["last_n", "text_summary_and_last_n", "full"])
def test_runner_marks_history_only_for_full_context_window(context_window):
    spec = _simple_spec()
    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(spec)
    config = ExperimentConfig(
        observation="image_only",
        context_window=context_window,
        context_n=2,
        prompt_cache=True,
        in_context_learning="zero_shot",
    )
    runner = build_runner(config, backend, spec)
    runner.last_rgb, state, _info = backend.reset(seed=spec.seed)
    transcript = [
        {
            "kind": "step",
            "event_type": "VALID",
            "action": f"MOVE_{index}",
            "position_after_row_col": (1, index + 2),
            "facing_after": "EAST",
            "prompt_feedback": f"feedback-{index}",
            "state_before": {"inventory": []},
            "_decision_frame_rgb": np.zeros((2, 2, 3), dtype=np.uint8),
        }
        for index in range(4)
    ]

    message = runner._build_message(
        state,
        feedback_templates.INITIAL_FEEDBACK,
        transcript,
        with_one_shot=False,
    )
    marked_blocks = [
        block
        for block in message["content"]
        if isinstance(block, dict) and "cache_breakpoint" in block
    ]

    if context_window == "full":
        assert marked_blocks
    else:
        assert marked_blocks == []


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
