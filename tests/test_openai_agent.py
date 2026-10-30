from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

import pytest

from interface.agents import openai_agent
from interface.agents.openai_agent import (
    OpenAIAgent,
    OpenAIConfig,
    OpenAISpendCapReached,
    SPEND,
    _build_body,
    _to_openai_messages,
    price_for,
    usage_cost_usd,
)
from interface.telemetry import normalize_token_usage, token_count_from_record
from scripts.run_pipeline import _build_agent_from_spec


class _FakeResponse:
    def __init__(self, payload, headers=None):
        self._payload = payload
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


_COMPLETION = {
    "choices": [{"message": {"content": "FINAL_OUTPUT: MOVE_FORWARD"}, "finish_reason": "stop"}],
    "usage": {
        "prompt_tokens": 1000,
        "completion_tokens": 400,
        "total_tokens": 1400,
        "prompt_tokens_details": {"cached_tokens": 200},
        "completion_tokens_details": {"reasoning_tokens": 350},
    },
    "service_tier": "flex",
}

_MESSAGES = [
    {"role": "system", "content": "system prompt"},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
        ],
    },
]


@pytest.fixture(autouse=True)
def _reset_spend():
    SPEND.reset()
    yield
    SPEND.reset()


def _capture(monkeypatch, payload=_COMPLETION, headers=None):
    seen = {"calls": 0}

    def fake_urlopen(req, timeout):
        seen["calls"] += 1
        seen["url"] = req.full_url
        seen["authorization"] = req.get_header("Authorization")
        seen["timeout"] = timeout
        seen["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse(payload, headers)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return seen


def test_request_body_is_reasoning_model_shaped(monkeypatch):
    seen = _capture(monkeypatch)
    agent = OpenAIAgent(
        OpenAIConfig(
            model="gpt-6-astra",
            max_tokens=64000,
            reasoning_effort="xhigh",
            service_tier="flex",
            image_detail="high",
            timeout=1800,
        ),
        api_key="secret",
    )
    reply = agent.generate(_MESSAGES)

    assert seen["url"] == "https://api.openai.com/v1/chat/completions"
    assert seen["authorization"] == "Bearer secret"
    assert seen["timeout"] == 1800
    body = seen["body"]
    assert body["model"] == "gpt-6-astra"
    assert body["max_completion_tokens"] == 64000
    assert body["reasoning_effort"] == "xhigh"
    assert body["service_tier"] == "flex"
    # Reasoning models reject sampling params and the legacy cap name.
    for banned in ("temperature", "top_p", "max_tokens"):
        assert banned not in body
    image_block = body["messages"][1]["content"][1]
    assert image_block["image_url"] == {"url": "data:image/png;base64,abc123", "detail": "high"}
    # The runner's message list is not mutated by the detail injection.
    assert "detail" not in _MESSAGES[1]["content"][1]["image_url"]
    assert reply.text == "FINAL_OUTPUT: MOVE_FORWARD"


def test_usage_surfaces_reasoning_and_cached_tokens(monkeypatch):
    _capture(monkeypatch)
    reply = OpenAIAgent(OpenAIConfig(), api_key="k").generate(_MESSAGES)
    assert reply.usage["input_tokens"] == 1000
    assert reply.usage["output_tokens"] == 400
    assert reply.usage["reasoning_tokens"] == 350
    assert reply.usage["cached_input_tokens"] == 200
    assert reply.token_truncated is False
    # Re-normalizing downstream must not fold cached tokens into input again.
    assert token_count_from_record({"usage": reply.usage}) == 1400


def test_length_finish_is_token_truncated(monkeypatch):
    payload = {
        "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 128},
    }
    _capture(monkeypatch, payload)
    reply = OpenAIAgent(OpenAIConfig(max_tokens=128), api_key="k").generate(_MESSAGES)
    assert reply.text == ""
    assert reply.token_truncated is True


def test_rate_limit_headers_are_recorded(monkeypatch):
    _capture(monkeypatch, headers={"x-ratelimit-limit-tokens": "500000", "other": "x"})
    agent = OpenAIAgent(OpenAIConfig(), api_key="k")
    agent(_MESSAGES)
    assert agent.last_rate_limits == {"x-ratelimit-limit-tokens": "500000"}
    assert agent.last_usage["output_tokens"] == 400


def test_insufficient_quota_is_not_retried(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(req, timeout):
        calls["n"] += 1
        raise urllib.error.HTTPError(
            req.full_url, 429, "Too Many Requests", {},
            io.BytesIO(b'{"error": {"code": "insufficient_quota"}}'),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    agent = OpenAIAgent(OpenAIConfig(max_attempts=5), api_key="k")
    with pytest.raises(RuntimeError, match="insufficient_quota"):
        agent.generate(_MESSAGES)
    assert calls["n"] == 1


def test_bad_request_reports_body(monkeypatch):
    def fake_urlopen(req, timeout):
        raise urllib.error.HTTPError(
            req.full_url, 400, "Bad Request", {}, io.BytesIO(b'{"error": "bad effort"}')
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="HTTP 400.*bad effort"):
        OpenAIAgent(OpenAIConfig(), api_key="k").generate(_MESSAGES)


def test_spend_cap_blocks_after_budget_spent(monkeypatch):
    seen = _capture(monkeypatch)
    cfg = OpenAIConfig(model="gpt-6-astra", service_tier="flex", spend_cap_usd=0.001)
    agent = OpenAIAgent(cfg, api_key="k")
    agent.generate(_MESSAGES)  # spends ~ $0.0145 > cap
    expected = usage_cost_usd(
        {"input_tokens": 1000, "cached_input_tokens": 200, "output_tokens": 400},
        price_for("gpt-6-astra", "flex"),
    )
    assert SPEND.usd == pytest.approx(expected)
    # The ledger is process-wide: a fresh agent is blocked too, before any request.
    with pytest.raises(OpenAISpendCapReached):
        OpenAIAgent(cfg, api_key="k").generate(_MESSAGES)
    assert seen["calls"] == 1


def test_usage_cost_prices_cached_input_at_cached_rate():
    prices = price_for("gpt-6-astra", "flex")  # (5.00, 0.50, 25.00)
    usd = usage_cost_usd(
        {"input_tokens": 1_000_000, "cached_input_tokens": 400_000, "output_tokens": 1_000_000}, prices
    )
    assert usd == pytest.approx(600_000 * 5 / 1e6 + 400_000 * 0.5 / 1e6 + 25.0)
    assert price_for("gpt-5.6-terra", None) == (2.00, 0.20, 12.00)


def test_markers_translate_and_runner_key_never_leaks():
    messages = [
        {"role": "system", "content": "sys " * 5},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "a", "cache_breakpoint": True},
                {"type": "text", "text": "tail"},
            ],
        },
    ]
    enabled = _to_openai_messages(messages, None, prompt_cache=True)
    assert enabled[1]["content"][0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    assert "cache_breakpoint" not in enabled[1]["content"][0]
    assert "prompt_cache_breakpoint" not in enabled[1]["content"][1]
    assert enabled[0]["content"][-1]["prompt_cache_breakpoint"] == {"mode": "explicit"}

    disabled = _to_openai_messages(messages, None, prompt_cache=False)
    assert disabled[0]["content"] == "sys " * 5
    assert disabled[1]["content"][0] == {"type": "text", "text": "a"}


def test_prompt_cache_options_only_when_enabled():
    kwargs = {
        "model": "gpt-6-luna",
        "max_tokens": 10,
        "reasoning_effort": None,
        "service_tier": None,
    }
    assert "prompt_cache_options" not in _build_body([], **kwargs)
    assert _build_body([], prompt_cache=True, **kwargs)["prompt_cache_options"] == {
        "mode": "explicit"
    }


def test_cost_prices_cache_writes_and_is_unchanged_without_them():
    prices = (10.0, 1.0, 50.0)
    usage = {
        "input_tokens": 1000,
        "cached_input_tokens": 400,
        "cache_creation_input_tokens": 200,
        "output_tokens": 10,
    }
    assert usage_cost_usd(usage, prices) == pytest.approx(
        (400 * 10 + 400 * 1 + 200 * 12.5 + 10 * 50) / 1e6
    )
    usage_without_writes = {
        "input_tokens": 1000,
        "cached_input_tokens": 400,
        "output_tokens": 10,
    }
    assert usage_cost_usd(usage_without_writes, prices) == pytest.approx(
        (600 * 10 + 400 * 1 + 10 * 50) / 1e6
    )


def test_normalize_is_idempotent():
    anthropic_usage = normalize_token_usage(
        {
            "input_tokens": 10,
            "output_tokens": 2,
            "cache_read_input_tokens": 80,
            "cache_creation_input_tokens": 10,
        }
    )
    assert anthropic_usage["input_tokens"] == 100
    assert normalize_token_usage(anthropic_usage) == anthropic_usage

    openai_usage = normalize_token_usage(
        {
            "prompt_tokens": 100,
            "completion_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": 80, "cache_write_tokens": 10},
        }
    )
    assert openai_usage["input_tokens"] == 100
    assert normalize_token_usage(openai_usage) == openai_usage


def test_more_than_four_cache_breakpoints_raises():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": str(index), "cache_breakpoint": True}
                for index in range(5)
            ],
        }
    ]
    with pytest.raises(ValueError, match="5 cache breakpoints"):
        _to_openai_messages(messages, None, prompt_cache=True)


def test_last_assistant_message_gets_breakpoint():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "next"},
    ]
    converted = _to_openai_messages(messages, None, prompt_cache=True)
    assistant = converted[2]["content"]
    assert assistant[-1]["prompt_cache_breakpoint"] == {"mode": "explicit"}


def test_spend_cap_requires_a_known_price():
    with pytest.raises(ValueError, match="No OpenAI price"):
        OpenAIAgent(OpenAIConfig(model="gpt-unknown", spend_cap_usd=5), api_key="k")


def test_reasoning_effort_none_rejected():
    with pytest.raises(ValueError, match="none"):
        OpenAIAgent(OpenAIConfig(reasoning_effort="none"), api_key="k")


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIAgent(OpenAIConfig())


def test_factory_builds_openai_agent_from_run_config(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    agent, model = _build_agent_from_spec(
        "gpt6_astra",
        {
            "provider": "openai",
            "model": "gpt-6-astra",
            "max_tokens": 64000,
            "reasoning_effort": "xhigh",
            "service_tier": "flex",
            "image_detail": "high",
            "timeout": 1800,
            "max_attempts": 30,
            "spend_cap_usd": 500,
            "temperature": 1.0,
        },
    )
    assert isinstance(agent, OpenAIAgent)
    assert model == "gpt-6-astra"
    c = agent.config
    assert (c.max_tokens, c.reasoning_effort, c.service_tier, c.image_detail) == (
        64000, "xhigh", "flex", "high",
    )
    assert (c.timeout, c.max_attempts, c.spend_cap_usd) == (1800.0, 30, 500.0)
    assert c.enable_prompt_cache is False


@pytest.mark.parametrize(
    ("model_config", "expected"),
    [
        ({"provider": "openai", "enable_prompt_cache": True}, True),
        ({"provider": "openai"}, False),
        ({"provider": "openai", "enable_prompt_cache": False}, False),
    ],
)
def test_openai_factory_applies_model_cache_flag(monkeypatch, model_config, expected):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    agent, _ = _build_agent_from_spec("test", model_config)
    assert isinstance(agent, OpenAIAgent)
    assert agent.config.enable_prompt_cache is expected


def test_factory_uses_experiment_cache_flag_unless_model_overrides(monkeypatch):
    import scripts.run_pipeline as pipeline

    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    config = {
        "experiment_config": {"prompt_cache": True},
        "models": {
            "claude_inherited": {"provider": "claude", "tasks": ["task-a"]},
            "claude_explicit": {
                "provider": "claude", "enable_prompt_cache": False, "tasks": ["task-b"]
            },
            "openai_inherited": {"provider": "openai", "tasks": ["task-c"]},
            "openai_explicit": {
                "provider": "openai", "enable_prompt_cache": False, "tasks": ["task-d"]
            },
        },
    }
    monkeypatch.setattr(pipeline, "load_run_config", lambda _path: config)
    monkeypatch.setattr(pipeline, "check_run_config_expectations", lambda *_args: None)
    monkeypatch.setattr(pipeline, "load_manifest", lambda *_args: [])
    monkeypatch.setattr(pipeline, "resolve_task_rows", lambda *_args: [{"task_id": "task"}])
    monkeypatch.setattr(pipeline, "check_tasks_runnable", lambda *_args: None)
    monkeypatch.setattr(pipeline, "_score_suite", lambda *_args, **_kwargs: ({}, 0.0))
    constructed_flags = []

    def capture_constructed_agent(_rows, agent, model_name, **_kwargs):
        constructed_flags.append(agent.config.enable_prompt_cache)
        return [], {}

    monkeypatch.setattr(pipeline, "_run_one_model", capture_constructed_agent)
    monkeypatch.setattr(pipeline, "_write_aggregate", lambda *_args: {})

    pipeline.run_from_config(
        run_config_path="unused.json",
    )
    assert constructed_flags == [True, False, True, False]


def test_explicit_openai_cache_flag_matches_resume_factory(monkeypatch):
    from scripts.resume_from_archive import build_live_agent

    monkeypatch.setenv("OPENAI_API_KEY", "k")
    model_config = {
        "provider": "openai",
        "enable_prompt_cache": True,
        "spend_cap_usd": 5,
    }
    pipeline_agent, _ = _build_agent_from_spec("test", model_config)
    resumed_agent = build_live_agent(
        model_config, spend_cap_usd=5, log=lambda _message: None
    )

    assert pipeline_agent.config.enable_prompt_cache is True
    assert resumed_agent.config.enable_prompt_cache is True


def test_main_parser_accepts_openai_agent(monkeypatch, capsys):
    import scripts.run_pipeline as pipeline

    factory_calls = []

    def fake_factory(name, model_cfg):
        factory_calls.append((name, model_cfg))
        return object(), "openai-test"

    monkeypatch.setattr(pipeline, "_build_agent_from_spec", fake_factory)
    monkeypatch.setattr(
        pipeline,
        "run_pipeline",
        lambda **_kwargs: {
            "scoring_calibration_summary": {"run_count": 0, "task_count": 0}
        },
    )

    pipeline.main(["--agent", "openai"])

    assert factory_calls == [("openai", {"provider": "openai"})]
    assert "Pipeline complete" in capsys.readouterr().out


def test_module_default_model_is_astra():
    assert openai_agent.DEFAULT_OPENAI_MODEL == "gpt-6-astra"


def test_gpt6_sol_flex_price_is_one_fifth_of_astra():
    # Pricing page fetched 2026-09-26: Sol flex $1 / $0.10 / $5 per 1M tokens.
    from interface.agents.openai_agent import price_for
    assert price_for("gpt-6-sol", "flex") == (1.00, 0.10, 5.00)
    astra = price_for("gpt-6-astra", "flex")
    assert all(abs(a / 5 - s) < 1e-9 for a, s in zip(astra, price_for("gpt-6-sol", "flex")))
