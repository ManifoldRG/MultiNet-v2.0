"""Per-model spend caps on the Claude and Kimi agents (OpenAI's are in test_openai_agent.py).

Each agent prices its own usage after every request, adds it to that model's
running total in the process-wide ledger, and refuses the next request once the
total reaches ``spend_cap_usd``. Totals are kept per model name.
"""
from __future__ import annotations

import json
import urllib.request

import pytest

import interface.agents.claude as claude_module
import interface.agents.kimi_k26 as kimi_module
from interface.agents.claude import (
    CLAUDE_PRICES,
    ClaudeAnthropicAgent,
    ClaudeAnthropicConfig,
    claude_usage_cost_usd,
)
from interface.agents.kimi_k26 import KIMI_PRICES, KimiK26Agent, KimiK26Config, kimi_usage_cost_usd
from interface.agents.spend_cap import SPEND, SpendCapReached
from scripts.run_pipeline import _build_agent_from_spec

_MESSAGES = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]

# Raw Anthropic usage: input_tokens is the uncached remainder.
_ANTHROPIC_USAGE = {
    "input_tokens": 500,
    "cache_read_input_tokens": 4000,
    "cache_creation_input_tokens": 480,
    "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 480},
    "output_tokens": 1000,
}
# Claude Opus 5.5 ($4 input, $0.20 read, $20 output; 1h write 2x input), sync.
_OPUS_SYNC_USD = (500 * 4 + 4000 * 0.20 + 480 * 8 + 1000 * 20) / 1e6

_MOONSHOT_COMPLETION = {
    "choices": [{"message": {"content": "FINAL_OUTPUT: MOVE_FORWARD"}, "finish_reason": "stop"}],
    "usage": {
        "prompt_tokens": 5000,
        "completion_tokens": 10000,
        "prompt_tokens_details": {"cached_tokens": 4000, "cache_write_tokens": 480},
    },
}
# Kimi K3 ($3 input, $0.30 hit, $15 output, $6 1h write).
_K3_1H_USD = (520 * 3 + 4000 * 0.30 + 480 * 6 + 10000 * 15) / 1e6


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


@pytest.fixture(autouse=True)
def _reset_spend():
    SPEND.reset()
    yield
    SPEND.reset()


def _serve(monkeypatch, payload):
    seen = {"calls": 0}

    def fake_urlopen(req, timeout):
        seen["calls"] += 1
        return _FakeResponse(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return seen


def _anthropic_message():
    return {
        "content": [{"type": "text", "text": "FINAL_OUTPUT: MOVE_FORWARD"}],
        "stop_reason": "end_turn",
        "usage": dict(_ANTHROPIC_USAGE),
    }


def test_claude_sync_records_spend_and_blocks_at_the_cap(monkeypatch):
    seen = _serve(monkeypatch, _anthropic_message())
    cfg = ClaudeAnthropicConfig(model="claude-opus-5-5", cache_ttl="1h", spend_cap_usd=0.02)
    ClaudeAnthropicAgent(cfg, api_key="k").generate(_MESSAGES)
    assert SPEND.total("claude-opus-5-5") == pytest.approx(_OPUS_SYNC_USD)  # ~$0.027
    # Every agent instance of the model shares the total: a fresh one is refused
    # before it sends anything.
    with pytest.raises(SpendCapReached, match=r"\$0\.02 for claude-opus-5-5"):
        ClaudeAnthropicAgent(cfg, api_key="k").generate(_MESSAGES)
    assert seen["calls"] == 1


def test_claude_batch_round_is_metered_at_half_price_and_checked_before_submit(monkeypatch):
    submitted = []

    def fake_run_message_batch(requests, **_kwargs):
        submitted.append(len(requests))
        return {r["custom_id"]: {"type": "succeeded", "message": _anthropic_message()} for r in requests}

    monkeypatch.setattr(claude_module, "run_message_batch", fake_run_message_batch)
    cfg = ClaudeAnthropicConfig(model="claude-opus-5-5", batch_cache_ttl="1h", spend_cap_usd=0.025)
    agent = ClaudeAnthropicAgent(cfg, api_key="k")
    agent.generate_batch([_MESSAGES, _MESSAGES])
    # The batch discount halves every rate, cache reads and writes included.
    assert SPEND.total("claude-opus-5-5") == pytest.approx(2 * _OPUS_SYNC_USD / 2)  # ~$0.027
    with pytest.raises(SpendCapReached):
        agent.generate_batch([_MESSAGES])
    assert submitted == [2]


def test_claude_writes_without_a_ttl_split_are_priced_at_the_requested_ttl():
    prices = CLAUDE_PRICES["claude-opus-5-5"]
    usage = {"input_tokens": 1000, "cache_creation_input_tokens": 1000, "output_tokens": 0}
    assert claude_usage_cost_usd(usage, prices, cache_ttl="5m", batch=False) == pytest.approx(1000 * 5 / 1e6)
    assert claude_usage_cost_usd(usage, prices, cache_ttl="1h", batch=False) == pytest.approx(1000 * 8 / 1e6)


def test_kimi_sync_records_spend_at_the_configured_write_tier_and_blocks(monkeypatch):
    seen = _serve(monkeypatch, _MOONSHOT_COMPLETION)
    cfg = KimiK26Config(model="kimi-k3", enable_thinking=True, prompt_cache_ttl="1h", spend_cap_usd=0.15)
    KimiK26Agent(cfg, api_key="k").generate(_MESSAGES)
    assert SPEND.total("kimi-k3") == pytest.approx(_K3_1H_USD)  # ~$0.156
    with pytest.raises(SpendCapReached, match="kimi-k3"):
        KimiK26Agent(cfg, api_key="k").generate(_MESSAGES)
    assert seen["calls"] == 1
    usage = {"input_tokens": 1000, "cache_creation_input_tokens": 1000, "output_tokens": 0}
    prices = KIMI_PRICES["kimi-k3"]
    assert kimi_usage_cost_usd(usage, prices, cache_ttl=None) == pytest.approx(1000 * 3 / 1e6)


def test_kimi_batch_round_is_metered_and_checked_before_submit(monkeypatch):
    submitted = []

    def fake_run_moonshot_batch(lines, **_kwargs):
        submitted.append(len(lines))
        return {line["custom_id"]: _MOONSHOT_COMPLETION for line in lines}

    monkeypatch.setattr(kimi_module, "run_moonshot_batch", fake_run_moonshot_batch)
    monkeypatch.delenv("KIMI_BATCH_TRANSPORT", raising=False)
    cfg = KimiK26Config(model="kimi-k3", enable_thinking=True, spend_cap_usd=0.25)
    agent = KimiK26Agent(cfg, api_key="k")
    agent.generate_batch([_MESSAGES, _MESSAGES])
    # Batch bodies carry no TTL, so writes are priced on the 5m tier (the input rate).
    one = (520 * 3 + 4000 * 0.30 + 480 * 3 + 10000 * 15) / 1e6
    assert SPEND.total("kimi-k3") == pytest.approx(2 * one)
    with pytest.raises(SpendCapReached):
        agent.generate_batch([_MESSAGES])
    assert submitted == [2]


def test_caps_are_per_model(monkeypatch):
    _serve(monkeypatch, _MOONSHOT_COMPLETION)
    kimi = KimiK26Config(model="kimi-k3", enable_thinking=True, spend_cap_usd=0.01)
    KimiK26Agent(kimi, api_key="k").generate(_MESSAGES)
    assert SPEND.total("kimi-k3") > 0.01
    _serve(monkeypatch, _anthropic_message())
    opus = ClaudeAnthropicConfig(model="claude-opus-5-5", spend_cap_usd=0.01)
    ClaudeAnthropicAgent(opus, api_key="k").generate(_MESSAGES)  # Kimi's spend is not Opus's


def test_no_cap_no_metering(monkeypatch):
    _serve(monkeypatch, _anthropic_message())
    ClaudeAnthropicAgent(ClaudeAnthropicConfig(model="claude-sonnet-4-6"), api_key="k").generate(_MESSAGES)
    assert SPEND.total("claude-sonnet-4-6") == 0.0


def test_a_cap_on_an_unpriced_model_fails_at_construction():
    with pytest.raises(ValueError, match="No Claude price"):
        ClaudeAnthropicAgent(ClaudeAnthropicConfig(model="claude-sonnet-4-6", spend_cap_usd=5), api_key="k")
    with pytest.raises(ValueError, match="No Kimi price"):
        KimiK26Agent(KimiK26Config(model="kimi-k2.6", spend_cap_usd=5), api_key="k")


def test_run_config_spend_cap_reaches_claude_and_kimi(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("MOONSHOT_API_KEY", "k")
    claude, _ = _build_agent_from_spec(
        "opus", {"provider": "claude", "model": "claude-opus-5-5", "spend_cap_usd": 175}
    )
    assert claude.config.spend_cap_usd == 175.0
    kimi, _ = _build_agent_from_spec(
        "k3", {"provider": "kimi", "model": "kimi-k3", "enable_thinking": True, "spend_cap_usd": 1300}
    )
    assert kimi.config.spend_cap_usd == 1300.0
