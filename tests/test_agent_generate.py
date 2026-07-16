"""`generate() -> Reply` on the three sync agents, plus `__call__` compat.

Each agent gains a `generate(messages) -> Reply` that surfaces stop_reason,
thinking (where the provider returns it), and token_truncated. `__call__` stays
byte-identical: it returns `reply.text` and mirrors `reply.usage`/`reply.thinking`
onto the mutable `last_usage`/`last_thinking` side-channels the pipeline reads.
"""
from __future__ import annotations

import json
import urllib.request

from interface.agents.claude import ClaudeAnthropicAgent, ClaudeAnthropicConfig
from interface.agents.kimi_k26 import KimiK26Agent, KimiK26Config
from interface.agents.qwen_vllm_api import QwenVLLMAPIAgent, QwenVLLMAPIConfig
from interface.agents.reply import Reply


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


_ANTHROPIC_OK = {
    "content": [
        {"type": "thinking", "thinking": "hmm"},
        {"type": "text", "text": "FINAL_OUTPUT: MOVE_NORTH"},
    ],
    "stop_reason": "max_tokens",
    "usage": {"input_tokens": 10, "output_tokens": 64000},
}

_OPENAI_OK = {
    "choices": [
        {
            "finish_reason": "length",
            "message": {"content": "partial", "reasoning_content": "thinking..."},
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 8000, "total_tokens": 8005},
}


def _patch(monkeypatch, payload):
    def fake_urlopen(req, timeout):
        return _FakeResponse(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


# --- Claude -----------------------------------------------------------------


def test_claude_generate_returns_reply(monkeypatch):
    _patch(monkeypatch, _ANTHROPIC_OK)
    agent = ClaudeAnthropicAgent(
        ClaudeAnthropicConfig(model="claude-opus-4-8", max_tokens=64000),
        api_key="secret",
    )
    reply = agent.generate([{"role": "user", "content": "go"}])
    assert isinstance(reply, Reply)
    assert reply.text == "FINAL_OUTPUT: MOVE_NORTH"
    assert reply.thinking == "hmm"
    assert reply.stop_reason == "max_tokens"
    assert reply.token_truncated is True
    assert reply.usage == {"input_tokens": 10, "output_tokens": 64000, "total_tokens": 64010}


def test_claude_call_is_compat(monkeypatch):
    _patch(monkeypatch, _ANTHROPIC_OK)
    agent = ClaudeAnthropicAgent(
        ClaudeAnthropicConfig(model="claude-opus-4-8", max_tokens=64000),
        api_key="secret",
    )
    text = agent([{"role": "user", "content": "go"}])
    assert text == "FINAL_OUTPUT: MOVE_NORTH"
    assert agent.last_usage == {"input_tokens": 10, "output_tokens": 64000, "total_tokens": 64010}
    assert agent.last_thinking == "hmm"


def test_claude_generate_not_truncated_on_end_turn(monkeypatch):
    payload = {
        "content": [{"type": "text", "text": "FINAL_OUTPUT: DONE"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 3, "output_tokens": 4},
    }
    _patch(monkeypatch, payload)
    agent = ClaudeAnthropicAgent(ClaudeAnthropicConfig(max_tokens=1024), api_key="secret")
    reply = agent.generate([{"role": "user", "content": "go"}])
    assert reply.stop_reason == "end_turn"
    assert reply.token_truncated is False
    assert reply.thinking is None


# --- Kimi -------------------------------------------------------------------


def test_kimi_generate_returns_reply(monkeypatch):
    _patch(monkeypatch, _OPENAI_OK)
    agent = KimiK26Agent(KimiK26Config(model="kimi-k2.6", max_tokens=8000), api_key="secret")
    reply = agent.generate([{"role": "user", "content": "go"}])
    assert isinstance(reply, Reply)
    assert reply.text == "partial"
    assert reply.thinking == "thinking..."
    assert reply.stop_reason == "length"
    assert reply.token_truncated is True
    assert reply.usage == {"input_tokens": 5, "output_tokens": 8000, "total_tokens": 8005}


def test_kimi_call_is_compat(monkeypatch):
    _patch(monkeypatch, _OPENAI_OK)
    agent = KimiK26Agent(KimiK26Config(model="kimi-k2.6", max_tokens=8000), api_key="secret")
    text = agent([{"role": "user", "content": "go"}])
    assert text == "partial"
    assert agent.last_usage == {"input_tokens": 5, "output_tokens": 8000, "total_tokens": 8005}
    assert agent.last_thinking == "thinking..."


def test_kimi_thinking_none_when_absent(monkeypatch):
    payload = {
        "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    }
    _patch(monkeypatch, payload)
    agent = KimiK26Agent(KimiK26Config(max_tokens=4096), api_key="secret")
    reply = agent.generate([{"role": "user", "content": "go"}])
    assert reply.thinking is None
    assert reply.stop_reason == "stop"
    assert reply.token_truncated is False


# --- Qwen (served vLLM) -----------------------------------------------------


def test_qwen_generate_returns_reply(monkeypatch):
    _patch(monkeypatch, _OPENAI_OK)
    agent = QwenVLLMAPIAgent(QwenVLLMAPIConfig(max_tokens=8000))
    reply = agent.generate([{"role": "user", "content": "go"}])
    assert isinstance(reply, Reply)
    assert reply.text == "partial"
    # Qwen reasons inline; no separate thinking channel.
    assert reply.thinking is None
    assert reply.stop_reason == "length"
    assert reply.token_truncated is True
    assert reply.usage == {"input_tokens": 5, "output_tokens": 8000, "total_tokens": 8005}


def test_qwen_call_is_compat(monkeypatch):
    _patch(monkeypatch, _OPENAI_OK)
    agent = QwenVLLMAPIAgent(QwenVLLMAPIConfig(max_tokens=8000))
    text = agent([{"role": "user", "content": "go"}])
    assert text == "partial"
    assert agent.last_usage == {"input_tokens": 5, "output_tokens": 8000, "total_tokens": 8005}
    assert agent.last_thinking is None
