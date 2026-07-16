"""Tests for `QwenVLLMAPIAgent.generate_batch` thread-pool fan-out."""

from __future__ import annotations

import random
import time

from interface.agents.qwen_vllm_api import QwenVLLMAPIAgent, QwenVLLMAPIConfig
from interface.agents.reply import Reply


def _msgs(tag: str) -> list[dict]:
    return [{"role": "user", "content": tag}]


def test_generate_batch_preserves_input_order_under_jitter(monkeypatch):
    """Results align with inputs by position, not completion order."""

    def fake_generate(self, messages):
        # Jittered latency: earlier-submitted items may finish later.
        time.sleep(0.01 * random.random())
        return Reply(text=messages[0]["content"])

    monkeypatch.setattr(QwenVLLMAPIAgent, "generate", fake_generate)

    agent = QwenVLLMAPIAgent(QwenVLLMAPIConfig())
    batch = [_msgs(f"item-{i}") for i in range(25)]

    replies = agent.generate_batch(batch)

    assert [r.text for r in replies] == [f"item-{i}" for i in range(25)]


def test_generate_batch_isolates_per_item_exception(monkeypatch):
    """One raising item yields a stub Reply; others are unaffected."""

    def fake_generate(self, messages):
        if messages[0]["content"] == "boom":
            raise RuntimeError("upstream failed after retries")
        return Reply(text=messages[0]["content"])

    monkeypatch.setattr(QwenVLLMAPIAgent, "generate", fake_generate)

    agent = QwenVLLMAPIAgent(QwenVLLMAPIConfig())
    batch = [_msgs("a"), _msgs("boom"), _msgs("c")]

    replies = agent.generate_batch(batch)

    assert replies[0].text == "a"
    assert replies[1].text == "" and replies[1].stop_reason == "batch_errored"
    assert replies[2].text == "c"


def test_generate_batch_leaves_last_usage_untouched(monkeypatch):
    def fake_generate(self, messages):
        return Reply(text="x", usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})

    monkeypatch.setattr(QwenVLLMAPIAgent, "generate", fake_generate)

    agent = QwenVLLMAPIAgent(QwenVLLMAPIConfig())
    assert agent.last_usage is None

    agent.generate_batch([_msgs("a"), _msgs("b")])

    assert agent.last_usage is None


def test_generate_batch_empty_returns_empty(monkeypatch):
    def fake_generate(self, messages):  # pragma: no cover - should not be called
        raise AssertionError("generate should not run for an empty batch")

    monkeypatch.setattr(QwenVLLMAPIAgent, "generate", fake_generate)

    agent = QwenVLLMAPIAgent(QwenVLLMAPIConfig())
    assert agent.generate_batch([]) == []


def test_generate_batch_fanout_bounds_workers(monkeypatch):
    """`batch_fanout` caps concurrent workers; order still preserved."""
    import threading

    live = 0
    peak = 0
    lock = threading.Lock()

    def fake_generate(self, messages):
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
        time.sleep(0.02)
        with lock:
            live -= 1
        return Reply(text=messages[0]["content"])

    monkeypatch.setattr(QwenVLLMAPIAgent, "generate", fake_generate)

    agent = QwenVLLMAPIAgent(QwenVLLMAPIConfig(batch_fanout=2))
    batch = [_msgs(f"i{i}") for i in range(8)]

    replies = agent.generate_batch(batch)

    assert [r.text for r in replies] == [f"i{i}" for i in range(8)]
    assert peak <= 2
