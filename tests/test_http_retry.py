"""Tests for the bounded retry-with-backoff helper used by the API agents."""

from __future__ import annotations

import json
import socket
import urllib.error
from email.message import Message

import pytest

from interface.agents.http_retry import (
    call_with_retry,
    is_retryable_error,
    retry_after_seconds,
)


def _http_error(code: int, retry_after=None) -> urllib.error.HTTPError:
    hdrs = Message()
    if retry_after is not None:
        hdrs["Retry-After"] = str(retry_after)
    return urllib.error.HTTPError("http://x", code, "err", hdrs, None)


class _Flaky:
    """Callable that raises a queue of errors then returns a result."""

    def __init__(self, errors, result="ok"):
        self.errors = list(errors)
        self.result = result
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.result


def test_returns_immediately_on_success():
    sleeps: list[float] = []
    fn = _Flaky([])
    assert call_with_retry(fn, sleep=sleeps.append) == "ok"
    assert fn.calls == 1
    assert sleeps == []


def test_retries_retryable_then_succeeds():
    sleeps: list[float] = []
    fn = _Flaky([_http_error(503), _http_error(429)], result="done")
    assert call_with_retry(fn, sleep=sleeps.append, rand=lambda: 0.0) == "done"
    assert fn.calls == 3
    assert len(sleeps) == 2


def test_gives_up_after_max_attempts_and_raises_last():
    sleeps: list[float] = []
    fn = _Flaky([_http_error(500)] * 10)
    with pytest.raises(urllib.error.HTTPError):
        call_with_retry(fn, max_attempts=4, sleep=sleeps.append, rand=lambda: 0.0)
    assert fn.calls == 4
    assert len(sleeps) == 3


def test_does_not_retry_non_retryable_status():
    sleeps: list[float] = []
    fn = _Flaky([_http_error(400)])
    with pytest.raises(urllib.error.HTTPError):
        call_with_retry(fn, sleep=sleeps.append)
    assert fn.calls == 1
    assert sleeps == []


def test_retries_timeouts():
    sleeps: list[float] = []
    fn = _Flaky([TimeoutError(), socket.timeout()], result="ok")
    assert call_with_retry(fn, sleep=sleeps.append, rand=lambda: 0.0) == "ok"
    assert fn.calls == 3


def test_is_retryable_classification():
    assert is_retryable_error(_http_error(429))
    assert is_retryable_error(_http_error(503))
    assert not is_retryable_error(_http_error(400))
    assert not is_retryable_error(_http_error(401))
    assert is_retryable_error(TimeoutError())
    assert is_retryable_error(urllib.error.URLError("conn refused"))
    assert not is_retryable_error(ValueError("nope"))


def test_backoff_honors_retry_after_header():
    captured: list[float] = []
    fn = _Flaky([_http_error(429, retry_after=7)], result="ok")
    call_with_retry(fn, sleep=captured.append, max_delay=60.0, rand=lambda: 0.0)
    assert captured == [7.0]


def test_retry_after_capped_at_max_delay():
    assert retry_after_seconds(_http_error(429, retry_after=120)) == 120.0
    captured: list[float] = []
    fn = _Flaky([_http_error(429, retry_after=120)], result="ok")
    call_with_retry(fn, sleep=captured.append, max_delay=30.0, rand=lambda: 0.0)
    assert captured == [30.0]


def test_backoff_is_exponential_with_cap():
    captured: list[float] = []
    fn = _Flaky([_http_error(500)] * 4, result="ok")
    call_with_retry(
        fn, max_attempts=5, base_delay=1.0, max_delay=4.0,
        sleep=captured.append, rand=lambda: 0.0,
    )
    assert captured == [1.0, 2.0, 4.0, 4.0]


# --------------------------------------------------------------------------- #
# Agent wiring: a transient API failure must not abort the (paid) call.
# --------------------------------------------------------------------------- #
class _FakeResp:
    def __init__(self, body: str):
        self._b = body.encode("utf-8")

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FlakyOpen:
    """Fake urlopen: fail with a retryable 503 (Retry-After 0) N times, then 200."""

    def __init__(self, fails: int, body: str):
        self.fails = fails
        self.body = body
        self.calls = 0

    def __call__(self, req, timeout=None):
        self.calls += 1
        if self.calls <= self.fails:
            raise _http_error(503, retry_after=0)
        return _FakeResp(self.body)


def test_claude_agent_retries_transient_then_succeeds(monkeypatch):
    from interface.agents import claude as claude_mod

    body = json.dumps(
        {"content": [{"type": "text", "text": "OK"}],
         "usage": {"input_tokens": 1, "output_tokens": 1}}
    )
    opener = _FlakyOpen(fails=2, body=body)
    monkeypatch.setattr(claude_mod.urllib.request, "urlopen", opener)

    agent = claude_mod.ClaudeAnthropicAgent(api_key="test-key")
    assert agent([{"role": "user", "content": "hi"}]) == "OK"
    assert opener.calls == 3


def test_kimi_agent_retries_transient_then_succeeds(monkeypatch):
    from interface.agents import kimi_k26 as kimi_mod

    body = json.dumps(
        {"choices": [{"message": {"content": "OK"}}],
         "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    )
    opener = _FlakyOpen(fails=2, body=body)
    monkeypatch.setattr(kimi_mod.urllib.request, "urlopen", opener)

    agent = kimi_mod.KimiK26Agent(api_key="test-key")
    assert agent([{"role": "user", "content": "hi"}]) == "OK"
    assert opener.calls == 3
