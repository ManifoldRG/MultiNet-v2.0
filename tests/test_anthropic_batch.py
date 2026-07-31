"""Anthropic Message Batches client + `ClaudeAnthropicAgent.generate_batch`.

The batch path must never drift from the sync path: per-item `params` are built
by the same `_build_request_body` code, and succeeded messages parse through the
same content-block logic as `_parse_response`. Providers return batch results in
arbitrary order, so the client maps back by `custom_id` and the agent restores
input order. A poll deadline cancels the batch and degrades missing ids to
`batch_expired` stubs rather than hanging.
"""
from __future__ import annotations

import io
import json
import urllib.error
from typing import List, Optional

import interface.agents.anthropic_batch as batch_mod
from interface.agents.anthropic_batch import run_message_batch
from interface.agents.claude import ClaudeAnthropicAgent, ClaudeAnthropicConfig
from interface.agents.reply import Reply


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self._body


def _json_resp(payload) -> _FakeResponse:
    return _FakeResponse(json.dumps(payload).encode("utf-8"))


def _text_resp(text: str) -> _FakeResponse:
    return _FakeResponse(text.encode("utf-8"))


def _jsonl(*records) -> str:
    return "\n".join(json.dumps(r) for r in records) + "\n"


def _succeeded(custom_id: str, text: str, *, stop_reason="end_turn", thinking=None):
    content = []
    if thinking is not None:
        content.append({"type": "thinking", "thinking": thinking})
    content.append({"type": "text", "text": text})
    return {
        "custom_id": custom_id,
        "result": {
            "type": "succeeded",
            "message": {
                "content": content,
                "stop_reason": stop_reason,
                "usage": {"input_tokens": 10, "output_tokens": 20},
            },
        },
    }


class _ScriptedURLopen:
    """Scripted urlopen keyed on (method, url), readable and network-free.

    `status_sequence` drives successive GETs on the batch object (the last value
    repeats). Cancel returns "canceling" with `results_url: null` — the real
    API's immediate cancel response — and flips subsequent status GETs onto
    `post_cancel_status_sequence` (defaults to repeating "canceling" forever).
    An "ended" status attaches `results_url`.
    """

    RESULTS_URL = "https://api.anthropic.com/v1/messages/batches/batch_1/results"

    def __init__(
        self,
        *,
        status_sequence: List[str],
        post_cancel_status_sequence: Optional[List[str]] = None,
        results_jsonl: Optional[str] = None,
        batch_id: str = "batch_1",
    ):
        self.status_sequence = list(status_sequence)
        self.post_cancel_status_sequence = list(post_cancel_status_sequence or ["canceling"])
        self.results_jsonl = results_jsonl
        self.batch_id = batch_id
        self.calls: List[tuple] = []
        self.canceled = False
        self._status_i = 0

    def _next_status(self) -> str:
        seq = self.post_cancel_status_sequence if self.canceled else self.status_sequence
        i = min(self._status_i, len(seq) - 1)
        self._status_i += 1
        return seq[i]

    def __call__(self, req, timeout=None):
        method = req.get_method()
        url = req.full_url
        self.calls.append((method, url))
        if method == "POST" and url.endswith("/v1/messages/batches"):
            return _json_resp({"id": self.batch_id, "processing_status": "in_progress"})
        if method == "POST" and url.endswith(f"/batches/{self.batch_id}/cancel"):
            self.canceled = True
            self._status_i = 0
            return _json_resp(
                {"id": self.batch_id, "processing_status": "canceling", "results_url": None}
            )
        if method == "GET" and url.endswith(f"/batches/{self.batch_id}"):
            status = self._next_status()
            body = {"id": self.batch_id, "processing_status": status}
            if status == "ended":
                body["results_url"] = self.RESULTS_URL
            return _json_resp(body)
        if method == "GET" and url == self.RESULTS_URL:
            return _text_resp(self.results_jsonl or "")
        raise AssertionError(f"unexpected request: {method} {url}")

    def count(self, method: str, suffix: str) -> int:
        return sum(1 for m, u in self.calls if m == method and u.endswith(suffix))


def _patch(monkeypatch, responder):
    monkeypatch.setattr(batch_mod.urllib.request, "urlopen", responder)
    monkeypatch.setattr(batch_mod.time, "sleep", lambda _s: None)


# --- client: run_message_batch ---------------------------------------------


def test_results_map_by_custom_id_out_of_order(monkeypatch):
    responder = _ScriptedURLopen(
        status_sequence=["in_progress", "ended"],
        # Deliberately shuffled relative to input order i0, i1, i2.
        results_jsonl=_jsonl(
            _succeeded("i2", "third"),
            _succeeded("i0", "first"),
            _succeeded("i1", "second"),
        ),
    )
    _patch(monkeypatch, responder)
    requests = [{"custom_id": f"i{i}", "params": {"model": "m"}} for i in range(3)]
    results = run_message_batch(
        requests, api_key="k", poll_interval_s=0, deadline_s=100
    )
    assert set(results) == {"i0", "i1", "i2"}
    assert results["i0"]["message"]["content"][-1]["text"] == "first"
    assert results["i1"]["message"]["content"][-1]["text"] == "second"
    assert results["i2"]["message"]["content"][-1]["text"] == "third"


def test_deadline_cancels_and_returns_partial(monkeypatch):
    """Deadline -> cancel -> grace poll to "ended" -> finished items salvaged."""
    responder = _ScriptedURLopen(
        status_sequence=["in_progress"],  # never ends before the deadline
        post_cancel_status_sequence=["canceling", "ended"],
        results_jsonl=_jsonl(_succeeded("i0", "finished before cancel")),
    )
    _patch(monkeypatch, responder)
    results = run_message_batch(
        [{"custom_id": "i0", "params": {}}, {"custom_id": "i1", "params": {}}],
        api_key="k",
        poll_interval_s=0.001,
        deadline_s=0.02,
        cancel_grace_s=5.0,
    )
    assert responder.count("POST", "/cancel") == 1
    assert set(results) == {"i0"}  # i1 never finished; simply absent
    assert results["i0"]["message"]["content"][-1]["text"] == "finished before cancel"


class _CancelRaisesURLopen(_ScriptedURLopen):
    """Like the base, but the deadline cancel POST 4xxs (the batch reached a
    terminal state between the last poll and the cancel — a benign race). The
    client must swallow it and still fall through to the grace poll + salvage."""

    def __call__(self, req, timeout=None):
        method = req.get_method()
        url = req.full_url
        if method == "POST" and url.endswith(f"/batches/{self.batch_id}/cancel"):
            self.calls.append((method, url))
            self.canceled = True
            self._status_i = 0
            raise urllib.error.HTTPError(
                url, 400, "already ended", {}, io.BytesIO(b"batch already ended")
            )
        return super().__call__(req, timeout=timeout)


def test_cancel_on_ended_race_still_salvages(monkeypatch):
    """Cancel POST raises (batch ended mid-flight) -> grace poll still salvages."""
    responder = _CancelRaisesURLopen(
        status_sequence=["in_progress"],  # never ends before the deadline
        post_cancel_status_sequence=["ended"],
        results_jsonl=_jsonl(_succeeded("i0", "finished before cancel")),
    )
    _patch(monkeypatch, responder)
    results = run_message_batch(
        [{"custom_id": "i0", "params": {}}, {"custom_id": "i1", "params": {}}],
        api_key="k",
        poll_interval_s=0.001,
        deadline_s=0.02,
        cancel_grace_s=5.0,
    )
    assert responder.count("POST", "/cancel") == 1  # cancel was attempted
    assert set(results) == {"i0"}  # finished item salvaged despite the cancel 4xx
    assert results["i0"]["message"]["content"][-1]["text"] == "finished before cancel"


def test_deadline_grace_expiry_returns_empty(monkeypatch):
    """Batch never reaches "ended" within cancel_grace_s -> {} (no results_url)."""
    responder = _ScriptedURLopen(status_sequence=["in_progress"])  # cancel -> "canceling" forever
    _patch(monkeypatch, responder)
    results = run_message_batch(
        [{"custom_id": "i0", "params": {}}],
        api_key="k",
        poll_interval_s=0.001,
        deadline_s=0.02,
        cancel_grace_s=0.02,
    )
    assert results == {}
    assert responder.count("POST", "/cancel") == 1


def test_malformed_results_line_is_skipped(monkeypatch):
    """One bad JSONL line (no custom_id) must not KeyError away the whole batch."""
    responder = _ScriptedURLopen(
        status_sequence=["ended"],
        results_jsonl=_jsonl(
            _succeeded("i0", "good"),
            {"result": {"type": "succeeded"}},  # malformed: custom_id absent
            _succeeded("i1", "also good"),
        ),
    )
    _patch(monkeypatch, responder)
    requests = [{"custom_id": f"i{i}", "params": {}} for i in range(2)]
    results = run_message_batch(requests, api_key="k", poll_interval_s=0, deadline_s=100)
    assert set(results) == {"i0", "i1"}


# --- agent: generate_batch --------------------------------------------------


def _agent():
    return ClaudeAnthropicAgent(
        ClaudeAnthropicConfig(
            model="claude-opus-4-8",
            max_tokens=64000,
            batch_poll_interval_s=0,
            batch_deadline_s=100,
        ),
        api_key="secret",
    )


def test_generate_batch_maps_back_in_input_order(monkeypatch):
    responder = _ScriptedURLopen(
        status_sequence=["ended"],
        results_jsonl=_jsonl(
            _succeeded("i2", "FINAL_OUTPUT: C"),
            _succeeded("i0", "FINAL_OUTPUT: A"),
            _succeeded("i1", "FINAL_OUTPUT: B"),
        ),
    )
    _patch(monkeypatch, responder)
    batch = [
        [{"role": "user", "content": "a"}],
        [{"role": "user", "content": "b"}],
        [{"role": "user", "content": "c"}],
    ]
    replies = _agent().generate_batch(batch)
    assert [r.text for r in replies] == [
        "FINAL_OUTPUT: A",
        "FINAL_OUTPUT: B",
        "FINAL_OUTPUT: C",
    ]


def test_errored_item_becomes_stub_reply(monkeypatch):
    responder = _ScriptedURLopen(
        status_sequence=["ended"],
        results_jsonl=_jsonl(
            _succeeded("i0", "FINAL_OUTPUT: A"),
            {"custom_id": "i1", "result": {"type": "errored", "error": {"type": "x"}}},
        ),
    )
    _patch(monkeypatch, responder)
    replies = _agent().generate_batch(
        [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]]
    )
    assert replies[0].text == "FINAL_OUTPUT: A"
    assert replies[1] == Reply(text="", stop_reason="batch_errored")


def _deadline_agent():
    return ClaudeAnthropicAgent(
        ClaudeAnthropicConfig(
            model="claude-opus-4-8",
            max_tokens=64000,
            batch_poll_interval_s=0.001,
            batch_deadline_s=0.02,
            batch_cancel_grace_s=0.02,
        ),
        api_key="secret",
    )


def test_deadline_salvages_finished_and_expires_missing(monkeypatch):
    """After a deadline cancel, finished items become real Replies and ids
    absent from the salvaged results become batch_expired stubs."""
    responder = _ScriptedURLopen(
        status_sequence=["in_progress"],
        post_cancel_status_sequence=["ended"],
        results_jsonl=_jsonl(_succeeded("i0", "FINAL_OUTPUT: A")),
    )
    _patch(monkeypatch, responder)
    replies = _deadline_agent().generate_batch(
        [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]]
    )
    assert responder.count("POST", "/cancel") == 1
    assert replies[0].text == "FINAL_OUTPUT: A"
    assert replies[0].stop_reason == "end_turn"
    assert replies[1] == Reply(text="", stop_reason="batch_expired")


def test_deadline_missing_ids_become_batch_expired(monkeypatch):
    responder = _ScriptedURLopen(status_sequence=["in_progress"])  # never ends
    _patch(monkeypatch, responder)
    replies = _deadline_agent().generate_batch(
        [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]]
    )
    assert responder.count("POST", "/cancel") == 1
    assert all(r == Reply(text="", stop_reason="batch_expired") for r in replies)


def test_generate_batch_reply_parsing(monkeypatch):
    responder = _ScriptedURLopen(
        status_sequence=["ended"],
        results_jsonl=_jsonl(
            _succeeded(
                "i0",
                "FINAL_OUTPUT: MOVE_NORTH",
                stop_reason="max_tokens",
                thinking="hmm",
            )
        ),
    )
    _patch(monkeypatch, responder)
    reply = _agent().generate_batch([[{"role": "user", "content": "go"}]])[0]
    assert isinstance(reply, Reply)
    assert reply.text == "FINAL_OUTPUT: MOVE_NORTH"
    assert reply.thinking == "hmm"
    assert reply.stop_reason == "max_tokens"
    assert reply.token_truncated is True
    assert reply.usage == {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}


def test_generate_batch_does_not_mutate_side_channels(monkeypatch):
    responder = _ScriptedURLopen(
        status_sequence=["ended"],
        results_jsonl=_jsonl(_succeeded("i0", "hi", thinking="reasoning")),
    )
    _patch(monkeypatch, responder)
    agent = _agent()
    agent.generate_batch([[{"role": "user", "content": "go"}]])
    assert agent.last_usage is None
    assert agent.last_thinking is None


def test_generate_batch_reuses_build_request_body(monkeypatch):
    """Per-item params must be the exact sync request body (never drift)."""
    captured = {}

    def _capture(requests, **kwargs):
        captured["requests"] = requests
        return {"i0": _succeeded("i0", "ok")["result"]}

    monkeypatch.setattr(batch_mod, "run_message_batch", _capture)
    # generate_batch imports run_message_batch into claude's namespace; patch there too.
    import interface.agents.claude as claude_mod

    monkeypatch.setattr(claude_mod, "run_message_batch", _capture)
    _agent().generate_batch([[{"role": "user", "content": "go"}]])
    params = captured["requests"][0]["params"]
    assert captured["requests"][0]["custom_id"] == "i0"
    assert params["model"] == "claude-opus-4-8"
    assert params["max_tokens"] == 64000
    # Opus 4.8 rejects sampling params -> body must omit temperature.
    assert "temperature" not in params
