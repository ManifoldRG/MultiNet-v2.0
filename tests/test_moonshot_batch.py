"""Moonshot (Kimi) Batch API client + `KimiK26Agent.generate_batch`.

Mirrors `tests/test_anthropic_batch.py`. The batch path must never drift from
the sync path: per-item bodies are built by the same `_build_body` code (with
sampling params omitted per the Moonshot spec), and succeeded chat-completions
bodies parse through the same logic as `generate`. The provider returns batch
results in arbitrary order, so the client maps back by `custom_id` and the agent
restores input order. A poll deadline cancels the batch and degrades missing ids
to `batch_expired` stubs rather than hanging; a normally-completed batch's
missing ids (errors landed in the error_file, not read) become `batch_errored`.
"""
from __future__ import annotations

import io
import json
import urllib.error
from typing import List, Optional

import interface.agents.moonshot_batch as batch_mod
from interface.agents.kimi_k26 import KimiK26Agent, KimiK26Config
from interface.agents.moonshot_batch import MoonshotBatchDeadline, run_moonshot_batch
from interface.agents.reply import Reply

BASE = "https://api.moonshot.ai/v1"


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


def _line(custom_id: str, text: str, *, finish_reason="stop", reasoning=None, wrap=True):
    message = {"content": text}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    body = {
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }
    if wrap:  # normal shape: response.body
        return {"custom_id": custom_id, "response": {"status_code": 200, "body": body}}
    return {"custom_id": custom_id, "body": body}  # fallback shape: top-level body


_TERMINAL = ("completed", "failed", "expired", "cancelled")


class _ScriptedURLopen:
    """Scripted urlopen for the Moonshot file-based batch flow, network-free.

    `status_sequence` drives successive GETs on the batch object (last value
    repeats). Cancel flips subsequent status GETs onto
    `post_cancel_status_sequence` (defaults to "cancelling" forever). Any
    terminal status attaches `output_file_id` when `results_jsonl` is present.
    """

    OUTPUT_FILE_ID = "out_1"

    def __init__(
        self,
        *,
        status_sequence: List[str],
        post_cancel_status_sequence: Optional[List[str]] = None,
        results_jsonl: Optional[str] = None,
        batch_id: str = "batch_1",
        input_file_id: str = "file_1",
    ):
        self.status_sequence = list(status_sequence)
        self.post_cancel_status_sequence = list(post_cancel_status_sequence or ["cancelling"])
        self.results_jsonl = results_jsonl
        self.batch_id = batch_id
        self.input_file_id = input_file_id
        self.calls: List[tuple] = []
        self.uploaded: List[bytes] = []
        self.created_body: Optional[dict] = None
        self.canceled = False
        self._status_i = 0

    def _next_status(self) -> str:
        seq = self.post_cancel_status_sequence if self.canceled else self.status_sequence
        i = min(self._status_i, len(seq) - 1)
        self._status_i += 1
        return seq[i]

    def _batch_obj(self, status: str) -> dict:
        obj = {"id": self.batch_id, "status": status}
        if status in _TERMINAL and self.results_jsonl is not None:
            obj["output_file_id"] = self.OUTPUT_FILE_ID
        return obj

    def __call__(self, req, timeout=None):
        method = req.get_method()
        url = req.full_url
        self.calls.append((method, url))
        if method == "POST" and url == f"{BASE}/files":
            self.uploaded.append(req.data)
            return _json_resp({"id": self.input_file_id})
        if method == "POST" and url == f"{BASE}/batches":
            self.created_body = json.loads(req.data.decode("utf-8"))
            return _json_resp({"id": self.batch_id, "status": "validating"})
        if method == "POST" and url == f"{BASE}/batches/{self.batch_id}/cancel":
            self.canceled = True
            self._status_i = 0
            return _json_resp(self._batch_obj("cancelling"))
        if method == "GET" and url == f"{BASE}/batches/{self.batch_id}":
            return _json_resp(self._batch_obj(self._next_status()))
        if method == "GET" and url == f"{BASE}/files/{self.OUTPUT_FILE_ID}/content":
            return _text_resp(self.results_jsonl or "")
        raise AssertionError(f"unexpected request: {method} {url}")

    def count(self, method: str, suffix: str) -> int:
        return sum(1 for m, u in self.calls if m == method and u.endswith(suffix))


def _patch(monkeypatch, responder):
    monkeypatch.setattr(batch_mod.urllib.request, "urlopen", responder)
    monkeypatch.setattr(batch_mod.time, "sleep", lambda _s: None)


def _lines(n: int) -> List[dict]:
    return [
        {"custom_id": f"i{i}", "method": "POST", "url": "/v1/chat/completions", "body": {}}
        for i in range(n)
    ]


# --- client: run_moonshot_batch --------------------------------------------


def test_results_map_by_custom_id_out_of_order(monkeypatch):
    responder = _ScriptedURLopen(
        status_sequence=["in_progress", "completed"],
        results_jsonl=_jsonl(  # shuffled relative to input order i0, i1, i2
            _line("i2", "third"),
            _line("i0", "first"),
            _line("i1", "second"),
        ),
    )
    _patch(monkeypatch, responder)
    results = run_moonshot_batch(_lines(3), api_key="k", poll_interval_s=0, deadline_s=100)
    assert set(results) == {"i0", "i1", "i2"}
    assert results["i0"]["choices"][0]["message"]["content"] == "first"
    assert results["i1"]["choices"][0]["message"]["content"] == "second"
    assert results["i2"]["choices"][0]["message"]["content"] == "third"


def test_upload_creates_batch_and_carries_jsonl(monkeypatch):
    """The lines are uploaded as a multipart JSONL file, then the batch is
    created referencing that file id at the chat/completions endpoint."""
    responder = _ScriptedURLopen(
        status_sequence=["completed"], results_jsonl=_jsonl(_line("i0", "ok"))
    )
    _patch(monkeypatch, responder)
    lines = [
        {"custom_id": "i0", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "m"}}
    ]
    run_moonshot_batch(lines, api_key="k", poll_interval_s=0, deadline_s=100)
    assert responder.count("POST", "/files") == 1
    # The uploaded multipart payload must contain the JSONL line verbatim.
    blob = responder.uploaded[0].decode("utf-8")
    assert 'purpose' in blob and 'batch' in blob
    parsed = [json.loads(l) for l in blob.splitlines() if l.strip().startswith("{")]
    assert {"custom_id": "i0", "method": "POST", "url": "/v1/chat/completions",
            "body": {"model": "m"}} in parsed
    # Batch create references the uploaded file at the chat/completions endpoint.
    assert responder.created_body["input_file_id"] == "file_1"
    assert responder.created_body["endpoint"] == "/v1/chat/completions"
    assert responder.created_body["completion_window"] == "24h"


def test_response_body_falls_back_to_top_level_body(monkeypatch):
    responder = _ScriptedURLopen(
        status_sequence=["completed"],
        results_jsonl=_jsonl(_line("i0", "wrapped"), _line("i1", "flat", wrap=False)),
    )
    _patch(monkeypatch, responder)
    results = run_moonshot_batch(_lines(2), api_key="k", poll_interval_s=0, deadline_s=100)
    assert results["i0"]["choices"][0]["message"]["content"] == "wrapped"
    assert results["i1"]["choices"][0]["message"]["content"] == "flat"


def test_line_without_custom_id_is_skipped(monkeypatch):
    responder = _ScriptedURLopen(
        status_sequence=["completed"],
        results_jsonl=_jsonl(
            _line("i0", "good"),
            {"response": {"body": {"choices": []}}},  # no custom_id
            _line("i1", "also good"),
        ),
    )
    _patch(monkeypatch, responder)
    results = run_moonshot_batch(_lines(2), api_key="k", poll_interval_s=0, deadline_s=100)
    assert set(results) == {"i0", "i1"}


def test_terminal_fail_collects_partials_without_raising(monkeypatch):
    """A `failed` batch with a partial output_file returns what exists; missing
    ids are simply absent (they become batch_errored at the agent layer)."""
    responder = _ScriptedURLopen(
        status_sequence=["in_progress", "failed"],
        results_jsonl=_jsonl(_line("i0", "done before fail")),
    )
    _patch(monkeypatch, responder)
    results = run_moonshot_batch(_lines(2), api_key="k", poll_interval_s=0, deadline_s=100)
    assert set(results) == {"i0"}
    assert responder.count("POST", "/cancel") == 0


def test_deadline_cancels_and_raises_with_partial(monkeypatch):
    """Deadline -> cancel -> grace poll to terminal -> raise carrying salvaged
    finished items."""
    responder = _ScriptedURLopen(
        status_sequence=["in_progress"],  # never terminal before the deadline
        post_cancel_status_sequence=["cancelling", "cancelled"],
        results_jsonl=_jsonl(_line("i0", "finished before cancel")),
    )
    _patch(monkeypatch, responder)
    try:
        run_moonshot_batch(
            _lines(2), api_key="k", poll_interval_s=0.001, deadline_s=0.02, cancel_grace_s=5.0
        )
        assert False, "expected MoonshotBatchDeadline"
    except MoonshotBatchDeadline as exc:
        assert set(exc.results) == {"i0"}
        assert exc.results["i0"]["choices"][0]["message"]["content"] == "finished before cancel"
    assert responder.count("POST", "/cancel") == 1


class _CancelRaisesURLopen(_ScriptedURLopen):
    """Like the base, but the deadline cancel POST 4xxs (the batch reached a
    terminal state between the last poll and the cancel — a benign race). The
    client must swallow it and still fall through to the grace poll + salvage."""

    def __call__(self, req, timeout=None):
        method = req.get_method()
        url = req.full_url
        if method == "POST" and url == f"{BASE}/batches/{self.batch_id}/cancel":
            self.calls.append((method, url))
            self.canceled = True
            self._status_i = 0
            raise urllib.error.HTTPError(
                url, 400, "already ended", {}, io.BytesIO(b"batch already ended")
            )
        return super().__call__(req, timeout=timeout)


def test_cancel_on_ended_race_still_salvages(monkeypatch):
    """Cancel POST raises (batch ended mid-flight) -> grace poll still salvages
    the finished item into the raised MoonshotBatchDeadline."""
    responder = _CancelRaisesURLopen(
        status_sequence=["in_progress"],  # never terminal before the deadline
        post_cancel_status_sequence=["cancelled"],
        results_jsonl=_jsonl(_line("i0", "finished before cancel")),
    )
    _patch(monkeypatch, responder)
    try:
        run_moonshot_batch(
            _lines(2), api_key="k", poll_interval_s=0.001, deadline_s=0.02, cancel_grace_s=5.0
        )
        assert False, "expected MoonshotBatchDeadline"
    except MoonshotBatchDeadline as exc:
        assert set(exc.results) == {"i0"}  # salvaged despite the cancel 4xx
        assert exc.results["i0"]["choices"][0]["message"]["content"] == "finished before cancel"
    assert responder.count("POST", "/cancel") == 1  # cancel was attempted


def test_deadline_grace_expiry_raises_empty(monkeypatch):
    responder = _ScriptedURLopen(status_sequence=["in_progress"])  # never terminal
    _patch(monkeypatch, responder)
    try:
        run_moonshot_batch(
            _lines(1), api_key="k", poll_interval_s=0.001, deadline_s=0.02, cancel_grace_s=0.02
        )
        assert False, "expected MoonshotBatchDeadline"
    except MoonshotBatchDeadline as exc:
        assert exc.results == {}
    assert responder.count("POST", "/cancel") == 1


# --- agent: generate_batch --------------------------------------------------


def _agent(**overrides):
    cfg = dict(
        model="kimi-k2.6",
        max_tokens=64000,
        enable_thinking=True,
        batch_poll_interval_s=0,
        batch_deadline_s=100,
    )
    cfg.update(overrides)
    return KimiK26Agent(KimiK26Config(**cfg), api_key="secret")


def test_generate_batch_maps_back_in_input_order(monkeypatch):
    responder = _ScriptedURLopen(
        status_sequence=["completed"],
        results_jsonl=_jsonl(
            _line("i2", "FINAL_OUTPUT: C"),
            _line("i0", "FINAL_OUTPUT: A"),
            _line("i1", "FINAL_OUTPUT: B"),
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


def test_missing_id_becomes_batch_errored_stub(monkeypatch):
    responder = _ScriptedURLopen(
        status_sequence=["completed"],
        results_jsonl=_jsonl(_line("i0", "FINAL_OUTPUT: A")),  # i1 absent (errored)
    )
    _patch(monkeypatch, responder)
    replies = _agent().generate_batch(
        [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]]
    )
    assert replies[0].text == "FINAL_OUTPUT: A"
    assert replies[1] == Reply(text="", stop_reason="batch_errored")


def _deadline_agent():
    return _agent(batch_poll_interval_s=0.001, batch_deadline_s=0.02, batch_cancel_grace_s=0.02)


def test_deadline_salvages_finished_and_expires_missing(monkeypatch):
    responder = _ScriptedURLopen(
        status_sequence=["in_progress"],
        post_cancel_status_sequence=["cancelled"],
        results_jsonl=_jsonl(_line("i0", "FINAL_OUTPUT: A")),
    )
    _patch(monkeypatch, responder)
    replies = _deadline_agent().generate_batch(
        [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]]
    )
    assert responder.count("POST", "/cancel") == 1
    assert replies[0].text == "FINAL_OUTPUT: A"
    assert replies[0].stop_reason == "stop"
    assert replies[1] == Reply(text="", stop_reason="batch_expired")


def test_deadline_missing_ids_become_batch_expired(monkeypatch):
    responder = _ScriptedURLopen(status_sequence=["in_progress"])  # never terminal
    _patch(monkeypatch, responder)
    replies = _deadline_agent().generate_batch(
        [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]]
    )
    assert responder.count("POST", "/cancel") == 1
    assert all(r == Reply(text="", stop_reason="batch_expired") for r in replies)


def test_generate_batch_reply_parsing(monkeypatch):
    responder = _ScriptedURLopen(
        status_sequence=["completed"],
        results_jsonl=_jsonl(
            _line("i0", "FINAL_OUTPUT: MOVE_NORTH", finish_reason="length", reasoning="hmm")
        ),
    )
    _patch(monkeypatch, responder)
    reply = _agent().generate_batch([[{"role": "user", "content": "go"}]])[0]
    assert isinstance(reply, Reply)
    assert reply.text == "FINAL_OUTPUT: MOVE_NORTH"
    assert reply.thinking == "hmm"
    assert reply.stop_reason == "length"
    assert reply.token_truncated is True
    assert reply.usage == {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}


def test_generate_batch_does_not_mutate_side_channels(monkeypatch):
    responder = _ScriptedURLopen(
        status_sequence=["completed"],
        results_jsonl=_jsonl(_line("i0", "hi", reasoning="reasoning")),
    )
    _patch(monkeypatch, responder)
    agent = _agent()
    agent.generate_batch([[{"role": "user", "content": "go"}]])
    assert agent.last_usage is None
    assert agent.last_thinking is None


def test_generate_batch_body_omits_temperature_carries_thinking_and_max_tokens(monkeypatch):
    """Per-line bodies: thinking enabled, explicit max_tokens, NO temperature
    (batch omits sampling params entirely, unlike the mode-forced sync path)."""
    captured = {}

    def _capture(lines, **kwargs):
        captured["lines"] = lines
        return {"i0": _line("i0", "ok")["response"]["body"]}

    monkeypatch.setattr(batch_mod, "run_moonshot_batch", _capture)
    import interface.agents.kimi_k26 as kimi_mod

    monkeypatch.setattr(kimi_mod, "run_moonshot_batch", _capture)
    _agent().generate_batch([[{"role": "user", "content": "go"}]])
    line = captured["lines"][0]
    assert line["custom_id"] == "i0"
    assert line["method"] == "POST"
    assert line["url"] == "/v1/chat/completions"
    body = line["body"]
    assert body["model"] == "kimi-k2.6"
    assert body["max_tokens"] == 64000
    assert body["thinking"] == {"type": "enabled"}
    assert "temperature" not in body


def test_generate_batch_body_thinking_disabled_when_off(monkeypatch):
    captured = {}

    def _capture(lines, **kwargs):
        captured["lines"] = lines
        return {"i0": _line("i0", "ok")["response"]["body"]}

    import interface.agents.kimi_k26 as kimi_mod

    monkeypatch.setattr(kimi_mod, "run_moonshot_batch", _capture)
    _agent(enable_thinking=False).generate_batch([[{"role": "user", "content": "go"}]])
    body = captured["lines"][0]["body"]
    assert body["thinking"] == {"type": "disabled"}
    assert "temperature" not in body
