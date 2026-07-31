"""Unit tests for the Claude Anthropic agent request-body construction.

The load-bearing invariant here is Opus-4.7+/Fable readiness: those model
families REJECT `temperature`/`top_p`/`top_k` with an HTTP 400, so the agent
must omit sampling params for them while still sending them for Sonnet 4.6 and
Opus <=4.6 (which still accept `temperature`). Thinking/effort are opt-in.
"""
from __future__ import annotations

from interface.agents.claude import (
    _build_request_body,
    _omits_sampling_params,
    _parse_response,
)


def _body(model: str, **kw):
    return _build_request_body(
        model=model,
        max_tokens=4096,
        temperature=kw.pop("temperature", 0.0),
        system=kw.pop("system", None),
        messages=kw.pop("messages", [{"role": "user", "content": "hi"}]),
        **kw,
    )


def test_opus_48_omits_temperature():
    body = _body("claude-opus-4-8")
    assert "temperature" not in body
    assert body["model"] == "claude-opus-4-8"
    assert body["max_tokens"] == 4096


def test_opus_47_and_fable_and_mythos_omit_sampling_params():
    for model in ("claude-opus-4-7", "claude-fable-5", "claude-mythos-5"):
        assert _omits_sampling_params(model), model
        assert "temperature" not in _body(model)


def test_sonnet_46_and_opus_46_keep_temperature():
    for model in ("claude-sonnet-4-6", "claude-opus-4-6"):
        assert not _omits_sampling_params(model), model
        body = _body(model, temperature=0.0)
        assert body["temperature"] == 0.0


def test_thinking_and_effort_are_added_when_requested():
    body = _body("claude-opus-4-8", enable_thinking=True, effort="low")
    # display="summarized" so Opus 4.8's thinking text is loggable (defaults to
    # omitted otherwise, silently dropping every thinking trace).
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"]["effort"] == "low"
    # still no temperature for the Opus family, even with thinking on
    assert "temperature" not in body


def test_thinking_off_by_default_sends_no_thinking_block():
    body = _body("claude-opus-4-8")
    assert "thinking" not in body
    assert "output_config" not in body


def test_system_and_messages_preserved():
    body = _body("claude-opus-4-8", system="sys", messages=[{"role": "user", "content": "x"}])
    assert body["system"] == "sys"
    assert body["messages"] == [{"role": "user", "content": "x"}]


def test_parse_response_captures_thinking_and_text():
    payload = {
        "content": [
            {"type": "thinking", "thinking": "I face EAST; goal is south, so TURN_RIGHT."},
            {"type": "text", "text": "FINAL_OUTPUT: TURN_RIGHT"},
        ],
        "usage": {"input_tokens": 100, "output_tokens": 40},
    }
    text, usage, thinking = _parse_response(payload)
    assert text == "FINAL_OUTPUT: TURN_RIGHT"
    assert thinking == "I face EAST; goal is south, so TURN_RIGHT."
    assert usage["input_tokens"] == 100 and usage["output_tokens"] == 40


def test_parse_response_thinking_none_when_absent():
    payload = {"content": [{"type": "text", "text": "FINAL_OUTPUT: MOVE_FORWARD"}]}
    text, _usage, thinking = _parse_response(payload)
    assert text == "FINAL_OUTPUT: MOVE_FORWARD"
    assert thinking is None
