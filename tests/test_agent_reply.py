from interface.agents.reply import Reply, detect_token_truncated


def test_reply_defaults():
    r = Reply(text="hi")
    assert r.usage is None and r.thinking is None
    assert r.stop_reason is None and r.token_truncated is False


def test_detect_by_stop_reason():
    assert detect_token_truncated("max_tokens", None, None)      # Anthropic
    assert detect_token_truncated("length", None, None)          # OpenAI-style
    assert not detect_token_truncated("end_turn", {"output_tokens": 99999}, 100)  # explicit stop wins


def test_detect_by_cap_fallback():
    assert detect_token_truncated(None, {"output_tokens": 8000}, 8000)
    assert not detect_token_truncated(None, {"output_tokens": 7999}, 8000)
    assert not detect_token_truncated(None, None, 8000)
    assert not detect_token_truncated(None, {"output_tokens": 8000}, None)
