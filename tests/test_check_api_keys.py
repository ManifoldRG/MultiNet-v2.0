from __future__ import annotations

import deploy.check_api_keys as cak


def test_run_checks_present_ok_absent_skipped_failed_nonzero():
    calls = []

    def fake_roundtrip(provider, key):
        calls.append((provider, key))
        if provider == "kimi":
            raise RuntimeError("Moonshot API HTTP 401: bad key")
        return {"provider": provider, "ok": True, "reply": "I am well!", "usage": {"output_tokens": 5}}

    keys = {"anthropic": "good-key", "kimi": "bad-key"}
    results, code = cak.run_checks(keys, roundtrip=fake_roundtrip)

    by_provider = {r["provider"]: r for r in results}
    assert by_provider["anthropic"]["status"] == "ok"
    assert by_provider["kimi"]["status"] == "failed"
    assert code == 1
    assert ("anthropic", "good-key") in calls and ("kimi", "bad-key") in calls


def test_run_checks_absent_key_is_skipped_and_not_called():
    calls = []

    def fake_roundtrip(provider, key):
        calls.append(provider)
        return {"provider": provider, "ok": True, "reply": "hi", "usage": None}

    results, code = cak.run_checks({"anthropic": None, "kimi": "k"}, roundtrip=fake_roundtrip)
    by_provider = {r["provider"]: r for r in results}
    assert by_provider["anthropic"]["status"] == "skipped"
    assert calls == ["kimi"]
    assert code == 0


def test_hello_roundtrip_routes_to_kimi_agent(monkeypatch):
    seen = {}

    class FakeKimi:
        def __init__(self, config=None, api_key=None):
            seen["config"] = config
            seen["api_key"] = api_key
            self.last_usage = {"output_tokens": 3}

        def __call__(self, messages):
            seen["messages"] = messages
            return "Doing great!"

    import interface.agents.kimi_k26 as kimi_mod
    monkeypatch.setattr(kimi_mod, "KimiK26Agent", FakeKimi)

    out = cak.hello_roundtrip("kimi", "ms-key")
    assert out["ok"] is True and out["reply"] == "Doing great!"
    assert seen["api_key"] == "ms-key"
    assert seen["messages"] == [{"role": "user", "content": cak.HELLO}]
    assert seen["config"].max_tokens == 32


def test_hello_roundtrip_routes_to_claude_agent(monkeypatch):
    seen = {}

    class FakeClaude:
        def __init__(self, config=None, api_key=None):
            seen["config"] = config
            seen["api_key"] = api_key
            self.last_usage = {"output_tokens": 4}

        def __call__(self, messages):
            seen["messages"] = messages
            return "All good, thanks!"

    import interface.agents.claude as claude_mod
    monkeypatch.setattr(claude_mod, "ClaudeAnthropicAgent", FakeClaude)

    out = cak.hello_roundtrip("anthropic", "sk-ant-key")
    assert out["ok"] is True and out["reply"] == "All good, thanks!"
    assert seen["api_key"] == "sk-ant-key"
    assert seen["messages"] == [{"role": "user", "content": cak.HELLO}]
    assert seen["config"].max_tokens == 32
