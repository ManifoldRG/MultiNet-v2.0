from __future__ import annotations

import json
import urllib.request

from interface.agents.kimi_k26 import KimiK26Agent, KimiK26Config
from scripts.run_pipeline import _build_agent_from_spec


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


def test_kimi_agent_posts_moonshot_chat_completion(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        seen["authorization"] = req.get_header("Authorization")
        seen["timeout"] = timeout
        seen["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse(
            {
                "choices": [{"message": {"content": "FINAL_OUTPUT: DONE"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    agent = KimiK26Agent(
        KimiK26Config(model="kimi-k2.6", max_tokens=128, timeout=5),
        api_key="secret",
    )

    result = agent(
        [
            {"role": "system", "content": "system prompt"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc123"},
                    },
                ],
            },
        ]
    )

    assert result == "FINAL_OUTPUT: DONE"
    assert agent.last_usage == {
        "input_tokens": 12,
        "output_tokens": 3,
        "total_tokens": 15,
    }
    assert seen["url"] == "https://api.moonshot.ai/v1/chat/completions"
    assert seen["authorization"] == "Bearer secret"
    assert seen["timeout"] == 5
    assert seen["body"]["model"] == "kimi-k2.6"
    assert seen["body"]["max_tokens"] == 128
    assert seen["body"]["thinking"] == {"type": "disabled"}


def test_kimi_timeout_override_env_is_not_consulted(monkeypatch):
    """The R1 KIMI_TIMEOUT_OVERRIDE stopgap was removed post-campaign; the
    per-call timeout must come only from KimiK26Config (set via run_config),
    never from an ambient env var."""
    seen = {}

    def fake_urlopen(req, timeout):
        seen["timeout"] = timeout
        return _FakeResponse(
            {"choices": [{"message": {"content": "FINAL_OUTPUT: DONE"}}],
             "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("KIMI_TIMEOUT_OVERRIDE", "9999")
    agent = KimiK26Agent(
        KimiK26Config(model="kimi-k2.6", max_tokens=8, timeout=5),
        api_key="secret",
    )
    agent([{"role": "user", "content": "hi"}])
    assert seen["timeout"] == 5, "env override leaked into the socket timeout"


def test_kimi_temperature_pinned_by_thinking_mode(monkeypatch):
    """Moonshot 400s unless temperature is exactly the value it allows for the
    mode: 1.0 with thinking on, 0.6 with thinking off. The agent pins it per mode
    regardless of the configured value (confirmed live via the M6 smoke)."""
    seen = {}

    def fake_urlopen(req, timeout):
        seen["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse({"choices": [{"message": {"content": "ok"}}], "usage": {}})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    # thinking off -> 0.6, even if the config asked for something else
    KimiK26Agent(KimiK26Config(temperature=0.0, enable_thinking=False), api_key="secret")(
        [{"role": "user", "content": "hi"}]
    )
    assert seen["body"]["temperature"] == 0.6

    # thinking on -> 1.0, even if the config asked for 0.6
    KimiK26Agent(KimiK26Config(temperature=0.6, enable_thinking=True), api_key="secret")(
        [{"role": "user", "content": "hi"}]
    )
    assert seen["body"]["temperature"] == 1.0


def test_run_config_builds_kimi_agent(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "secret")

    agent, label = _build_agent_from_spec(
        "kimi",
        {"provider": "kimi", "model": "kimi-k2.6", "max_tokens": 64, "timeout": 10},
    )

    assert isinstance(agent, KimiK26Agent)
    assert label == "kimi-k2.6"
    assert agent.config.max_tokens == 64
    assert agent.config.timeout == 10


def test_run_config_wires_kimi_enable_thinking(monkeypatch):
    """enable_thinking must flow from the run-config into the Kimi agent (M5).
    It is a generation-affecting knob in the cache hash, so dropping it would
    make the hash and the actual model call disagree."""
    monkeypatch.setenv("MOONSHOT_API_KEY", "secret")

    on, _ = _build_agent_from_spec(
        "kimi", {"provider": "kimi", "model": "kimi-k2.6", "enable_thinking": True}
    )
    off, _ = _build_agent_from_spec(
        "kimi", {"provider": "kimi", "model": "kimi-k2.6", "enable_thinking": False}
    )
    assert on.config.enable_thinking is True
    assert off.config.enable_thinking is False
