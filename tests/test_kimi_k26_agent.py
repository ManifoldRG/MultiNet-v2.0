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
