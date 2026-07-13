from __future__ import annotations

import json
import sys
import types

from interface.agents.qwen_vllm import QwenVLLMAgent, QwenVLLMConfig, _to_openai_messages
from interface.agents.qwen_vllm_api import QwenVLLMAPIAgent, QwenVLLMAPIConfig


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeCompletion:
    text = " OK "
    token_ids = [101, 102]


class FakeOutput:
    prompt_token_ids = [1, 2, 3]
    outputs = [FakeCompletion()]


class FakeLLM:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        FakeLLM.instances.append(self)

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return [FakeOutput()]


def _install_fake_vllm(monkeypatch):
    fake = types.SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSamplingParams)
    monkeypatch.setitem(sys.modules, "vllm", fake)
    FakeLLM.instances.clear()


def test_qwen_vllm_agent_calls_offline_chat_and_records_usage(monkeypatch):
    _install_fake_vllm(monkeypatch)

    agent = QwenVLLMAgent(
        config=QwenVLLMConfig(
            model="Qwen/Qwen3.6-27B",
            max_tokens=64,
            max_model_len=4096,
            gpu_memory_utilization=0.85,
            enable_thinking=False,
            engine_kwargs={"max_num_batched_tokens": 8192},
        )
    )
    out = agent([{"role": "user", "content": "Reply with OK."}])

    assert out == "OK"
    assert agent.last_usage == {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
    llm = FakeLLM.instances[-1]
    assert llm.kwargs["model"] == "Qwen/Qwen3.6-27B"
    assert llm.kwargs["max_model_len"] == 4096
    assert llm.kwargs["gpu_memory_utilization"] == 0.85
    assert llm.kwargs["max_num_batched_tokens"] == 8192
    messages, kwargs = llm.calls[-1]
    assert messages == [{"role": "user", "content": "Reply with OK."}]
    assert kwargs["sampling_params"].kwargs["max_tokens"] == 64
    assert kwargs["chat_template_kwargs"] == {"enable_thinking": False}


def test_qwen_vllm_message_conversion_preserves_image_urls():
    messages = _to_openai_messages(
        [
            {"role": "system", "content": "sys"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            },
        ]
    )

    assert messages[0] == {"role": "system", "content": "sys"}
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
    ]


def test_run_pipeline_builds_qwen_vllm_agent(monkeypatch):
    _install_fake_vllm(monkeypatch)

    from scripts.run_pipeline import _build_agent_from_spec

    agent, label = _build_agent_from_spec(
        "qwen36_vllm",
        {
            "provider": "qwen_vllm",
            "model": "Qwen/Qwen3.6-27B",
            "max_tokens": 32,
            "max_model_len": 2048,
            "gpu_memory_utilization": 0.8,
            "enable_prefix_caching": True,
            "enable_thinking": False,
        },
    )

    assert label == "Qwen/Qwen3.6-27B"
    assert isinstance(agent, QwenVLLMAgent)
    assert FakeLLM.instances[-1].kwargs["model"] == "Qwen/Qwen3.6-27B"
    assert FakeLLM.instances[-1].kwargs["max_model_len"] == 2048


def test_qwen_vllm_api_agent_posts_openai_chat(monkeypatch):
    seen = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [{"message": {"content": " FINAL_OUTPUT: DONE "}}],
                    "usage": {
                        "prompt_tokens": 7,
                        "completion_tokens": 3,
                        "total_tokens": 10,
                    },
                }
            ).encode()

    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        seen["timeout"] = timeout
        seen["body"] = json.loads(req.data.decode())
        seen["auth"] = req.headers.get("Authorization")
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    agent = QwenVLLMAPIAgent(
        QwenVLLMAPIConfig(
            model="Qwen/Qwen3.6-27B",
            base_url="http://127.0.0.1:8000/v1",
            api_key="EMPTY",
            max_tokens=32,
            enable_thinking=False,
        )
    )

    assert agent([{"role": "user", "content": "move"}]) == "FINAL_OUTPUT: DONE"
    assert seen["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert seen["body"]["model"] == "Qwen/Qwen3.6-27B"
    assert seen["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert seen["auth"] == "Bearer EMPTY"
    assert agent.last_usage == {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}


def test_run_pipeline_builds_qwen_vllm_api_agent(monkeypatch):
    from scripts.run_pipeline import _build_agent_from_spec

    agent, label = _build_agent_from_spec(
        "qwen36_vllm_server",
        {
            "provider": "qwen_vllm_api",
            "model": "Qwen/Qwen3.6-27B",
            "base_url": "http://127.0.0.1:8000/v1",
            "max_tokens": 64,
            "timeout": 240,
            "enable_thinking": False,
        },
    )

    assert label == "Qwen/Qwen3.6-27B"
    assert isinstance(agent, QwenVLLMAPIAgent)
    assert agent.config.base_url == "http://127.0.0.1:8000/v1"
    assert agent.config.max_tokens == 64
