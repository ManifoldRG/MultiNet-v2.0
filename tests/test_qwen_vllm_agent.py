from __future__ import annotations

import sys
import types

from interface.agents.qwen_vllm import QwenVLLMAgent, QwenVLLMConfig, _to_openai_messages


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
