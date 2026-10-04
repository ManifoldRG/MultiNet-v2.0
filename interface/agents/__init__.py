"""LLM agents for the interface runner."""

from __future__ import annotations

import os

# Stable defaults for HF Hub downloads on Windows (local Transformers path).
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from interface.agents.claude import (
    DEFAULT_CLAUDE_MODEL,
    ClaudeAnthropicAgent,
    ClaudeAnthropicConfig,
)
from interface.agents.kimi_k26 import (
    DEFAULT_KIMI_K26_MODEL,
    KimiK26Agent,
    KimiK26Config,
)
from interface.agents.qwen35_vl import (
    DEFAULT_QWEN35_VL_MODEL,
    Qwen35VLAgent,
    Qwen35VLConfig,
)
from interface.agents.qwen_vllm import (
    DEFAULT_QWEN_VLLM_MODEL,
    QwenVLLMAgent,
    QwenVLLMConfig,
)
from interface.agents.qwen_vllm_api import (
    QwenVLLMAPIAgent,
    QwenVLLMAPIConfig,
)

__all__ = [
    "DEFAULT_CLAUDE_MODEL",
    "DEFAULT_KIMI_K26_MODEL",
    "DEFAULT_QWEN35_VL_MODEL",
    "DEFAULT_QWEN_VLLM_MODEL",
    "ClaudeAnthropicAgent",
    "ClaudeAnthropicConfig",
    "KimiK26Agent",
    "KimiK26Config",
    "Qwen35VLAgent",
    "Qwen35VLConfig",
    "QwenVLLMAgent",
    "QwenVLLMAPIAgent",
    "QwenVLLMAPIConfig",
    "QwenVLLMConfig",
]
