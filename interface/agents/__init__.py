"""LLM agents for the interface runner (Claude Sonnet API, Qwen 3.5 VL local)."""

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
from interface.agents.qwen35_vl import (
    DEFAULT_QWEN35_VL_MODEL,
    Qwen35VLAgent,
    Qwen35VLConfig,
)

__all__ = [
    "DEFAULT_CLAUDE_MODEL",
    "DEFAULT_QWEN35_VL_MODEL",
    "ClaudeAnthropicAgent",
    "ClaudeAnthropicConfig",
    "Qwen35VLAgent",
    "Qwen35VLConfig",
]
