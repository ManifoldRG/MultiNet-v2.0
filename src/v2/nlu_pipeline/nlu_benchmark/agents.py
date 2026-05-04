from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer

# Stable defaults for HF Hub downloads on Windows (local Transformers path).
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


DEFAULT_LOCAL_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"


DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-20250514"


def _parse_data_image_url(url: str) -> tuple[str, str]:
    """Split ``data:<mime>;base64,<data>`` into media type and raw base64 payload."""
    if not isinstance(url, str) or not url.startswith("data:"):
        raise ValueError("Expected a data: URL with base64 image payload.")
    rest = url[5:]
    if ";base64," not in rest:
        raise ValueError("Expected ';base64,' in image data URL.")
    meta, _, b64 = rest.partition(";base64,")
    media_type = (meta.strip() or "image/png").split(";")[0].strip()
    return media_type, b64.strip()


def _openai_blocks_to_anthropic(blocks: List[dict]) -> List[dict]:
    """Convert runner/OpenAI-style content blocks to Anthropic Messages ``content`` blocks."""
    out: List[dict] = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t == "text":
            out.append({"type": "text", "text": str(b.get("text", ""))})
        elif t == "image_url":
            url_holder = b.get("image_url")
            url = url_holder.get("url") if isinstance(url_holder, dict) else url_holder
            if isinstance(url, str) and url.startswith("data:"):
                mt, raw_b64 = _parse_data_image_url(url)
                out.append({"type": "image", "source": {"type": "base64", "media_type": mt, "data": raw_b64}})
    return out


def _anthropic_turn_content(content: object, role: str) -> object:
    if isinstance(content, str):
        return content if role != "assistant" else content.strip()
    if isinstance(content, list):
        anthropic_blocks = _openai_blocks_to_anthropic(content)
        if not anthropic_blocks:
            return ""
        if len(anthropic_blocks) == 1 and anthropic_blocks[0].get("type") == "text":
            return str(anthropic_blocks[0].get("text", ""))
        return anthropic_blocks
    return str(content)


def _anthropic_chat_turns(messages: List[dict]) -> Tuple[Optional[str], List[Dict[str, object]]]:
    """Split OpenAI-style chat messages into Anthropic `system` + user/assistant turns."""
    system_parts: List[str] = []
    turns: List[Dict[str, object]] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "system":
            system_parts.append(str(content))
        elif role in ("user", "assistant"):
            turns.append({"role": role, "content": _anthropic_turn_content(content, role)})
        else:
            raise ValueError(f"Unsupported message role for Claude agent: {role!r}")
    system = "\n\n".join(system_parts) if system_parts else None
    return system, turns


def _anthropic_messages_http(
    api_key: str,
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    system: Optional[str],
    messages: List[Dict[str, object]],
    timeout: Optional[float],
) -> str:
    """POST /v1/messages (Anthropic Messages API); uses stdlib only."""
    body: Dict[str, object] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        "temperature": temperature,
    }
    if system:
        body["system"] = system

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout or 60.0) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise RuntimeError(f"Anthropic API HTTP {e.code}: {detail}") from e

    parts: List[str] = []
    for block in payload.get("content", []) or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts).strip()


@dataclass
class ClaudeAnthropicConfig:
    model: str = DEFAULT_CLAUDE_MODEL
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: Optional[float] = 60.0


@dataclass
class ClaudeAnthropicAgent:
    """Claude via Anthropic Messages API (`ANTHROPIC_API_KEY`). Supports vision user turns."""

    config: ClaudeAnthropicConfig = field(default_factory=ClaudeAnthropicConfig)
    api_key: Optional[str] = None

    def __post_init__(self) -> None:
        key = (self.api_key or os.environ.get("ANTHROPIC_API_KEY") or "").strip()
        if not key:
            raise ValueError(
                "No Anthropic API key found. Set ANTHROPIC_API_KEY or pass api_key=... to ClaudeAnthropicAgent."
            )
        self.api_key = key

    def __call__(self, messages: List[dict]) -> str:
        system, turns = _anthropic_chat_turns(messages)
        return _anthropic_messages_http(
            self.api_key,
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=system,
            messages=turns,
            timeout=self.config.timeout,
        )


@dataclass
class LocalLLMConfig:
    # Open-source/open-weight local models (examples):
    # - Qwen/Qwen2.5-0.5B-Instruct
    # - google/gemma-2-2b-it
    model: str = DEFAULT_LOCAL_MODEL
    temperature: float = 0.0
    max_new_tokens: int = 64
    device_map: str = "auto"


@dataclass
class LocalTransformersAgent:
    """Local agent using Hugging Face Transformers (no inference credits)."""

    config: LocalLLMConfig = field(default_factory=LocalLLMConfig)
    tokenizer: Optional[AutoTokenizer] = None
    model: Optional[AutoModelForCausalLM] = None

    def __post_init__(self) -> None:
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model)
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model,
                device_map=self.config.device_map,
            )

    def __call__(self, messages: List[Dict[str, str]]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            do_sample=self.config.temperature > 0,
        )

        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = generated[0][prompt_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


if __name__ == "__main__":
    agent = LocalTransformersAgent(config=LocalLLMConfig())
    out = agent(
        [
            {"role": "system", "content": "Reply with one short sentence."},
            {"role": "user", "content": "What is 2+2?"},
        ]
    )
    print(out)
