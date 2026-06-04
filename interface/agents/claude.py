"""Claude Sonnet agent via the Anthropic Messages API."""

from __future__ import annotations

import json
import logging
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from interface.agents.runner_messages import (
    ContentPart,
    parse_runner_content,
    split_system_prompt,
)

logger = logging.getLogger(__name__)

DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"
_AGENT_NAME = "Claude agent"


def _parts_to_anthropic(parts: List[ContentPart]) -> List[dict]:
    blocks: List[dict] = []
    for part in parts:
        if part.kind == "text":
            blocks.append({"type": "text", "text": part.text})
        elif part.image is not None:
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": part.image.media_type,
                        "data": part.image.data_b64,
                    },
                }
            )
    return blocks


def _anthropic_turn_content(content: object, role: str) -> object:
    if isinstance(content, str):
        return content.strip() if role == "assistant" else content
    parsed = parse_runner_content(content)
    if isinstance(parsed, str):
        return parsed.strip() if role == "assistant" else parsed
    blocks = _parts_to_anthropic(parsed)
    if not blocks:
        return ""
    if len(blocks) == 1 and blocks[0].get("type") == "text":
        text = str(blocks[0].get("text", ""))
        return text.strip() if role == "assistant" else text
    return blocks


def _to_anthropic_turns(messages: List[dict]) -> Tuple[Optional[str], List[Dict[str, object]]]:
    system, raw_turns = split_system_prompt(messages)
    turns: List[Dict[str, object]] = []
    for message in raw_turns:
        role = message.get("role")
        if role not in ("user", "assistant"):
            raise ValueError(f"Unsupported message role for {_AGENT_NAME}: {role!r}")
        turns.append(
            {
                "role": role,
                "content": _anthropic_turn_content(message.get("content", ""), role),
            }
        )
    return system, turns


def _post_messages(
    api_key: str,
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    system: Optional[str],
    messages: List[Dict[str, object]],
    timeout: Optional[float],
) -> str:
    body: Dict[str, object] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        "temperature": temperature,
    }
    if system:
        body["system"] = system

    raw = json.dumps(body).encode("utf-8")
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Anthropic request: model=%s json_bytes=%d", model, len(raw))

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=raw,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    effective_timeout = timeout or 180.0
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=effective_timeout) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"Anthropic API HTTP {exc.code}: {detail}") from exc
    except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
        if isinstance(exc, urllib.error.URLError) and not isinstance(
            exc.reason, (TimeoutError, socket.timeout)
        ):
            raise RuntimeError(f"Anthropic API request failed: {exc}") from exc
        raise RuntimeError(
            f"Anthropic API timed out after {effective_timeout:.0f}s "
            f"(vision payloads can be slow; pass a larger timeout via ClaudeAnthropicConfig)."
        ) from exc

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Anthropic Messages API: model=%s elapsed=%.2fs",
            model,
            time.perf_counter() - t0,
        )

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
    timeout: Optional[float] = 180.0


@dataclass
class ClaudeAnthropicAgent:
    """Claude via Anthropic Messages API (`ANTHROPIC_API_KEY`). Supports vision user turns."""

    config: ClaudeAnthropicConfig = field(default_factory=ClaudeAnthropicConfig)
    api_key: Optional[str] = None

    def __post_init__(self) -> None:
        key = (self.api_key or os.environ.get("ANTHROPIC_API_KEY") or "").strip()
        if not key:
            raise ValueError(
                "No Anthropic API key found. Set ANTHROPIC_API_KEY or pass api_key=... "
                "to ClaudeAnthropicAgent."
            )
        self.api_key = key

    def __call__(self, messages: List[dict]) -> str:
        system, turns = _to_anthropic_turns(messages)
        return _post_messages(
            self.api_key,
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=system,
            messages=turns,
            timeout=self.config.timeout,
        )
