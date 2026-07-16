"""Kimi K2.6 agent via the Moonshot OpenAI-compatible API."""

from __future__ import annotations

import json
import logging
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from interface.agents.http_retry import call_with_retry
from interface.agents.reply import Reply, detect_token_truncated
from interface.telemetry import normalize_token_usage

logger = logging.getLogger(__name__)

DEFAULT_KIMI_K26_MODEL = "kimi-k2.6"
_MOONSHOT_CHAT_URL = "https://api.moonshot.ai/v1/chat/completions"
_AGENT_NAME = "Kimi agent"
# Moonshot kimi-k2.6 dictates the sampling temperature BY MODE and returns HTTP 400
# on any other value ("only X is allowed for this model"): thinking-on requires 1.0,
# thinking-off requires 0.6. Confirmed live via the M6 smoke (2026-07-02).
_KIMI_TEMPERATURE_THINKING = 1.0
_KIMI_TEMPERATURE_NO_THINKING = 0.6


# Moonshot caches identical request prefixes automatically and bills the reused
# span at the cache-hit input rate (no per-message cache_control field exists in
# the OpenAI-compatible schema). Because the agent re-sends an append-only history
# at a fixed temperature, the stable system+history prefix is cache-eligible as-is.
def _to_openai_messages(messages: List[dict]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for message in messages:
        role = message.get("role")
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"Unsupported message role for {_AGENT_NAME}: {role!r}")
        content = message.get("content", "")
        if role == "assistant" and isinstance(content, str):
            content = content.strip()
        out.append({"role": role, "content": content})
    return out


def _post_chat_completions(
    api_key: str,
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    messages: List[Dict[str, object]],
    timeout: Optional[float],
    enable_thinking: bool,
    max_attempts: int = 5,
) -> Reply:
    body: Dict[str, object] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        # Moonshot pins the temperature per mode (see constants) — send the only
        # value it accepts for this thinking mode, ignoring the configured value,
        # or the request 400s. `temperature` is kept in the signature/hash for
        # provenance but is not sent verbatim.
        "temperature": (
            _KIMI_TEMPERATURE_THINKING if enable_thinking else _KIMI_TEMPERATURE_NO_THINKING
        ),
        "thinking": {"type": "enabled" if enable_thinking else "disabled"},
    }

    raw = json.dumps(body).encode("utf-8")
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Moonshot request: model=%s json_bytes=%d", model, len(raw))

    req = urllib.request.Request(
        _MOONSHOT_CHAT_URL,
        data=raw,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    effective_timeout = timeout or 180.0
    t0 = time.perf_counter()

    def _do_request():
        with urllib.request.urlopen(req, timeout=effective_timeout) as resp:
            return json.loads(resp.read().decode())

    try:
        payload = call_with_retry(_do_request, max_attempts=max_attempts)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"Moonshot API HTTP {exc.code}: {detail}") from exc
    except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
        if isinstance(exc, urllib.error.URLError) and not isinstance(
            exc.reason, (TimeoutError, socket.timeout)
        ):
            raise RuntimeError(f"Moonshot API request failed: {exc}") from exc
        raise RuntimeError(
            f"Moonshot API timed out after {effective_timeout:.0f}s "
            f"(vision payloads can be slow; pass a larger timeout via KimiK26Config)."
        ) from exc

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Moonshot chat/completions: model=%s elapsed=%.2fs",
            model,
            time.perf_counter() - t0,
        )

    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = str(message.get("content") or "").strip()
    usage = normalize_token_usage(payload.get("usage"))
    stop_reason = choice.get("finish_reason")
    # Moonshot returns the thinking trace out-of-band in `reasoning_content`
    # (the OpenAI schema has no such field); capture it — the pipeline otherwise
    # discards it.
    reasoning = message.get("reasoning_content")
    thinking = str(reasoning).strip() or None if reasoning else None
    return Reply(
        text=text,
        usage=usage,
        thinking=thinking,
        stop_reason=stop_reason,
        token_truncated=detect_token_truncated(stop_reason, usage, max_tokens),
    )


@dataclass
class KimiK26Config:
    model: str = DEFAULT_KIMI_K26_MODEL
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: Optional[float] = 180.0
    enable_thinking: bool = False
    max_attempts: int = 5


@dataclass
class KimiK26Agent:
    """Kimi K2.6 via Moonshot API (`MOONSHOT_API_KEY`). Supports vision user turns."""

    config: KimiK26Config = field(default_factory=KimiK26Config)
    api_key: Optional[str] = None
    last_usage: Optional[Dict[str, int]] = field(default=None, init=False)
    last_thinking: Optional[str] = field(default=None, init=False)

    def __post_init__(self) -> None:
        key = (self.api_key or os.environ.get("MOONSHOT_API_KEY") or "").strip()
        if not key:
            raise ValueError(
                "No Moonshot API key found. Set MOONSHOT_API_KEY or pass api_key=... "
                "to KimiK26Agent."
            )
        self.api_key = key

    def generate(self, messages: List[dict]) -> Reply:
        return _post_chat_completions(
            self.api_key,
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=_to_openai_messages(messages),
            timeout=self.config.timeout,
            enable_thinking=self.config.enable_thinking,
            max_attempts=self.config.max_attempts,
        )

    def __call__(self, messages: List[dict]) -> str:
        reply = self.generate(messages)
        self.last_usage = reply.usage
        self.last_thinking = reply.thinking
        return reply.text
