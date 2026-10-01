"""OpenAI reasoning-model agent (GPT-6 Astra, GPT-5.6 Sol/Terra/Luna) via Chat Completions.

Wire shape follows the OpenAI reasoning-model rules (developers.openai.com, 2026-09):
``max_completion_tokens`` caps visible AND reasoning tokens together;
``reasoning_effort`` selects depth (Astra: low/medium/high/xhigh/max — ``none``
400s); ``temperature``/``top_p`` are unsupported on reasoning models, so no
sampling parameter is ever sent. Chat Completions returns no reasoning text —
only its size, in ``usage.completion_tokens_details.reasoning_tokens`` (billed as
output), which is surfaced as ``usage["reasoning_tokens"]``.

``service_tier="flex"`` bills at Batch-API rates on synchronous calls; when flex
capacity is short OpenAI answers ``429 Resource Unavailable`` (not charged), which
the shared retry helper backs off on like any other 429.

Optional hard spend cap (``spend_cap_usd``): a per-process running total of
priced usage across every agent instance; once reached, further requests raise
instead of spending. Restarting the process resets it, so pair it with an
OpenAI project budget limit.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

from interface.agents.http_retry import call_with_retry
from interface.agents.reply import Reply, detect_token_truncated
from interface.telemetry import normalize_token_usage

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_MODEL = "gpt-6-astra"
_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_AGENT_NAME = "OpenAI agent"

# USD per 1M tokens: (input, cached input, output). Source:
# https://developers.openai.com/api/docs/pricing (fetched 2026-09-12). Flex and
# Batch share one rate; "default" is the standard tier. Long-context (>272K
# input) surcharges never apply to this benchmark's ~1K-token prompts.
OPENAI_PRICES: Dict[str, Dict[str, Tuple[float, float, float]]] = {
    "gpt-6-astra": {"default": (10.00, 1.00, 50.00), "flex": (5.00, 0.50, 25.00)},
    # gpt-6-sol / gpt-6-luna: same pricing page, fetched 2026-09-26.
    "gpt-6-sol": {"default": (2.00, 0.20, 10.00), "flex": (1.00, 0.10, 5.00)},
    "gpt-6-luna": {"default": (0.10, 0.01, 0.50), "flex": (0.05, 0.005, 0.25)},
    "gpt-5.6-sol": {"default": (4.00, 0.40, 20.00), "flex": (2.00, 0.20, 10.00)},
    "gpt-5.6-terra": {"default": (2.00, 0.20, 12.00), "flex": (1.00, 0.10, 6.00)},
    "gpt-5.6-luna": {"default": (0.20, 0.02, 1.20), "flex": (0.10, 0.01, 0.60)},
}

_RATE_LIMIT_HEADERS = (
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens",
)


def price_for(model: str, service_tier: Optional[str]) -> Tuple[float, float, float]:
    """Per-1M-token (input, cached input, output) USD for ``model`` on ``service_tier``."""
    tier = "flex" if service_tier == "flex" else "default"
    try:
        return OPENAI_PRICES[model][tier]
    except KeyError:
        raise ValueError(f"No OpenAI price on record for model {model!r}") from None


def usage_cost_usd(usage: Optional[Mapping[str, int]], prices: Tuple[float, float, float]) -> float:
    """Price one normalized usage dict. Cached input is billed at the cached rate."""
    if not usage:
        return 0.0
    p_in, p_cached, p_out = prices
    total_in = int(usage.get("input_tokens", 0))
    cached = min(int(usage.get("cached_input_tokens", 0)), total_in)
    out = int(usage.get("output_tokens", 0))
    return ((total_in - cached) * p_in + cached * p_cached + out * p_out) / 1_000_000


class _SpendLedger:
    """Process-wide priced-usage total, shared by every agent instance."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.usd = 0.0

    def add(self, usd: float) -> float:
        with self._lock:
            self.usd += usd
            return self.usd

    def reset(self) -> None:
        with self._lock:
            self.usd = 0.0


SPEND = _SpendLedger()


class OpenAISpendCapReached(RuntimeError):
    """Raised before a request once the process-wide spend cap is reached."""


def _to_openai_messages(messages: List[dict], image_detail: Optional[str]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for message in messages:
        role = message.get("role")
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"Unsupported message role for {_AGENT_NAME}: {role!r}")
        content = message.get("content", "")
        if role == "assistant" and isinstance(content, str):
            content = content.strip()
        if image_detail and isinstance(content, list):
            content = [
                {**block, "image_url": {**block["image_url"], "detail": image_detail}}
                if block.get("type") == "image_url" and isinstance(block.get("image_url"), dict)
                else block
                for block in content
            ]
        out.append({"role": role, "content": content})
    return out


def _build_body(
    messages: List[Dict[str, object]],
    *,
    model: str,
    max_tokens: int,
    reasoning_effort: Optional[str],
    service_tier: Optional[str],
) -> Dict[str, object]:
    body: Dict[str, object] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
    }
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    if service_tier:
        body["service_tier"] = service_tier
    return body


def _usage_from_payload(raw: object) -> Optional[Dict[str, int]]:
    usage = normalize_token_usage(raw)
    if usage is None or not isinstance(raw, dict):
        return usage
    reasoning = (raw.get("completion_tokens_details") or {}).get("reasoning_tokens")
    if reasoning is not None:
        usage["reasoning_tokens"] = int(reasoning)
    # Deliberately NOT ``cache_read_input_tokens``: normalize_token_usage folds that
    # Anthropic key back INTO input_tokens, and OpenAI's prompt_tokens already
    # includes cached tokens — reusing the name would double-count on re-normalize.
    cached = (raw.get("prompt_tokens_details") or {}).get("cached_tokens")
    if cached is not None:
        usage["cached_input_tokens"] = int(cached)
    return usage


def _reply_from_completion(payload: Dict[str, object], *, max_tokens: int) -> Reply:
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = str(message.get("content") or "").strip()
    if not text and message.get("refusal"):
        logger.warning("%s: model refused: %s", _AGENT_NAME, str(message["refusal"])[:200])
    usage = _usage_from_payload(payload.get("usage"))
    stop_reason = choice.get("finish_reason")
    return Reply(
        text=text,
        usage=usage,
        thinking=None,
        stop_reason=stop_reason,
        token_truncated=detect_token_truncated(stop_reason, usage, max_tokens),
    )


@dataclass
class OpenAIConfig:
    model: str = DEFAULT_OPENAI_MODEL
    # Provenance only: reasoning models reject sampling params, so it is never sent.
    temperature: float = 0.0
    # Sent as max_completion_tokens (reasoning + visible output together).
    max_tokens: int = 4096
    reasoning_effort: Optional[str] = None
    service_tier: Optional[str] = None
    # "low" | "high" | "auto"; None leaves the API default.
    image_detail: Optional[str] = None
    timeout: Optional[float] = 900.0
    max_attempts: int = 5
    spend_cap_usd: Optional[float] = None
    base_url: str = _DEFAULT_BASE_URL


@dataclass
class OpenAIAgent:
    """OpenAI reasoning model via Chat Completions (`OPENAI_API_KEY`). Supports vision user turns."""

    config: OpenAIConfig = field(default_factory=OpenAIConfig)
    api_key: Optional[str] = None
    last_usage: Optional[Dict[str, int]] = field(default=None, init=False)
    last_thinking: Optional[str] = field(default=None, init=False)
    last_rate_limits: Dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        key = (self.api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
        if not key:
            raise ValueError(
                "No OpenAI API key found. Set OPENAI_API_KEY or pass api_key=... to OpenAIAgent."
            )
        self.api_key = key
        if self.config.reasoning_effort == "none":
            # Astra 400s on "none"; the whole R1 comparison is thinking-on anyway.
            raise ValueError("reasoning_effort='none' is rejected by gpt-6-astra; pick low..max.")
        # Fail closed at construction: a spend cap we cannot price is no cap at all.
        self._prices = (
            price_for(self.config.model, self.config.service_tier)
            if self.config.spend_cap_usd is not None
            else None
        )

    def _post(self, body: Dict[str, object]) -> Tuple[Dict[str, object], Dict[str, str]]:
        req = urllib.request.Request(
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        timeout = self.config.timeout or 900.0

        def _do_request():
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    headers = {
                        h: resp.headers.get(h)
                        for h in _RATE_LIMIT_HEADERS
                        if getattr(resp, "headers", None) is not None and resp.headers.get(h)
                    }
                    return json.loads(resp.read().decode()), headers
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")
                # Billing exhaustion is also a 429 but will never clear on retry.
                if exc.code == 429 and "insufficient_quota" in detail:
                    raise RuntimeError(
                        f"OpenAI API quota exhausted (HTTP 429 insufficient_quota): {detail}"
                    ) from exc
                exc.detail = detail  # type: ignore[attr-defined]
                raise

        try:
            return call_with_retry(_do_request, max_attempts=self.config.max_attempts)
        except urllib.error.HTTPError as exc:
            detail = getattr(exc, "detail", "")
            raise RuntimeError(f"OpenAI API HTTP {exc.code}: {detail}") from exc
        except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
            if isinstance(exc, urllib.error.URLError) and not isinstance(
                exc.reason, (TimeoutError, socket.timeout)
            ):
                raise RuntimeError(f"OpenAI API request failed: {exc}") from exc
            raise RuntimeError(f"OpenAI API timed out after {timeout:.0f}s.") from exc

    def generate(self, messages: List[dict]) -> Reply:
        cap = self.config.spend_cap_usd
        if cap is not None and SPEND.usd >= cap:
            raise OpenAISpendCapReached(
                f"OpenAI spend cap ${cap:.2f} reached (running total ${SPEND.usd:.2f}); refusing request."
            )
        body = _build_body(
            _to_openai_messages(messages, self.config.image_detail),
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            reasoning_effort=self.config.reasoning_effort,
            service_tier=self.config.service_tier,
        )
        payload, headers = self._post(body)
        if headers:
            self.last_rate_limits = headers
        reply = _reply_from_completion(payload, max_tokens=self.config.max_tokens)
        if self._prices is not None:
            SPEND.add(usage_cost_usd(reply.usage, self._prices))
        return reply

    def __call__(self, messages: List[dict]) -> str:
        reply = self.generate(messages)
        self.last_usage = reply.usage
        self.last_thinking = reply.thinking
        return reply.text
