"""Kimi K2.6 / K3 agent via the Moonshot OpenAI-compatible API."""

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

from interface.agents.billing import BillingExhausted
from interface.agents.http_retry import call_with_retry
from interface.agents.moonshot_batch import MoonshotBatchDeadline, run_moonshot_batch
from interface.agents.reply import Reply, detect_token_truncated
from interface.agents.runner_messages import strip_cache_markers
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
# kimi-k3 always thinks and takes a different request shape: no `thinking`
# object (the request errors), no sampling params (temperature/top_p are fixed
# and any other value errors), the cap as `max_completion_tokens`, and depth as
# top-level `reasoning_effort` ("low" | "high" | "max"; Moonshot default "max").
_KIMI_K3_PREFIX = "kimi-k3"
_KIMI_K3_REASONING_EFFORTS = ("low", "high", "max")


def _is_k3(model: str) -> bool:
    return model.startswith(_KIMI_K3_PREFIX)


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
        content = strip_cache_markers(message.get("content", ""))
        if role == "assistant" and isinstance(content, str):
            content = content.strip()
        out.append({"role": role, "content": content})
    return out


def _build_body(
    messages: List[Dict[str, object]],
    *,
    model: str,
    max_tokens: int,
    enable_thinking: bool,
    include_sampling: bool = True,
    reasoning_effort: Optional[str] = None,
    cache_ttl: Optional[str] = None,
) -> Dict[str, object]:
    """Assemble the Moonshot chat/completions request body.

    Shared by the sync path and `generate_batch` so `thinking`/`max_tokens`
    handling can never drift. `include_sampling` is the ONE deliberate
    difference: the sync path sends the mode-forced temperature (Moonshot 400s
    on any other value), while the batch path omits sampling params entirely per
    the batch spec. kimi-k3 never gets `thinking` or sampling params. Only the
    sync path passes `cache_ttl` (`prompt_cache_options`): Moonshot's cache-write
    TTL tiers are a kimi-k3 feature, and its batch support for kimi-k3 is unverified.
    """
    if _is_k3(model):
        body: Dict[str, object] = {
            "model": model,
            "max_completion_tokens": max_tokens,
            "messages": messages,
        }
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
        if cache_ttl:
            body["prompt_cache_options"] = {"mode": "implicit", "ttl": cache_ttl}
        return body
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        "thinking": {"type": "enabled" if enable_thinking else "disabled"},
    }
    if include_sampling:
        # Moonshot pins the temperature per mode (see constants) — send the only
        # value it accepts for this thinking mode, ignoring the configured value,
        # or the request 400s.
        body["temperature"] = (
            _KIMI_TEMPERATURE_THINKING if enable_thinking else _KIMI_TEMPERATURE_NO_THINKING
        )
    if cache_ttl:
        body["prompt_cache_options"] = {"mode": "implicit", "ttl": cache_ttl}
    return body


def _reply_from_completion(payload: Dict[str, object], *, max_tokens: int) -> Reply:
    """Turn one chat/completions body into a Reply.

    Shared by the sync path and `generate_batch` so succeeded batch bodies parse
    identically to sync responses.
    """
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


def _post_chat_completions(
    api_key: str,
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    messages: List[Dict[str, object]],
    timeout: Optional[float],
    enable_thinking: bool,
    cache_ttl: Optional[str] = None,
    max_attempts: int = 5,
    reasoning_effort: Optional[str] = None,
) -> Reply:
    # `temperature` stays in the signature for provenance/cache-hash parity but
    # the mode-forced value is what actually ships (see `_build_body`).
    body = _build_body(
        messages,
        model=model,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
        cache_ttl=cache_ttl,
    )

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
    # Per-call socket timeout comes only from KimiK26Config (set via run_config's
    # model_config["timeout"]). Deep-thinking (64k) runs must configure a value
    # well above 600s — Moonshot returns legitimate sub-64k replies slowly. The
    # R1 KIMI_TIMEOUT_OVERRIDE env stopgap (2026-07-21) was removed post-campaign
    # (2026-07-22) in favour of the run_config default; see analysis/R1_TEARDOWN.md.
    effective_timeout = timeout or 180.0
    t0 = time.perf_counter()

    def _do_request():
        with urllib.request.urlopen(req, timeout=effective_timeout) as resp:
            return json.loads(resp.read().decode())

    try:
        payload = call_with_retry(_do_request, max_attempts=max_attempts)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        if any(marker in detail.lower() for marker in ("insufficient balance", "exceeded_current_quota")):
            raise BillingExhausted("kimi", f"Moonshot API HTTP {exc.code}: {detail}") from exc
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

    return _reply_from_completion(payload, max_tokens=max_tokens)


@dataclass
class KimiK26Config:
    model: str = DEFAULT_KIMI_K26_MODEL
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: Optional[float] = 180.0
    enable_thinking: bool = False
    # kimi-k3 only: "low" | "high" | "max"; None sends nothing (Moonshot default).
    reasoning_effort: Optional[str] = None
    prompt_cache_ttl: Optional[str] = None
    max_attempts: int = 5
    # Moonshot Batch API knobs (used only by generate_batch).
    batch_poll_interval_s: float = 30.0
    batch_deadline_s: float = 7200.0
    batch_cancel_grace_s: float = 300.0
    batch_completion_window: str = "24h"


@dataclass
class KimiK26Agent:
    """Kimi K2.6 / K3 via Moonshot API (`MOONSHOT_API_KEY`). Supports vision user turns."""

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
        if _is_k3(self.config.model):
            # K3 cannot disable thinking; a thinking-off config would mislabel the run.
            if not self.config.enable_thinking:
                raise ValueError(f"{self.config.model} always thinks; set enable_thinking: true.")
            effort = self.config.reasoning_effort
            if effort is not None and effort not in _KIMI_K3_REASONING_EFFORTS:
                raise ValueError(
                    f"{self.config.model} reasoning_effort must be one of "
                    f"{_KIMI_K3_REASONING_EFFORTS}, got {effort!r}."
                )
        elif self.config.reasoning_effort is not None:
            raise ValueError(f"{self.config.model} does not take reasoning_effort (kimi-k3 only).")

    def generate(self, messages: List[dict]) -> Reply:
        return _post_chat_completions(
            self.api_key,
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=_to_openai_messages(messages),
            timeout=self.config.timeout,
            enable_thinking=self.config.enable_thinking,
            cache_ttl=self.config.prompt_cache_ttl,
            max_attempts=self.config.max_attempts,
            reasoning_effort=self.config.reasoning_effort,
        )

    def generate_batch(self, batch: List[List[dict]]) -> List[Reply]:
        """Run every message list through the Moonshot Batch API, in input order.

        KIMI_BATCH_TRANSPORT=sync switches the transport to bounded-concurrency
        sync chat completions (full price, no Batch API dependency) while
        keeping the lockstep round contract: one Reply per input, in order,
        per-item failures isolated as ``sync_error`` stubs. KIMI_SYNC_FANOUT
        bounds the thread pool (default 15).

        Bodies are built by the same `_build_body` as sync (with sampling params
        omitted per the batch spec) and succeeded bodies parse through the same
        `_reply_from_completion`. Does not touch ``last_usage``/``last_thinking``
        — batch calls carry their result in the returned ``Reply`` so N calls can
        be in flight at once. Ids absent from a completed batch (errors landed in
        the untouched error_file) become ``batch_errored`` stubs; ids missing
        after a poll-deadline cancel become ``batch_expired``.
        """
        if os.environ.get("KIMI_BATCH_TRANSPORT", "batch") == "sync":
            from concurrent.futures import ThreadPoolExecutor

            def _one(messages: List[dict]) -> Reply:
                try:
                    return self.generate(messages)
                except Exception:
                    logging.getLogger(__name__).warning(
                        "kimi sync-transport item failed", exc_info=True
                    )
                    return Reply(text="", stop_reason="sync_error")

            fanout = int(os.environ.get("KIMI_SYNC_FANOUT", "15"))
            with ThreadPoolExecutor(max_workers=max(1, fanout)) as pool:
                return list(pool.map(_one, batch))

        lines = [
            {
                "custom_id": f"i{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": _build_body(
                    _to_openai_messages(messages),
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    enable_thinking=self.config.enable_thinking,
                    include_sampling=False,
                    reasoning_effort=self.config.reasoning_effort,
                ),
            }
            for i, messages in enumerate(batch)
        ]
        try:
            results = run_moonshot_batch(
                lines,
                api_key=self.api_key,
                poll_interval_s=self.config.batch_poll_interval_s,
                deadline_s=self.config.batch_deadline_s,
                cancel_grace_s=self.config.batch_cancel_grace_s,
                completion_window=self.config.batch_completion_window,
            )
            missing_stop_reason = "batch_errored"
        except MoonshotBatchDeadline as exc:
            results = exc.results
            missing_stop_reason = "batch_expired"

        replies: List[Reply] = []
        for i in range(len(batch)):
            body = results.get(f"i{i}")
            if body is None:
                replies.append(Reply(text="", stop_reason=missing_stop_reason))
            else:
                replies.append(_reply_from_completion(body, max_tokens=self.config.max_tokens))
        return replies

    def __call__(self, messages: List[dict]) -> str:
        reply = self.generate(messages)
        self.last_usage = reply.usage
        self.last_thinking = reply.thinking
        return reply.text
