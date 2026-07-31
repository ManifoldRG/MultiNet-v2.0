"""Qwen chat agent via a vLLM OpenAI-compatible server."""

from __future__ import annotations

import json
import logging
import socket
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, List, Optional

from interface.agents.http_retry import call_with_retry
from interface.agents.qwen_vllm import DEFAULT_QWEN_VLLM_MODEL, _to_openai_messages
from interface.agents.reply import Reply, detect_token_truncated
from interface.telemetry import normalize_token_usage

logger = logging.getLogger(__name__)


def _chat_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/chat/completions"


def _post_chat_completions(
    *,
    base_url: str,
    api_key: str | None,
    model: str,
    max_tokens: int,
    temperature: float,
    messages: List[dict[str, object]],
    timeout: Optional[float],
    enable_thinking: bool | None,
    extra_body: dict[str, Any] | None = None,
    max_attempts: int = 3,
) -> Reply:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0 if temperature <= 0 else temperature,
    }
    if enable_thinking is not None:
        body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    if extra_body:
        body.update(extra_body)

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    raw = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(_chat_url(base_url), data=raw, headers=headers, method="POST")
    effective_timeout = timeout or 180.0
    t0 = time.perf_counter()

    def _do_request() -> dict[str, Any]:
        with urllib.request.urlopen(req, timeout=effective_timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    try:
        payload = call_with_retry(_do_request, max_attempts=max_attempts)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM OpenAI API HTTP {exc.code}: {detail}") from exc
    except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
        if isinstance(exc, urllib.error.URLError) and not isinstance(
            exc.reason, (TimeoutError, socket.timeout)
        ):
            raise RuntimeError(f"vLLM OpenAI API request failed: {exc}") from exc
        raise RuntimeError(f"vLLM OpenAI API timed out after {effective_timeout:.0f}s") from exc

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "vLLM chat/completions: model=%s elapsed=%.2fs",
            model,
            time.perf_counter() - t0,
        )

    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = str(message.get("content") or "").strip()
    usage = normalize_token_usage(payload.get("usage"))
    stop_reason = choice.get("finish_reason")
    # Qwen reasons inline in `content`; there is no separate thinking channel.
    return Reply(
        text=text,
        usage=usage,
        thinking=None,
        stop_reason=stop_reason,
        token_truncated=detect_token_truncated(stop_reason, usage, max_tokens),
    )


@dataclass
class QwenVLLMAPIConfig:
    model: str = DEFAULT_QWEN_VLLM_MODEL
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str | None = "EMPTY"
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: Optional[float] = 180.0
    enable_thinking: bool | None = False
    extra_body: dict[str, Any] = field(default_factory=dict)
    max_attempts: int = 3
    # Max concurrent HTTP requests when fanning out a batch; None → min(len, 16).
    batch_fanout: Optional[int] = None


@dataclass
class QwenVLLMAPIAgent:
    """Qwen via vLLM's OpenAI-compatible HTTP server.

    This lets multiple local worker processes share one vLLM engine, so vLLM can
    batch concurrent maze queries instead of loading one offline engine per
    worker process.
    """

    config: QwenVLLMAPIConfig = field(default_factory=QwenVLLMAPIConfig)
    last_usage: dict[str, int] | None = field(default=None, init=False)
    last_thinking: str | None = field(default=None, init=False)

    def reset_usage(self) -> None:
        self.last_usage = None

    def generate(self, messages: List[dict]) -> Reply:
        return _post_chat_completions(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=_to_openai_messages(messages),
            timeout=self.config.timeout,
            enable_thinking=self.config.enable_thinking,
            extra_body=self.config.extra_body,
            max_attempts=self.config.max_attempts,
        )

    def generate_batch(self, batch: List[List[dict]]) -> List[Reply]:
        """Fan a batch of message lists out over concurrent HTTP requests.

        The vLLM server continuous-batches concurrent requests server-side, so a
        thread pool (one thread blocked per in-flight request) is enough to keep
        the engine saturated. Results are returned in input order regardless of
        completion order. A per-item failure (after ``generate``'s internal
        retries) becomes ``Reply(text="", stop_reason="batch_errored")`` so one
        bad item never sinks the whole batch. ``last_usage`` is left untouched —
        only ``__call__`` writes that side-channel.
        """
        if not batch:
            return []
        max_workers = min(len(batch), self.config.batch_fanout or 16)

        def _one(messages: List[dict]) -> Reply:
            try:
                return self.generate(messages)
            except Exception:  # noqa: BLE001 - isolate per-item failures
                logger.exception("generate_batch item failed after retries")
                return Reply(text="", stop_reason="batch_errored")

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            # Submit in input order; collect via the futures list (not
            # as_completed) so ordering is structural, not timing-dependent.
            futures = [pool.submit(_one, messages) for messages in batch]
            return [future.result() for future in futures]

    def __call__(self, messages: List[dict]) -> str:
        reply = self.generate(messages)
        self.last_usage = reply.usage
        self.last_thinking = reply.thinking
        return reply.text
