"""Claude agent (Sonnet/Opus/Fable) via the Anthropic Messages API."""

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

from interface.agents.anthropic_batch import run_message_batch
from interface.agents.http_retry import call_with_retry
from interface.agents.reply import Reply, detect_token_truncated
from interface.agents.runner_messages import (
    ContentPart,
    parse_runner_content,
    split_system_prompt,
)
from interface.telemetry import normalize_token_usage

logger = logging.getLogger(__name__)

DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"
_AGENT_NAME = "Claude agent"

# Model families that REJECT sampling params (temperature/top_p/top_k) with an
# HTTP 400: Opus 4.7+, Fable, Mythos. Sonnet 4.6 and Opus <=4.6 still accept
# `temperature`, so we only drop it for these. Depth on the thinking families is
# controlled by adaptive thinking + `output_config.effort`, never `temperature`.
_SAMPLING_UNSUPPORTED_PREFIXES = (
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-fable",
    "claude-mythos",
)


def _omits_sampling_params(model: str) -> bool:
    m = (model or "").lower()
    return any(m.startswith(prefix) for prefix in _SAMPLING_UNSUPPORTED_PREFIXES)


def _build_request_body(
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    system: object,
    messages: List[Dict[str, object]],
    enable_thinking: bool = False,
    effort: Optional[str] = None,
) -> Dict[str, object]:
    """Assemble the Anthropic Messages request body.

    `temperature` is omitted for model families that reject it (see
    `_omits_sampling_params`); sending it there would 400. Thinking is opt-in:
    when enabled we send adaptive thinking, and `effort` (low..max) tunes depth.
    """
    body: Dict[str, object] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if not _omits_sampling_params(model):
        body["temperature"] = temperature
    if enable_thinking:
        # display="summarized" makes thinking text loggable — Opus 4.8 defaults
        # to "omitted", which would leave Reply.thinking empty for the whole paid
        # run (billing is identical either way). Per
        # docs/batch-api-lockstep-runner-design.md. Shared by the sync and batch
        # paths (both build the body here), so they can never drift.
        body["thinking"] = {"type": "adaptive", "display": "summarized"}
    if effort:
        output_config = body.setdefault("output_config", {})
        assert isinstance(output_config, dict)
        output_config["effort"] = effort
    if system:
        body["system"] = system
    return body


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


_CACHE_CONTROL = {"type": "ephemeral"}


def _as_block_list(content: object) -> List[dict]:
    if isinstance(content, list):
        return [dict(block) for block in content]
    return [{"type": "text", "text": "" if content is None else str(content)}]


def _apply_prompt_cache(
    system: Optional[str], turns: List[Dict[str, object]]
) -> Tuple[object, List[Dict[str, object]]]:
    """Add cache_control breakpoints so each turn reuses the prior prompt prefix.

    Two breakpoints (well under the 4-breakpoint cap): the stable system prompt,
    and the last content block of the most-recently-appended turn. Each next call
    reads the prefix the previous call wrote. Anthropic returns the cache split in
    `usage.cache_read_input_tokens`/`cache_creation_input_tokens`, which
    `normalize_token_usage` folds back into the full input-token count.
    """
    system_out: object = system
    if system:
        system_out = [{"type": "text", "text": system, "cache_control": _CACHE_CONTROL}]
    if turns:
        last = dict(turns[-1])
        blocks = _as_block_list(last["content"])
        # Skip an empty trailing text block: Anthropic rejects a cache_control on an
        # empty text block (and it would never be a useful cache breakpoint anyway).
        if blocks and not (blocks[-1].get("type") == "text" and not blocks[-1].get("text")):
            blocks[-1] = {**blocks[-1], "cache_control": _CACHE_CONTROL}
            last["content"] = blocks
            turns = turns[:-1] + [last]
    return system_out, turns


def _parse_response(
    payload: Dict[str, object],
) -> Tuple[str, Optional[Dict[str, int]], Optional[str]]:
    """Split an Anthropic Messages response into (final text, usage, thinking).

    The visible answer is the concatenation of `text` blocks; the model's
    reasoning lives in separate `thinking` blocks which the pipeline otherwise
    discards (only ~a handful of the reported output tokens are the FINAL_OUTPUT).
    We keep the thinking so the episode log can show *why* the model chose an
    action.
    """
    parts: List[str] = []
    thinking_parts: List[str] = []
    for block in payload.get("content", []) or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            parts.append(str(block.get("text", "")))
        elif block.get("type") == "thinking":
            thinking_parts.append(str(block.get("thinking", "")))
    thinking = "\n".join(t for t in thinking_parts if t).strip() or None
    return "".join(parts).strip(), normalize_token_usage(payload.get("usage")), thinking


def _post_messages(
    api_key: str,
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    system: object,
    messages: List[Dict[str, object]],
    timeout: Optional[float],
    max_attempts: int = 5,
    enable_thinking: bool = False,
    effort: Optional[str] = None,
) -> Reply:
    body = _build_request_body(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=messages,
        enable_thinking=enable_thinking,
        effort=effort,
    )

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

    def _do_request():
        with urllib.request.urlopen(req, timeout=effective_timeout) as resp:
            return json.loads(resp.read().decode())

    try:
        payload = call_with_retry(_do_request, max_attempts=max_attempts)
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

    text, usage, thinking = _parse_response(payload)
    stop_reason = payload.get("stop_reason")
    return Reply(
        text=text,
        usage=usage,
        thinking=thinking,
        stop_reason=stop_reason,
        token_truncated=detect_token_truncated(stop_reason, usage, max_tokens),
    )


@dataclass
class ClaudeAnthropicConfig:
    model: str = DEFAULT_CLAUDE_MODEL
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: Optional[float] = 180.0
    max_attempts: int = 5
    enable_prompt_cache: bool = True
    enable_thinking: bool = False
    effort: Optional[str] = None
    # Message Batches API polling knobs (used only by generate_batch).
    batch_poll_interval_s: float = 30.0
    batch_deadline_s: float = 7200.0
    batch_cancel_grace_s: float = 300.0


@dataclass
class ClaudeAnthropicAgent:
    """Claude via Anthropic Messages API (`ANTHROPIC_API_KEY`). Supports vision user turns."""

    config: ClaudeAnthropicConfig = field(default_factory=ClaudeAnthropicConfig)
    api_key: Optional[str] = None
    last_usage: Optional[Dict[str, int]] = field(default=None, init=False)
    last_thinking: Optional[str] = field(default=None, init=False)

    def __post_init__(self) -> None:
        key = (self.api_key or os.environ.get("ANTHROPIC_API_KEY") or "").strip()
        if not key:
            raise ValueError(
                "No Anthropic API key found. Set ANTHROPIC_API_KEY or pass api_key=... "
                "to ClaudeAnthropicAgent."
            )
        self.api_key = key

    def generate(self, messages: List[dict]) -> Reply:
        system, turns = _to_anthropic_turns(messages)
        if self.config.enable_prompt_cache:
            system, turns = _apply_prompt_cache(system, turns)
        return _post_messages(
            self.api_key,
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=system,
            messages=turns,
            timeout=self.config.timeout,
            max_attempts=self.config.max_attempts,
            enable_thinking=self.config.enable_thinking,
            effort=self.config.effort,
        )

    def _params_for(self, messages: List[dict]) -> Dict[str, object]:
        """The exact request body `generate` would send for these messages.

        Shared with `generate` via `_build_request_body` so batched bodies can
        never drift from sync bodies.
        """
        system, turns = _to_anthropic_turns(messages)
        if self.config.enable_prompt_cache:
            system, turns = _apply_prompt_cache(system, turns)
        return _build_request_body(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=system,
            messages=turns,
            enable_thinking=self.config.enable_thinking,
            effort=self.config.effort,
        )

    def _reply_from_result(self, result: Optional[dict]) -> Reply:
        """Turn one batch `result` object into a Reply.

        Absent ids (missing after a deadline cancel) map to ``batch_expired``;
        non-succeeded results (errored/canceled/expired) become empty stubs with
        ``batch_<type>``. Succeeded messages parse exactly like the sync path.
        """
        if result is None:
            return Reply(text="", stop_reason="batch_expired")
        result_type = result.get("type")
        if result_type != "succeeded":
            return Reply(text="", stop_reason=f"batch_{result_type}")
        message = result.get("message", {})
        text, usage, thinking = _parse_response(message)
        stop_reason = message.get("stop_reason")
        return Reply(
            text=text,
            usage=usage,
            thinking=thinking,
            stop_reason=stop_reason,
            token_truncated=detect_token_truncated(stop_reason, usage, self.config.max_tokens),
        )

    def generate_batch(self, batch: List[List[dict]]) -> List[Reply]:
        """Run every message list through the Message Batches API, in input order.

        Does not touch ``last_usage``/``last_thinking`` — batch calls carry their
        result in the returned ``Reply`` so N calls can be in flight at once.
        """
        requests = [
            {"custom_id": f"i{i}", "params": self._params_for(messages)}
            for i, messages in enumerate(batch)
        ]
        results = run_message_batch(
            requests,
            api_key=self.api_key,
            poll_interval_s=self.config.batch_poll_interval_s,
            deadline_s=self.config.batch_deadline_s,
            cancel_grace_s=self.config.batch_cancel_grace_s,
        )
        return [self._reply_from_result(results.get(f"i{i}")) for i in range(len(batch))]

    def __call__(self, messages: List[dict]) -> str:
        reply = self.generate(messages)
        self.last_usage = reply.usage
        self.last_thinking = reply.thinking
        return reply.text
