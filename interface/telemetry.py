"""Shared telemetry normalization for interface producers and scorer consumers."""

from __future__ import annotations

from typing import Any


TOKEN_COUNT_KEYS = ("total_tokens", "token_count", "tokens", "model_tokens")


def normalize_token_usage(usage: Any) -> dict[str, int] | None:
    """Normalize provider token usage into input, output, and total counts."""
    if not isinstance(usage, dict):
        return None
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    total_tokens = usage.get("total_tokens")
    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = int(input_tokens or 0) + int(output_tokens or 0)

    normalized = {}
    if input_tokens is not None:
        normalized["input_tokens"] = int(input_tokens)
    if output_tokens is not None:
        normalized["output_tokens"] = int(output_tokens)
    if total_tokens is not None:
        normalized["total_tokens"] = int(total_tokens)
    return normalized or None


def token_count_from_record(record: dict[str, Any]) -> int | None:
    """Extract one token total without counting nested aliases twice."""
    for container in (record, record.get("info"), record.get("metadata")):
        if not isinstance(container, dict):
            continue
        for key in TOKEN_COUNT_KEYS:
            if container.get(key) is not None:
                return int(container[key])
        usage = normalize_token_usage(container.get("usage"))
        if usage is not None and usage.get("total_tokens") is not None:
            return usage["total_tokens"]
    return None
