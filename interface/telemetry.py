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

    # Anthropic prompt caching reports `input_tokens` as the *uncached remainder*
    # and splits the rest into cache_read/cache_creation. Fold those back so
    # `input_tokens` stays the full prompt size (matching the no-cache semantics
    # and OpenAI's `prompt_tokens`, which already counts cached tokens).
    cache_read = usage.get("cache_read_input_tokens")
    cache_creation = usage.get("cache_creation_input_tokens")
    if input_tokens is not None and (cache_read is not None or cache_creation is not None):
        input_tokens = int(input_tokens) + int(cache_read or 0) + int(cache_creation or 0)

    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = int(input_tokens or 0) + int(output_tokens or 0)

    normalized = {}
    if input_tokens is not None:
        normalized["input_tokens"] = int(input_tokens)
    if output_tokens is not None:
        normalized["output_tokens"] = int(output_tokens)
    if total_tokens is not None:
        normalized["total_tokens"] = int(total_tokens)
    if cache_read is not None:
        normalized["cache_read_input_tokens"] = int(cache_read)
    if cache_creation is not None:
        normalized["cache_creation_input_tokens"] = int(cache_creation)
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
