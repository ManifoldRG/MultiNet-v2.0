"""Immutable model-call result carried between agent and runner.

`Reply` replaces the agents' mutable ``last_usage`` side-channel so N model
calls can be in flight concurrently (batch-API lockstep runner) without one
call clobbering another's usage/stop metadata.

``token_truncated`` (not ``truncated``) is deliberate: ``truncated`` already
means *environment* truncation elsewhere in this codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Reply:
    text: str
    usage: Optional[Dict[str, int]] = None
    thinking: Optional[str] = None
    # Provider value verbatim, or "batch_errored"/"batch_expired"/"batch_canceled".
    stop_reason: Optional[str] = None
    token_truncated: bool = False


def detect_token_truncated(
    stop_reason: Optional[str],
    usage: Optional[dict],
    max_tokens: Optional[int],
) -> bool:
    """True iff the model output was cut off by the token cap.

    An explicit stop reason wins: ``max_tokens`` (Anthropic) or ``length``
    (OpenAI-style) means truncation; any other explicit reason means it was
    not. When ``stop_reason`` is None, fall back to comparing reported output
    tokens against the cap (only when both are present).
    """
    if stop_reason is not None:
        return stop_reason in ("max_tokens", "length")
    if usage is None or max_tokens is None:
        return False
    return usage.get("output_tokens", 0) >= max_tokens
