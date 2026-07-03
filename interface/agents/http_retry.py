"""Bounded retry-with-backoff for the urllib-based API agents.

Transient API failures (HTTP 429/5xx, timeouts, connection errors) are retried so
a single blip does not abort an in-flight episode and discard the tokens it has
already spent. Non-retryable errors (auth, bad request) propagate immediately.
"""

from __future__ import annotations

import random
import socket
import time
import urllib.error
from typing import Callable, Optional, TypeVar

T = TypeVar("T")

# 429 = rate limit; 500/502/503/504 = transient server-side failures.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


def is_retryable_error(exc: BaseException) -> bool:
    """True for transient failures worth retrying. HTTPError is checked before
    URLError because it is a subclass."""
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in _RETRYABLE_STATUS
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return True
    if isinstance(exc, urllib.error.URLError):
        return True
    return False


def retry_after_seconds(exc: BaseException) -> Optional[float]:
    """Parse a numeric ``Retry-After`` header (seconds) if present. The HTTP-date
    form is unsupported and falls back to exponential backoff."""
    headers = getattr(exc, "headers", None)
    if headers is None:
        return None
    raw = headers.get("Retry-After")
    if not raw:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _backoff_delay(
    exc: BaseException, attempt: int, base_delay: float, max_delay: float,
    rand: Callable[[], float],
) -> float:
    server = retry_after_seconds(exc)
    if server is not None:
        return min(server, max_delay)
    capped = min(max_delay, base_delay * (2 ** (attempt - 1)))
    return capped + rand() * base_delay  # full jitter on the base step


def call_with_retry(
    do_request: Callable[[], T],
    *,
    max_attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
    rand: Callable[[], float] = random.random,
) -> T:
    """Call ``do_request`` with up to ``max_attempts`` tries, sleeping with
    exponential backoff (or the server's Retry-After) between retryable failures.
    Re-raises the last exception once attempts are exhausted or it is not
    retryable."""
    for attempt in range(1, max_attempts + 1):
        try:
            return do_request()
        except Exception as exc:
            if attempt >= max_attempts or not is_retryable_error(exc):
                raise
            sleep(_backoff_delay(exc, attempt, base_delay, max_delay, rand))
    raise AssertionError("unreachable")  # pragma: no cover
