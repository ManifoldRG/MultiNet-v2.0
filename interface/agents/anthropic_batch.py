"""Anthropic Message Batches client (create -> poll -> fetch results).

The lockstep runner issues one query per maze per tick; this batches them into a
single `POST /v1/messages/batches`, polls `processing_status` until `"ended"`,
then streams the JSONL results and maps them back by `custom_id`. Batch results
arrive in ARBITRARY order, so the caller keys on `custom_id` — never position.

A poll deadline (wall-clock, `time.monotonic`) bounds the wait: on expiry we
`POST .../cancel`, then keep polling until the batch reaches `"ended"` — a
freshly-canceled batch sits in `"canceling"` with `results_url` null until the
items already in progress finish, and giving up early would discard every
finished (already billed) item. A cancel always ends the batch, so past the
grace period (`cancel_grace_s`) we keep waiting, up to `cancel_wait_s` from the
cancel. Only a batch that has still not ended then returns `{}`, with its id
logged so its results can be fetched by hand. Every HTTP call goes through
`http_retry.call_with_retry`, matching the sync agent.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional

from interface.agents.billing import BillingExhausted
from interface.agents.http_retry import call_with_retry

logger = logging.getLogger(__name__)

_ANTHROPIC_VERSION = "2023-06-01"

# Creating a batch while the account's prepaid credit is momentarily exhausted
# (e.g. a parallel run drained it before auto-refill fired) returns HTTP 400
# "Your credit balance is too low ...". Raised, that would error every
# in-flight unit of the lockstep round, so batch creation waits for the refill
# instead, bounded by `credit_wait_s`.
_CREDIT_WAIT_S = 3600.0
_CREDIT_POLL_S = 60.0

# How long after a deadline cancel we wait for the batch to end before dropping
# its results. A canceled batch ends once its in-progress items finish, which
# can outlast the cancel grace (one batch in the mechanics-15 3D run ended 6
# minutes after it, with 6 billed results). A round can take the low-credit
# wait + batch_deadline_s + this: 3600 + 7200 + 1800 = 12600, the coordinator's
# documented --stale-after-seconds.
_CANCEL_WAIT_S = 1800.0


class AnthropicBatchHTTPError(RuntimeError):
    """A non-retryable HTTP error from the batches API, with its status/body."""

    def __init__(self, code: int, detail: str):
        super().__init__(f"Anthropic Batches HTTP {code}: {detail}")
        self.code = code
        self.detail = detail


def _is_low_credit(exc: BaseException) -> bool:
    return (
        isinstance(exc, AnthropicBatchHTTPError)
        and exc.code == 400
        and "credit balance" in exc.detail.lower()
    )


def _request(
    url: str,
    *,
    api_key: str,
    method: str,
    body: Optional[dict] = None,
    timeout: float = 180.0,
    max_attempts: int = 5,
) -> str:
    """One retried HTTP call to the batches API; returns the raw response body.

    `body` is JSON-encoded when present (POSTs); GETs send none. Returns text so
    the JSONL results endpoint can be parsed line-by-line while JSON endpoints
    `json.loads` the whole thing.
    """
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {
        "x-api-key": api_key,
        "anthropic-version": _ANTHROPIC_VERSION,
    }
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    def _do_request() -> str:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode()

    try:
        return call_with_retry(_do_request, max_attempts=max_attempts)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise AnthropicBatchHTTPError(exc.code, detail) from exc


def _request_json(url: str, **kwargs) -> dict:
    return json.loads(_request(url, **kwargs))


def _parse_results_jsonl(text: str) -> Dict[str, dict]:
    """Map each JSONL line's `custom_id` to its `result` object."""
    out: Dict[str, dict] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        custom_id = record.get("custom_id")
        if custom_id is None:
            continue  # one malformed line must not discard the whole batch
        out[custom_id] = record.get("result", {})
    return out


def _create_batch(
    url: str,
    requests: List[dict],
    *,
    api_key: str,
    credit_wait_s: float,
    credit_poll_s: float,
) -> dict:
    """POST the batch, waiting out a low-credit 400 for up to `credit_wait_s`."""
    deadline = time.monotonic() + credit_wait_s
    while True:
        try:
            return _request_json(
                url, api_key=api_key, method="POST", body={"requests": requests}
            )
        except AnthropicBatchHTTPError as exc:
            if not _is_low_credit(exc):
                raise
            if time.monotonic() >= deadline:
                # Still out of credit after the wait: billing, so the run parks.
                raise BillingExhausted("claude", f"Anthropic Batches HTTP {exc.code}: {exc.detail}") from exc
            logger.warning(
                "Anthropic batch create refused for low credit balance; "
                "retrying in %.0fs (waiting for auto-refill): %s",
                credit_poll_s,
                exc.detail,
            )
            time.sleep(credit_poll_s)


def _poll_until_ended(
    batch: dict,
    status_url: str,
    *,
    api_key: str,
    poll_interval_s: float,
    deadline: float,
) -> dict:
    """Poll the batch object until `processing_status == "ended"` or the
    monotonic `deadline` passes; returns the most recent batch object either
    way."""
    while batch.get("processing_status") != "ended" and time.monotonic() < deadline:
        time.sleep(poll_interval_s)
        batch = _request_json(status_url, api_key=api_key, method="GET")
    return batch


def run_message_batch(
    requests: List[dict],
    *,
    api_key: str,
    poll_interval_s: float = 30.0,
    deadline_s: float = 7200.0,
    cancel_grace_s: float = 300.0,
    base_url: str = "https://api.anthropic.com",
    credit_wait_s: float = _CREDIT_WAIT_S,
    credit_poll_s: float = _CREDIT_POLL_S,
    cancel_wait_s: Optional[float] = None,
) -> Dict[str, dict]:
    """Create a message batch, poll to completion, return `{custom_id: result}`.

    `requests` is `[{"custom_id": str, "params": <full /v1/messages body>}]`.
    On deadline expiry the batch is canceled, then polled until it reaches
    `"ended"` so already-finished (billed) item results are salvaged; missing
    `custom_id`s are simply absent. Past `cancel_grace_s` it logs and keeps
    waiting, up to `cancel_wait_s` (default `_CANCEL_WAIT_S`) from the cancel; a
    batch that never ends by then (`results_url` still null) returns `{}`.
    A low-credit refusal at creation is retried every `credit_poll_s` for up to
    `credit_wait_s` (see `_create_batch`).
    """
    base = base_url.rstrip("/")
    created = _create_batch(
        f"{base}/v1/messages/batches",
        requests,
        api_key=api_key,
        credit_wait_s=credit_wait_s,
        credit_poll_s=credit_poll_s,
    )
    batch_id = created["id"]
    status_url = f"{base}/v1/messages/batches/{batch_id}"
    cancel_url = f"{status_url}/cancel"
    logger.info("Anthropic batch %s created with %d requests", batch_id, len(requests))

    batch = _poll_until_ended(
        created,
        status_url,
        api_key=api_key,
        poll_interval_s=poll_interval_s,
        deadline=time.monotonic() + deadline_s,
    )
    if batch.get("processing_status") != "ended":
        # Deadline hit: cancel, then wait for the batch to settle to "ended" —
        # its finished items are already billed and must be salvaged.
        # The cancel POST can 4xx if the batch reached a terminal state between the
        # last poll and here; that race is benign, so log and fall through to the
        # grace poll, which handles every terminal state and salvages results.
        logger.warning("Anthropic batch %s passed its %.0fs deadline; canceling it", batch_id, deadline_s)
        try:
            batch = _request_json(cancel_url, api_key=api_key, method="POST")
        except Exception as exc:  # noqa: BLE001 - the batch may already have ended
            logger.warning(
                "Anthropic batch %s cancel failed (batch may have already ended): %s", batch_id, exc
            )
        canceled_at = time.monotonic()
        batch = _poll_until_ended(
            batch,
            status_url,
            api_key=api_key,
            poll_interval_s=poll_interval_s,
            deadline=canceled_at + cancel_grace_s,
        )
        wait_s = _CANCEL_WAIT_S if cancel_wait_s is None else cancel_wait_s
        if batch.get("processing_status") != "ended" and wait_s > cancel_grace_s:
            # A cancel always ends the batch once the items in progress finish;
            # dropping it now would discard the finished ones, which are billed.
            logger.warning(
                "Anthropic batch %s still %s %.0fs after its cancel; waiting up to %.0fs for it to end",
                batch_id, batch.get("processing_status"), cancel_grace_s, wait_s,
            )
            batch = _poll_until_ended(
                batch,
                status_url,
                api_key=api_key,
                poll_interval_s=poll_interval_s,
                deadline=canceled_at + wait_s,
            )

    results_url = batch.get("results_url")
    if not results_url:
        logger.error(
            "Anthropic batch %s has not ended (%s); its results are not used, and any "
            "finished items are billed: fetch them by batch id",
            batch_id, batch.get("processing_status"),
        )
        return {}
    return _parse_results_jsonl(_request(results_url, api_key=api_key, method="GET"))
