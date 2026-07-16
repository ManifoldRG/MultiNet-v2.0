"""Anthropic Message Batches client (create -> poll -> fetch results).

The lockstep runner issues one query per maze per tick; this batches them into a
single `POST /v1/messages/batches`, polls `processing_status` until `"ended"`,
then streams the JSONL results and maps them back by `custom_id`. Batch results
arrive in ARBITRARY order, so the caller keys on `custom_id` — never position.

A poll deadline (wall-clock, `time.monotonic`) bounds the wait: on expiry we
`POST .../cancel`, then keep polling for a short grace period (`cancel_grace_s`)
until the batch reaches `"ended"` — a freshly-canceled batch sits in
`"canceling"` with `results_url` null, and giving up immediately would discard
every finished (already billed) item. If the batch still is not ended after the
grace period, whatever we have is returned (possibly `{}`). Every HTTP call
goes through `http_retry.call_with_retry`, matching the sync agent.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional

from interface.agents.http_retry import call_with_retry

logger = logging.getLogger(__name__)

_ANTHROPIC_VERSION = "2023-06-01"


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
        raise RuntimeError(f"Anthropic Batches HTTP {exc.code}: {detail}") from exc


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
) -> Dict[str, dict]:
    """Create a message batch, poll to completion, return `{custom_id: result}`.

    `requests` is `[{"custom_id": str, "params": <full /v1/messages body>}]`.
    On deadline expiry the batch is canceled, then polled for up to
    `cancel_grace_s` more until it reaches `"ended"` so already-finished (billed)
    item results are salvaged; missing `custom_id`s are simply absent. If it
    never ends within the grace period (`results_url` still null), returns `{}`.
    """
    base = base_url.rstrip("/")
    created = _request_json(
        f"{base}/v1/messages/batches",
        api_key=api_key,
        method="POST",
        body={"requests": requests},
    )
    batch_id = created["id"]
    status_url = f"{base}/v1/messages/batches/{batch_id}"
    cancel_url = f"{status_url}/cancel"

    batch = _poll_until_ended(
        created,
        status_url,
        api_key=api_key,
        poll_interval_s=poll_interval_s,
        deadline=time.monotonic() + deadline_s,
    )
    if batch.get("processing_status") != "ended":
        # Deadline hit: cancel, then give the batch a grace window to settle to
        # "ended" — its finished items are already billed and must be salvaged.
        # The cancel POST can 4xx if the batch reached a terminal state between the
        # last poll and here; that race is benign, so log and fall through to the
        # grace poll, which handles every terminal state and salvages results.
        try:
            batch = _request_json(cancel_url, api_key=api_key, method="POST")
        except Exception as exc:  # noqa: BLE001 - the batch may already have ended
            logger.warning(
                "Anthropic batch cancel failed (batch may have already ended): %s", exc
            )
        batch = _poll_until_ended(
            batch,
            status_url,
            api_key=api_key,
            poll_interval_s=poll_interval_s,
            deadline=time.monotonic() + cancel_grace_s,
        )

    results_url = batch.get("results_url")
    if not results_url:
        return {}
    return _parse_results_jsonl(_request(results_url, api_key=api_key, method="GET"))
