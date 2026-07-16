"""Moonshot (Kimi) Batch API client (upload -> create -> poll -> download).

The lockstep runner issues one query per maze per tick; this batches them into a
single Moonshot batch job. Moonshot's batch flow is FILE-based (unlike
Anthropic's inline requests): a JSONL file is uploaded (`POST /files`,
`purpose="batch"`, one `{custom_id, method, url, body}` per line), a batch is
created referencing that file (`POST /batches`), polled until a terminal status,
and its `output_file_id` downloaded as JSONL. Results arrive in ARBITRARY order,
so the caller keys on `custom_id` — never position. Failed items land in a
separate `error_file_id` (not read here); they are simply absent from the
returned map and become stubs at the agent layer.

A poll deadline (wall-clock, `time.monotonic`) bounds the wait: on expiry we
`POST .../cancel`, then keep polling for a short grace period (`cancel_grace_s`)
until the batch reaches a terminal status, salvage whatever `output_file`
exists, and raise `MoonshotBatchDeadline` carrying those partials. The exception
(rather than a bare partial dict, which is how a *completed* batch's partials
return) lets the agent distinguish deadline-missing ids (`batch_expired`) from
error-missing ids (`batch_errored`). Every HTTP call goes through
`http_retry.call_with_retry`, matching the sync agent.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
import uuid
from typing import Dict, List, Optional

from interface.agents.http_retry import call_with_retry

logger = logging.getLogger(__name__)

# Terminal statuses. `completed` is success; `failed`/`expired`/`cancelled` are
# terminal-fail (partial output may still exist and is salvaged).
_TERMINAL = ("completed", "failed", "expired", "cancelled")


class MoonshotBatchDeadline(Exception):
    """Raised when the poll deadline expired; carries salvaged partial results.

    `.results` maps `custom_id -> chat_completion_body` for items that finished
    before the deadline cancel (possibly empty). The agent maps every id absent
    from `.results` to a ``batch_expired`` stub.
    """

    def __init__(self, results: Dict[str, dict]):
        super().__init__("Moonshot batch poll deadline expired")
        self.results = results


def _http(
    url: str,
    *,
    api_key: str,
    method: str,
    data: Optional[bytes] = None,
    content_type: Optional[str] = None,
    timeout: float = 180.0,
    max_attempts: int = 5,
) -> str:
    """One retried HTTP call to the batches API; returns the raw response body.

    `data`/`content_type` are set for POSTs (JSON or multipart); GETs send none.
    Text is returned so the JSONL output endpoint parses line-by-line while JSON
    endpoints `json.loads` the whole thing.
    """
    headers = {"Authorization": f"Bearer {api_key}"}
    if content_type is not None:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    def _do_request() -> str:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode()

    try:
        return call_with_retry(_do_request, max_attempts=max_attempts)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"Moonshot Batches HTTP {exc.code}: {detail}") from exc


def _http_json(url: str, **kwargs) -> dict:
    return json.loads(_http(url, **kwargs))


def _multipart_file(jsonl: bytes, *, purpose: str) -> tuple[bytes, str]:
    """Encode a `purpose` field + a JSONL file part as multipart/form-data.

    Built with stdlib only (uuid boundary + manual encoding) so no `requests`
    dependency creeps in. Returns (body_bytes, boundary).
    """
    boundary = uuid.uuid4().hex
    dash = f"--{boundary}"
    parts: List[bytes] = [
        f"{dash}\r\n".encode(),
        b'Content-Disposition: form-data; name="purpose"\r\n\r\n',
        f"{purpose}\r\n".encode(),
        f"{dash}\r\n".encode(),
        b'Content-Disposition: form-data; name="file"; filename="batch_input.jsonl"\r\n',
        b"Content-Type: application/jsonl\r\n\r\n",
        jsonl,
        f"\r\n{dash}--\r\n".encode(),
    ]
    return b"".join(parts), boundary


def _upload_input_file(
    lines: List[dict], *, api_key: str, base: str, timeout: float, max_attempts: int
) -> str:
    jsonl = ("\n".join(json.dumps(line) for line in lines) + "\n").encode("utf-8")
    body, boundary = _multipart_file(jsonl, purpose="batch")
    created = _http_json(
        f"{base}/files",
        api_key=api_key,
        method="POST",
        data=body,
        content_type=f"multipart/form-data; boundary={boundary}",
        timeout=timeout,
        max_attempts=max_attempts,
    )
    return created["id"]


def _parse_output_jsonl(text: str) -> Dict[str, dict]:
    """Map each JSONL line's `custom_id` to its chat-completion body.

    Each line is `{custom_id, response: {body: {...}}}`; the body falls back to a
    top-level `body` key. Lines without a `custom_id` or without any body are
    skipped so one malformed line cannot discard the whole batch.
    """
    out: Dict[str, dict] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        custom_id = record.get("custom_id")
        if custom_id is None:
            continue
        response = record.get("response") or {}
        body = response.get("body")
        if body is None:
            body = record.get("body")
        if body is None:
            continue
        out[custom_id] = body
    return out


def _is_terminal(batch: dict) -> bool:
    return batch.get("status") in _TERMINAL


def _poll_until_terminal(
    batch: dict,
    status_url: str,
    *,
    api_key: str,
    poll_interval_s: float,
    deadline: float,
    max_attempts: int,
) -> dict:
    """Poll the batch object until its `status` is terminal or the monotonic
    `deadline` passes; returns the most recent batch object either way."""
    while not _is_terminal(batch) and time.monotonic() < deadline:
        time.sleep(poll_interval_s)
        batch = _http_json(status_url, api_key=api_key, method="GET", max_attempts=max_attempts)
    return batch


def _collect_output(
    batch: dict, base: str, *, api_key: str, timeout: float, max_attempts: int
) -> Dict[str, dict]:
    output_file_id = batch.get("output_file_id")
    if not output_file_id:
        return {}
    text = _http(
        f"{base}/files/{output_file_id}/content",
        api_key=api_key,
        method="GET",
        timeout=timeout,
        max_attempts=max_attempts,
    )
    return _parse_output_jsonl(text)


def run_moonshot_batch(
    lines: List[dict],
    *,
    api_key: str,
    poll_interval_s: float = 30.0,
    deadline_s: float = 7200.0,
    cancel_grace_s: float = 300.0,
    base_url: str = "https://api.moonshot.ai/v1",
    completion_window: str = "24h",
    timeout: float = 180.0,
    max_attempts: int = 5,
) -> Dict[str, dict]:
    """Upload lines, create a batch, poll to a terminal status, return
    `{custom_id: chat_completion_body}`.

    `lines` is `[{"custom_id", "method", "url", "body"}]`. On a normal terminal
    status (`completed` or the terminal-fail states) the salvaged output map is
    returned; ids absent from it (errors live in the untouched `error_file`)
    become `batch_errored` at the agent layer. On poll-deadline expiry the batch
    is canceled, polled for up to `cancel_grace_s` more to a terminal status,
    and `MoonshotBatchDeadline` is raised carrying whatever finished — so the
    agent can distinguish those (`batch_expired`) from errors.
    """
    base = base_url.rstrip("/")
    input_file_id = _upload_input_file(
        lines, api_key=api_key, base=base, timeout=timeout, max_attempts=max_attempts
    )
    created = _http_json(
        f"{base}/batches",
        api_key=api_key,
        method="POST",
        data=json.dumps(
            {
                "input_file_id": input_file_id,
                "endpoint": "/v1/chat/completions",
                "completion_window": completion_window,
            }
        ).encode("utf-8"),
        content_type="application/json",
        timeout=timeout,
        max_attempts=max_attempts,
    )
    batch_id = created["id"]
    status_url = f"{base}/batches/{batch_id}"
    cancel_url = f"{status_url}/cancel"

    batch = _poll_until_terminal(
        created,
        status_url,
        api_key=api_key,
        poll_interval_s=poll_interval_s,
        deadline=time.monotonic() + deadline_s,
        max_attempts=max_attempts,
    )
    if not _is_terminal(batch):
        # Deadline hit: cancel, then give the batch a grace window to settle to a
        # terminal status — its finished items are already billed and salvaged.
        # The cancel POST can 4xx if the batch reached a terminal state between the
        # last poll and here; that race is benign, so log and fall through to the
        # grace poll + salvage, which handle every terminal state.
        try:
            batch = _http_json(
                cancel_url,
                api_key=api_key,
                method="POST",
                data=b"",
                content_type="application/json",
                timeout=timeout,
                max_attempts=max_attempts,
            )
        except Exception as exc:  # noqa: BLE001 - the batch may already have ended
            logger.warning(
                "Moonshot batch cancel failed (batch may have already ended): %s", exc
            )
        batch = _poll_until_terminal(
            batch,
            status_url,
            api_key=api_key,
            poll_interval_s=poll_interval_s,
            deadline=time.monotonic() + cancel_grace_s,
            max_attempts=max_attempts,
        )
        raise MoonshotBatchDeadline(
            _collect_output(
                batch, base, api_key=api_key, timeout=timeout, max_attempts=max_attempts
            )
        )

    return _collect_output(
        batch, base, api_key=api_key, timeout=timeout, max_attempts=max_attempts
    )
