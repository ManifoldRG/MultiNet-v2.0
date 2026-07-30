"""Unit tests for the pure (worktree-independent) helpers in
scripts/resume_from_archive.py. The replay itself runs against an as-run
worktree and is exercised by the script's own replay-verify gate."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from scripts.resume_from_archive import (
    IN_PROGRESS_END_REASON,
    build_checkpoint_payload,
    rehydrate_agent_messages,
    scripted_action_from_spec,
    trailing_parse_failures,
)


def _query(idx: int, parse_ok: bool) -> dict:
    return {"kind": "query", "query_index": idx, "parse_ok": parse_ok, "agent_messages": []}


def _step(idx: int, failures_after: int = 0) -> dict:
    return {"kind": "step", "step_index": idx, "consecutive_failures_after": failures_after}


def test_trailing_parse_failures_counts_only_the_trailing_run():
    records = [_query(1, False), _query(2, True), _query(3, False), _query(4, False)]
    assert trailing_parse_failures(records) == 2
    # A successful final parse resets the live counter to zero, even with
    # earlier mid-episode outage blips (queries 13/15/68 in the R1 archive).
    assert trailing_parse_failures([_query(1, False), _query(2, True)]) == 0
    assert trailing_parse_failures([]) == 0


def test_build_checkpoint_payload_clears_the_infra_kill():
    episode = {
        "end_reason": "parse_failed",
        "task_spec": {"seed": 7},
        "transcript": [
            {"kind": "reset"},
            _query(1, True),
            _step(1, failures_after=1),
            _query(2, False),
        ],
    }
    payload = build_checkpoint_payload(episode)
    assert payload["seed"] == 7
    assert payload["query_count"] == 2
    assert payload["parse_failures"] == 1
    assert payload["consecutive_failures"] == 1
    assert payload["finished"] is False
    assert payload["end_reason"] == IN_PROGRESS_END_REASON


def test_build_checkpoint_payload_refuses_finished_episodes():
    episode = {"end_reason": "success", "task_spec": {"seed": 0}, "transcript": []}
    with pytest.raises(SystemExit):
        build_checkpoint_payload(episode)


def test_rehydrate_restores_present_images_and_placeholders_missing(tmp_path: Path):
    qdir = tmp_path / "queries" / "query_001"
    qdir.mkdir(parents=True)
    (qdir / "have.png").write_bytes(b"png-bytes")
    transcript = [
        {
            "kind": "query",
            "query_index": 1,
            "agent_messages": [
                {"role": "system", "content": "plain string is untouched"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {"type": "image", "file": "have.png"},
                        {"type": "image", "file": "lost.png"},
                    ],
                },
            ],
        }
    ]
    missing = rehydrate_agent_messages(transcript, tmp_path)
    assert missing == 1
    blocks = transcript[0]["agent_messages"][1]["content"]
    assert blocks[0] == {"type": "text", "text": "hello"}
    expected_b64 = base64.b64encode(b"png-bytes").decode("ascii")
    assert blocks[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + expected_b64},
    }
    assert blocks[2]["type"] == "text" and "lost.png" in blocks[2]["text"]


def test_scripted_action_spec_parsing():
    assert scripted_action_from_spec("scripted:MOVE_FORWARD") == "MOVE_FORWARD"
    with pytest.raises(SystemExit):
        scripted_action_from_spec("MOVE_FORWARD")
    with pytest.raises(SystemExit):
        scripted_action_from_spec("scripted:")
