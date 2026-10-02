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
    drive_continue_loop,
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


# --------------------------------------------------------------------------
# drive_continue_loop: duck-typed against the old code's stepper/agent, so it
# is testable from the main checkout with fakes.
# --------------------------------------------------------------------------

class _FakeState:
    step_count = 0


class _FakeStepper:
    """Minimal stand-in: yields `rounds` query rounds, then None."""

    def __init__(self, rounds: int) -> None:
        self.rounds = rounds
        self.query_count = 0
        self.applied: list[object] = []
        self.action_queue: list[str] = []
        self.state = _FakeState()

    def next_query(self):
        if self.query_count >= self.rounds:
            return None
        self.query_count += 1  # the real stepper reserves the index here
        return [{"role": "user", "content": "go"}]

    def apply_reply(self, reply):
        self.applied.append(reply)


class _FakeAgent:
    def __init__(self, replies=(), raises_on: int | None = None) -> None:
        self.calls = 0
        self.raises_on = raises_on

    def generate(self, messages):
        self.calls += 1
        if self.raises_on is not None and self.calls == self.raises_on:
            raise RuntimeError("Moonshot API timed out after 600s")
        return f"reply-{self.calls}"


def test_continue_loop_runs_to_exhaustion_of_the_stepper():
    stepper, agent = _FakeStepper(rounds=3), _FakeAgent()
    new_queries, stopped_early = drive_continue_loop(
        stepper, agent, max_new_queries=10, log=lambda _m: None
    )
    assert (new_queries, stopped_early) == (3, None)
    assert stepper.applied == ["reply-1", "reply-2", "reply-3"]
    assert stepper.query_count == 3


def test_continue_loop_stops_at_the_new_query_cap_and_rolls_the_index_back():
    stepper, agent = _FakeStepper(rounds=10), _FakeAgent()
    new_queries, stopped_early = drive_continue_loop(
        stepper, agent, max_new_queries=2, log=lambda _m: None
    )
    assert new_queries == 2
    assert stopped_early == "max_new_queries(2)"
    # query_count equals the APPLIED query count: the reserved-but-unused
    # index is rolled back.
    assert stepper.query_count == 2


def test_continue_loop_flushes_instead_of_crashing_on_an_agent_error():
    """A transport failure must never discard hours of paid work: the loop
    reports it as an interruption so main() still writes episode artifacts."""
    stepper, agent = _FakeStepper(rounds=10), _FakeAgent(raises_on=3)
    new_queries, stopped_early = drive_continue_loop(
        stepper, agent, max_new_queries=10, log=lambda _m: None
    )
    assert new_queries == 2  # the two that applied cleanly
    assert stopped_early.startswith("agent_error:RuntimeError: Moonshot API timed out")
    assert stepper.query_count == 2
    assert stepper.applied == ["reply-1", "reply-2"]


def test_continue_loop_stops_when_a_scripted_agent_reports_exhausted():
    class _Exhausted(_FakeAgent):
        exhausted = True

    stepper = _FakeStepper(rounds=5)
    new_queries, stopped_early = drive_continue_loop(
        stepper, _Exhausted(), max_new_queries=10, log=lambda _m: None
    )
    assert (new_queries, stopped_early) == (0, "scripted_agent_exhausted")
    assert stepper.query_count == 0
