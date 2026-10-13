"""next-run-fixes item 17 (cheap part): a serial episode killed mid-run leaves a
resumable partial and its spend record.

run_episode checkpoints at every query boundary (stateless chat) and appends
one usage line per query to queries.jsonl. A process killed mid-episode (OOM,
watchdog, Ctrl-C) leaves both in the run dir; the next run of the same unit
resumes from the checkpoint through the replay gate instead of starting over.
Both files go when the episode finishes (episode.json holds the full record).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from interface.config import ExperimentConfig
from interface.loader import default_maze_path
from interface.smoke_tests.plans import v01_empty_room_trajectory
from pipeline.run_stage3 import run_episode

PLAN = v01_empty_room_trajectory()
QUERIES = len(PLAN) - 1  # stepping onto the goal ends the episode before DONE
TASK = default_maze_path("V01_empty_room.json")


class _Agent:
    def __init__(self, actions, die_at=None):
        self._actions = iter(actions)
        self._die_at = die_at
        self.calls = 0
        self.last_usage = None

    def __call__(self, messages):
        self.calls += 1
        if self._die_at is not None and self.calls >= self._die_at:
            raise KeyboardInterrupt  # the process is killed mid-query
        self.last_usage = {"input_tokens": 100 + self.calls, "output_tokens": 2, "total_tokens": 102 + self.calls}
        return f"FINAL_OUTPUT: {next(self._actions, 'DONE')}"


def test_a_killed_process_leaves_a_checkpoint_and_a_usage_line_per_query(tmp_path):
    out = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        run_episode(TASK, ExperimentConfig(), _Agent(PLAN, die_at=4), 0, out)
    assert (out / "checkpoint.json").exists() and not (out / "episode.json").exists()
    ledger = [json.loads(line) for line in (out / "queries.jsonl").read_text().splitlines()]
    assert [row["query_index"] for row in ledger] == [1, 2, 3]
    assert [row["usage"]["input_tokens"] for row in ledger] == [101, 102, 103]


def test_the_next_run_resumes_from_the_checkpoint(tmp_path):
    out = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        run_episode(TASK, ExperimentConfig(), _Agent(PLAN, die_at=4), 0, out)
    rest = _Agent(PLAN[3:])
    episode = run_episode(TASK, ExperimentConfig(), rest, 0, out)
    assert episode["end_reason"] == "success" and episode["query_count"] == QUERIES
    assert rest.calls == QUERIES - 3  # only the unanswered rest is paid for
    assert episode["resumed_from"] == {"checkpoint": True, "killed_at_query": 3}
    assert not (out / "checkpoint.json").exists() and not (out / "queries.jsonl").exists()


def test_multi_turn_chat_is_not_checkpointed(tmp_path):
    out = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        run_episode(TASK, ExperimentConfig(chat_history="rolling", context_window="current"),
                    _Agent(PLAN, die_at=3), 0, out)
    assert not (out / "checkpoint.json").exists()
    assert len((out / "queries.jsonl").read_text().splitlines()) == 2  # the spend record still survives
