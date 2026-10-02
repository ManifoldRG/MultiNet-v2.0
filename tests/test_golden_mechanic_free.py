"""Mechanic-free mazes must play byte-identically to the recorded golden.

See ``tests/golden_mechanic_free.py`` for what is recorded and how to
regenerate it. A failure here means a change altered what the model is told
(or the failure-counting control flow) on mazes without teleporters, death
tiles or ice, which breaks comparability with earlier runs.
"""

from __future__ import annotations

import json

import pytest

from tests.golden_mechanic_free import CONFIGS, GOLDEN_PATH, MAZES, play

_GOLDEN = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def test_golden_covers_every_maze_and_config():
    assert sorted(_GOLDEN) == sorted(f"{m}/{c}" for m in MAZES for c in CONFIGS)


@pytest.mark.parametrize("maze", MAZES)
@pytest.mark.parametrize("config_name", sorted(CONFIGS))
def test_mechanic_free_episode_matches_golden(maze, config_name):
    golden = _GOLDEN[f"{maze}/{config_name}"]
    actual = play(maze, config_name)

    assert actual["steps"] == golden["steps"]
    assert actual["system"] == golden["system"]
    assert len(actual["user_turns"]) == len(golden["user_turns"])
    for i, (got, want) in enumerate(zip(actual["user_turns"], golden["user_turns"])):
        assert got == want, f"user turn {i} differs"
    assert actual == golden
