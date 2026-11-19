"""Snapshots of every request each scratchpad arm sends in the scripted
episode (NL and 2D): the runner messages as the model reads them and a digest
of each provider's request body. A prompt change shows up here as a diff to
review; regenerate with UPDATE_MEMORY_ARM_SNAPSHOTS=1 once it is intended."""

from __future__ import annotations

import os

import pytest

from gridworld.backends.minigrid_backend import MiniGridBackend
from interface.config import ExperimentConfig
from interface.runner import build_runner
from scratchpad_test_utils import (
    ARMS,
    IMAGE_CELL,
    NL_CELL,
    SCRATCHPAD_ARMS,
    drive,
    read_snapshot,
    snapshot,
    write_snapshot,
)
from start_map_test_utils import r1_spec

CELLS = {"nl": NL_CELL, "2d": IMAGE_CELL}


@pytest.mark.parametrize("domain", sorted(CELLS))
@pytest.mark.parametrize("arm", SCRATCHPAD_ARMS)
def test_arm_requests_match_their_snapshot(arm, domain):
    spec = r1_spec(12)
    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(spec)
    runner = build_runner(ExperimentConfig(**{**CELLS[domain], **ARMS[arm]}), backend, spec)
    _, requests = drive(runner)
    actual = snapshot(requests)
    name = f"{arm}_{domain}"
    if os.environ.get("UPDATE_MEMORY_ARM_SNAPSHOTS"):
        write_snapshot(name, actual)
    expected = read_snapshot(name)
    assert len(actual) == len(expected)
    for index, (got, want) in enumerate(zip(actual, expected), start=1):
        assert got["messages"] == want["messages"], f"query {index}: runner messages changed"
        assert got["provider_body_sha256"] == want["provider_body_sha256"], (
            f"query {index}: a provider request body changed"
        )
