"""The default memory arm (full_context) is today's R2 harness, byte for byte.

The snapshots in tests/fixtures/memory_arms/default_*.json were written by the
code before the memory arms existed (main at 4b8b54d): every request of a
scripted episode on the R2 NL and 2D cells, and a digest of the body each
provider agent builds from it. The pinned run hashes were computed by that code
too, so cached R2 episodes stay cached.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gridworld.backends.minigrid_backend import MiniGridBackend
from interface.config import ExperimentConfig
from interface.runner import build_runner
from scripts.run_pipeline import _expected_run_hash
from scratchpad_test_utils import IMAGE_CELL, NL_CELL, drive, read_snapshot, snapshot
from start_map_test_utils import r1_spec

REPO_ROOT = Path(__file__).resolve().parent.parent

# _expected_run_hash(r1_spec(12), "gpt6_astra", 0, "minigrid", ...) for each
# mechanics-15 run config's experiment_config and gpt6_astra model config,
# computed before the memory arms existed.
_PINNED_R2_RUN_HASHES = {
    "nl": "91acbde4aa7d42e7edc08dffbd6cd92169f79bf3056d19048d29f588ceddb7fb",
    "2d": "52bb8852af4ddb068142da38fd783adcd97261adca73d5ec4c29e8d953ac1013",
    "3d": "52bb8852af4ddb068142da38fd783adcd97261adca73d5ec4c29e8d953ac1013",
}


def _runner(cell: dict):
    spec = r1_spec(12)
    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(spec)
    return build_runner(ExperimentConfig(**cell), backend, spec)


@pytest.mark.parametrize("name, cell", [("default_nl", NL_CELL), ("default_2d", IMAGE_CELL)])
def test_default_requests_match_the_pre_memory_snapshot(name, cell):
    _, requests = drive(_runner(cell))
    expected = read_snapshot(name)
    actual = snapshot(requests)
    assert len(actual) == len(expected)
    for index, (got, want) in enumerate(zip(actual, expected), start=1):
        assert got["messages"] == want["messages"], f"query {index}: runner messages changed"
        assert got["provider_body_sha256"] == want["provider_body_sha256"], (
            f"query {index}: a provider request body changed"
        )


@pytest.mark.parametrize("domain", sorted(_PINNED_R2_RUN_HASHES))
def test_r2_run_hashes_are_unchanged(domain):
    cfg = json.loads(
        (REPO_ROOT / f"gridworld/fixtures/run_config.mechanics_15_{domain}.json").read_text()
    )
    exp = ExperimentConfig.from_dict(cfg["experiment_config"])
    got = _expected_run_hash(
        r1_spec(12), "gpt6_astra", 0, "minigrid",
        experiment_config=exp, model_config=cfg["models"]["gpt6_astra"],
    )
    assert got == _PINNED_R2_RUN_HASHES[domain]
