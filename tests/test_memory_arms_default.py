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

# Billing keys of a model config: what a run may spend and what a token costs,
# never what the model sees or emits. Their place in the run hash and their
# values change outside the memory arms (one PR drops both from the hash,
# another re-sizes the mechanics-15 spend caps), so the pins leave them out.
# The memory fields are experiment_config fields, which the pins cover in full.
_BILLING_KEYS = ("spend_cap_usd", "service_tier")

# _expected_run_hash(r1_spec(12), "gpt6_astra", 0, "minigrid", ...) for each
# mechanics-15 run config's experiment_config and gpt6_astra model config minus
# _BILLING_KEYS, computed before the memory arms existed (main at 4b8b54d). With
# the billing keys in, main gave 52bb8852... (2d, 3d) and 91acbde4... (nl).
_PINNED_R2_RUN_HASHES = {
    "nl": "32f59c64483048796b7a0112acbc847bc0120c5d3e4b0e9728950d5a66eeca4a",
    "2d": "cff9adca355e855ae805c2c701da40a16a5a480a965297bd3da13d5303d91ab3",
    "3d": "cff9adca355e855ae805c2c701da40a16a5a480a965297bd3da13d5303d91ab3",
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
    model_config = {
        k: v for k, v in cfg["models"]["gpt6_astra"].items() if k not in _BILLING_KEYS
    }
    got = _expected_run_hash(
        r1_spec(12), "gpt6_astra", 0, "minigrid",
        experiment_config=exp, model_config=model_config,
    )
    assert got == _PINNED_R2_RUN_HASHES[domain]
