"""ExperimentConfig.memory: the arm field, its cap, its checks and its place in
the run hash."""

from __future__ import annotations

import pytest

from interface.config import ExperimentConfig
from scripts.run_pipeline import _experiment_config_hash_payload, _expected_run_hash
from scratchpad_test_utils import NL_CELL
from start_map_test_utils import r1_spec

WINDOW = {**NL_CELL, "context_window": "last_n", "context_n": 5, "memory": "scratchpad_window"}
FULL_SCRATCHPAD = {**NL_CELL, "memory": "full_context_scratchpad"}


def _hash(cell: dict) -> str:
    return _expected_run_hash(
        r1_spec(12), "m", 0, "minigrid", experiment_config=ExperimentConfig(**cell)
    )


def test_defaults_are_the_harness_without_a_scratchpad():
    config = ExperimentConfig()
    assert config.memory == "full_context"
    assert config.scratchpad_max_chars == 2000
    assert config.scratchpad is False
    assert ExperimentConfig(**WINDOW).scratchpad is True
    assert ExperimentConfig(**FULL_SCRATCHPAD).scratchpad is True


def test_round_trips_through_dict():
    config = ExperimentConfig(**{**WINDOW, "scratchpad_max_chars": 1500})
    assert ExperimentConfig.from_dict(config.to_dict()) == config
    assert ExperimentConfig.from_dict(ExperimentConfig().to_dict()) == ExperimentConfig()


def test_default_arm_serializes_as_before_the_arms_existed():
    """Omitted when off, as tile_labels does: what a default run persists
    (episode.json config, run_inputs.json, a job plan's units) stays the bytes
    it was before the memory arms."""
    for cell in (ExperimentConfig(), ExperimentConfig(**NL_CELL)):
        assert {"memory", "scratchpad_max_chars"}.isdisjoint(cell.to_dict())
    for cell in (WINDOW, FULL_SCRATCHPAD):
        d = ExperimentConfig(**cell).to_dict()
        assert (d["memory"], d["scratchpad_max_chars"]) == (cell["memory"], 2000)


def test_a_cap_without_a_scratchpad_is_an_error():
    # It would change nothing and be dropped from to_dict, so a config that
    # sets it most likely forgot the memory arm.
    with pytest.raises(ValueError, match="scratchpad_max_chars only applies"):
        ExperimentConfig(**{**NL_CELL, "scratchpad_max_chars": 99})
    ExperimentConfig(**{**NL_CELL, "scratchpad_max_chars": 2000})


def test_unknown_arm_is_rejected():
    with pytest.raises(ValueError, match="memory must be one of"):
        ExperimentConfig(**{**NL_CELL, "memory": "scratchpad"})


@pytest.mark.parametrize("cap", [0, -1, True, 1.5, "2000", None])
def test_cap_must_be_a_positive_int(cap):
    with pytest.raises(ValueError, match="scratchpad_max_chars"):
        ExperimentConfig(**{**FULL_SCRATCHPAD, "scratchpad_max_chars": cap})


@pytest.mark.parametrize(
    "cell, window",
    [
        ({**WINDOW, "context_window": "full"}, "last_n"),
        ({**WINDOW, "context_window": "text_summary_and_last_n"}, "last_n"),
        ({**FULL_SCRATCHPAD, "context_window": "last_n"}, "full"),
        ({**FULL_SCRATCHPAD, "context_window": "current"}, "full"),
    ],
)
def test_each_scratchpad_arm_fixes_its_history_window(cell, window):
    with pytest.raises(ValueError, match=f"needs context_window='{window}'"):
        ExperimentConfig(**cell)


@pytest.mark.parametrize("cell", [WINDOW, FULL_SCRATCHPAD])
@pytest.mark.parametrize("chat", ["rolling", "full"])
def test_scratchpad_arms_need_stateless_chat(cell, chat):
    with pytest.raises(ValueError, match="needs chat_history='stateless'"):
        ExperimentConfig(**{**cell, "chat_history": chat})


def test_full_context_accepts_any_window():
    for window in ("current", "last_n", "text_summary", "text_summary_and_last_n", "full"):
        ExperimentConfig(**{**NL_CELL, "context_window": window})


def test_default_arm_leaves_the_memory_fields_out_of_the_hash():
    payload = _experiment_config_hash_payload(ExperimentConfig(**NL_CELL))
    assert "memory" not in payload and "scratchpad_max_chars" not in payload
    # A plain dict (a job plan's unit) hashes like the dataclass, with or
    # without the default arm spelled out.
    raw = ExperimentConfig(**NL_CELL).to_dict()
    assert _experiment_config_hash_payload(raw) == payload
    spelled = {**raw, "memory": "full_context", "scratchpad_max_chars": 2000}
    assert _experiment_config_hash_payload(spelled) == payload


def test_scratchpad_arms_hash_their_arm_cap_and_window():
    payload = _experiment_config_hash_payload(ExperimentConfig(**WINDOW))
    assert payload["memory"] == "scratchpad_window"
    assert payload["scratchpad_max_chars"] == 2000
    hashes = {
        _hash(NL_CELL),
        _hash(WINDOW),
        _hash(FULL_SCRATCHPAD),
        _hash({**WINDOW, "context_n": 3}),
        _hash({**WINDOW, "scratchpad_max_chars": 1000}),
        _hash({**FULL_SCRATCHPAD, "scratchpad_max_chars": 1000}),
    }
    assert len(hashes) == 6
    # prompt_cache stays out of the hash in every arm.
    assert _hash({**FULL_SCRATCHPAD, "prompt_cache": False}) == _hash(FULL_SCRATCHPAD)
