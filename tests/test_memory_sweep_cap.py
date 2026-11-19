"""A scratchpad cap set once for a whole sweep.

A sweep is a condition set over a run config's shared ``experiment_config``
block. A ``scratchpad_max_chars`` in that block applies to every scratchpad arm
the variants produce and is inert on the default (full_context) arm: that cell
persists and hashes exactly as it would with no cap anywhere. A cap set on the
default arm itself (its variant, the cell, or a run with no scratchpad arm at
all) is still an error. The rule lives in ``_condition_configs``, which the
local pipeline and the fleet's ``prepare_job`` both use.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from interface.config import ExperimentConfig
from prompting_experiments import CONDITION_SETS
from prompting_experiments.core import ConditionSet, Variant
from scratchpad_test_utils import NL_CELL
from scripts.run_pipeline import _condition_configs, _expected_run_hash
from start_map_test_utils import r1_spec

SWEEP = "Memory sweep (test)"
ARM_OVERRIDES = {
    "full_context": {},
    "scratchpad_window": {
        "memory": "scratchpad_window", "context_window": "last_n", "context_n": 5,
    },
    "full_context_scratchpad": {"memory": "full_context_scratchpad"},
}
SHARED = {**NL_CELL, "scratchpad_max_chars": 1000}


def _condition_set(overrides: dict[str, dict]) -> ConditionSet:
    return ConditionSet(
        name=SWEEP,
        comparisons=(),
        decision="test only",
        variants={name: Variant(name, name, cfg) for name, cfg in overrides.items()},
    )


@pytest.fixture
def sweep(monkeypatch):
    """Register the memory sweep (no such condition set ships yet)."""
    monkeypatch.setitem(CONDITION_SETS, SWEEP, _condition_set(ARM_OVERRIDES))
    return SWEEP


def _hash(cell: ExperimentConfig, variant: str) -> str:
    return _expected_run_hash(
        r1_spec(12), "m", 0, "minigrid",
        condition_set=SWEEP, prompt_variant=variant, experiment_config=cell,
    )


# --------------------------------------------------------------------------- #
# (a) the sweep's cap reaches every scratchpad arm; the default arm is untouched
# --------------------------------------------------------------------------- #
def test_a_sweep_cap_reaches_every_scratchpad_arm_and_skips_the_default_arm(sweep):
    capped = dict(_condition_configs(sweep, base_overrides=SHARED))
    plain = dict(_condition_configs(sweep, base_overrides=NL_CELL))
    assert set(capped) == set(ARM_OVERRIDES)
    assert capped["scratchpad_window"].scratchpad_max_chars == 1000
    assert capped["full_context_scratchpad"].scratchpad_max_chars == 1000
    assert capped["scratchpad_window"].context_n == 5
    # The default arm is the cell it would be with no cap anywhere.
    default = capped["full_context"]
    assert default == plain["full_context"] == ExperimentConfig(**NL_CELL)
    assert default.to_dict() == plain["full_context"].to_dict()
    assert {"memory", "scratchpad_max_chars"}.isdisjoint(default.to_dict())
    assert _hash(default, "full_context") == _hash(plain["full_context"], "full_context")
    # The capped arms hash apart from the uncapped ones.
    for arm in ("scratchpad_window", "full_context_scratchpad"):
        assert _hash(capped[arm], arm) != _hash(plain[arm], arm)


def test_one_batch_of_the_sweep_can_run_the_default_arm_alone(sweep):
    """A batch runs one prompt_variant; the cap is still the sweep's, so the
    default arm alone takes it as inert, not as an error."""
    [(name, cell)] = _condition_configs(sweep, prompt_variant="full_context", base_overrides=SHARED)
    assert (name, cell) == ("full_context", ExperimentConfig(**NL_CELL))
    [(_, window)] = _condition_configs(sweep, prompt_variant="scratchpad_window", base_overrides=SHARED)
    assert window.scratchpad_max_chars == 1000


def test_a_variant_s_own_cap_wins_over_the_sweep_s(monkeypatch):
    own = {**ARM_OVERRIDES, "full_context_scratchpad": {
        "memory": "full_context_scratchpad", "scratchpad_max_chars": 300,
    }}
    monkeypatch.setitem(CONDITION_SETS, SWEEP, _condition_set(own))
    cells = dict(_condition_configs(SWEEP, base_overrides=SHARED))
    assert cells["full_context_scratchpad"].scratchpad_max_chars == 300
    assert cells["scratchpad_window"].scratchpad_max_chars == 1000


def test_the_sweep_cap_is_still_checked(sweep):
    with pytest.raises(ValueError, match="scratchpad_max_chars must be a positive int"):
        _condition_configs(sweep, base_overrides={**NL_CELL, "scratchpad_max_chars": 0})


def _stub_run_config(tmp_path: Path, shared: dict, name: str) -> Path:
    from interface.loader import default_maze_path

    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({
        "experiment_config": shared,
        "models": {"stub": {"provider": "openai", "model": "stub-model", "group": "stub",
                            "tasks": [str(default_maze_path("V01_empty_room.json"))]}},
    }))
    return path


def _run_sweep_locally(tmp_path: Path, shared: dict, root: str) -> dict[str, Path]:
    from scripts.run_pipeline import run_from_config
    from test_memory_arm_run_configs import _NotesAgent
    from test_run_pipeline import _MANIFEST, _STABLE_DIFFICULTY_MAX

    run_from_config(
        run_config_path=_stub_run_config(tmp_path, shared, root), manifest_path=_MANIFEST,
        seeds=[0], conditions=SWEEP, artifacts_root=tmp_path / root, run_set_id="sweep",
        agent_factory=lambda name, cfg: (_NotesAgent(), cfg["model"]),
        difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
    )
    return {p.parent.name: p.parent for p in (tmp_path / root).rglob("episode.json")}


def test_a_local_sweep_caps_its_scratchpad_arms_and_persists_the_default_arm_as_before(
    sweep, tmp_path
):
    from scorer.io import load_json

    capped = _run_sweep_locally(tmp_path, SHARED, "capped")
    plain = _run_sweep_locally(tmp_path, NL_CELL, "plain")
    assert set(capped) == set(plain) == {
        "full_context",
        "scratchpad_window__memory_scratchpad_window",
        "full_context_scratchpad__memory_full_context_scratchpad",
    }
    for name in ("scratchpad_window__memory_scratchpad_window",
                 "full_context_scratchpad__memory_full_context_scratchpad"):
        for persisted in (load_json(capped[name] / "episode.json")["config"],
                          load_json(capped[name] / "run_inputs.json")["experiment_config"]):
            assert persisted["scratchpad_max_chars"] == 1000
    # The default arm: the same persisted config and the same run hash as the
    # sweep without a cap.
    for file, key in (("episode.json", "config"), ("run_inputs.json", "experiment_config")):
        got = load_json(capped["full_context"] / file)[key]
        assert got == load_json(plain["full_context"] / file)[key]
        assert {"memory", "scratchpad_max_chars"}.isdisjoint(got)
    assert (load_json(capped["full_context"] / "run_inputs.json")["inputs_hash"]
            == load_json(plain["full_context"] / "run_inputs.json")["inputs_hash"])


def _plan(tmp_path: Path, shared: dict, root: str, prompt_variant: str | None = None) -> dict:
    from scripts.distributed_run_pipeline import prepare_job
    from test_run_pipeline import _MANIFEST, _STABLE_DIFFICULTY_MAX

    plan = prepare_job(
        run_config_path=_stub_run_config(tmp_path, shared, root), manifest_path=_MANIFEST,
        seeds=[0], conditions=SWEEP, prompt_variant=prompt_variant,
        artifacts_root=tmp_path / root, run_set_id="sweep",
        difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
    )
    return {unit["prompt_variant"]: unit for unit in plan["units"]}


def test_a_fleet_plan_applies_the_same_rule(sweep, tmp_path):
    capped = _plan(tmp_path, SHARED, "capped")
    plain = _plan(tmp_path, NL_CELL, "plain")
    assert set(capped) == set(ARM_OVERRIDES)
    for arm in ("scratchpad_window", "full_context_scratchpad"):
        assert capped[arm]["experiment_config"]["scratchpad_max_chars"] == 1000
    default, before = capped["full_context"], plain["full_context"]
    assert default["experiment_config"] == before["experiment_config"]
    assert {"memory", "scratchpad_max_chars"}.isdisjoint(default["experiment_config"])
    assert default["episode_inputs_hash"] == before["episode_inputs_hash"]
    assert default["run_dir"] == before["run_dir"]
    # A batch that plans the default arm alone takes the sweep's cap too.
    [alone] = _plan(tmp_path, SHARED, "alone", prompt_variant="full_context").values()
    assert alone["episode_inputs_hash"] == before["episode_inputs_hash"]


# --------------------------------------------------------------------------- #
# (b) a cap on the default arm itself is still an error
# --------------------------------------------------------------------------- #
def test_a_cap_on_the_default_arm_itself_is_still_an_error(monkeypatch, tmp_path):
    only = "scratchpad_max_chars only applies"
    # The cell itself.
    with pytest.raises(ValueError, match=only):
        ExperimentConfig(**SHARED)
    # A run with no conditions: its shared block is its one (default) cell.
    with pytest.raises(ValueError, match=only):
        _condition_configs(None, base_overrides=SHARED)
    # The default arm's own variant.
    monkeypatch.setitem(CONDITION_SETS, SWEEP, _condition_set(
        {**ARM_OVERRIDES, "full_context": {"scratchpad_max_chars": 1000}}
    ))
    with pytest.raises(ValueError, match=only):
        _condition_configs(SWEEP, base_overrides=NL_CELL)
    # A sweep with no scratchpad arm to apply the cap to.
    monkeypatch.setitem(CONDITION_SETS, SWEEP, _condition_set(
        {"last_3": {"context_window": "last_n"}, "full": {}}
    ))
    with pytest.raises(ValueError, match=only):
        _condition_configs(SWEEP, base_overrides=SHARED)


def test_both_entry_points_refuse_it_before_any_work(tmp_path):
    from scripts.distributed_run_pipeline import prepare_job
    from scripts.run_pipeline import run_from_config
    from test_run_pipeline import _MANIFEST, _STABLE_DIFFICULTY_MAX

    def no_agent(*_args):
        raise AssertionError("agent factory reached")

    path = _stub_run_config(tmp_path, SHARED, "default_arm_cap")
    with pytest.raises(ValueError, match="scratchpad_max_chars only applies"):
        run_from_config(
            run_config_path=path, manifest_path=_MANIFEST, seeds=[0],
            artifacts_root=tmp_path / "local", agent_factory=no_agent,
            difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
        )
    with pytest.raises(ValueError, match="scratchpad_max_chars only applies"):
        prepare_job(
            run_config_path=path, manifest_path=_MANIFEST, seeds=[0], conditions=None,
            artifacts_root=tmp_path / "fleet", run_set_id="x",
            difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
        )
    assert not (tmp_path / "local" / "runs").exists()
