"""Run-config ``render.observability``: the same manifest mazes under fog.

A 2D render block may impose view_cone / fog_of_war (and view_size) on every
task. It reaches every entry point the 3D render block does: local runs, the
coordinator's plan, serial and lockstep workers, and archive resume, under its
own versioned label (artifact directory and run hash). Without the key a 2D run
is untouched (its pinned hashes live in test_run_pipeline / test_distributed_3d).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gridworld.custom_env import UNKNOWN_BASE
from gridworld.render_settings import FOG_RENDER_VERSION, RenderSettings
from interface.agents.reply import Reply
from interface.loader import default_maze_path
from interface.smoke_tests.plans import v01_empty_room_trajectory
from scorer.io import load_json, task_spec_from_payload
from scripts.distributed_run_pipeline import (
    CoordinatorStore,
    package_run_archive,
    prepare_job,
    run_assigned_unit,
    run_lockstep_worker,
)
from scripts.run_pipeline import run_from_config

_REPO = Path(__file__).resolve().parents[1]
_MANIFEST = _REPO / "gridworld" / "fixtures" / "manifest.json"
FOG = {"observability": "fog_of_war"}
LABEL = f"minigrid_fog_of_war_v7_r{FOG_RENDER_VERSION}"
PROVENANCE = {"backend": "minigrid", "observability": "fog_of_war", "view_size": 7,
              "fog_render_version": FOG_RENDER_VERSION}
DMAX = 1000.0
TASK_ID = "validation_10_v01_empty_room"


def _run_config(tmp_path: Path, render=None, name="run_config.json", observation="image_only") -> Path:
    cfg = {
        "experiment_config": {"observation": observation},
        "models": {"stub": {"provider": "claude", "model": "stub-model", "group": "stub",
                            "tasks": [str(default_maze_path("V01_empty_room.json"))]}},
    }
    if render is not None:
        cfg["render"] = render
    path = tmp_path / name
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


class _ReplayAgent:
    def __init__(self):
        self._actions = iter(v01_empty_room_trajectory())
        self.last_usage = None

    def __call__(self, messages):
        self.last_usage = {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}
        return f"FINAL_OUTPUT: {next(self._actions, 'DONE')}"


def _replay_factory(_name, model_cfg):
    return _ReplayAgent(), model_cfg["model"]


def _no_paid_call(*_args, **_kwargs):
    raise AssertionError("agent factory reached: a paid model call would follow")


def _has_unknown_cells(png: Path) -> bool:
    pixels = np.asarray(Image.open(png).convert("RGB")).reshape(-1, 3)
    return bool((pixels == UNKNOWN_BASE).all(axis=1).any())


def _assert_fog_run(run_dir: Path) -> dict:
    sidecar = load_json(run_dir / "run_inputs.json")
    episode = load_json(run_dir / "episode.json")
    assert sidecar["backend"] == LABEL
    assert sidecar["render"] == episode["render"] == PROVENANCE
    rules = episode["task_spec"]["rules"]
    assert (rules["observability"], rules["view_size"]) == ("fog_of_war", 7)
    assert episode["success"] is True  # fog changes the frame, not the dynamics
    frames = sorted((run_dir / "frames").glob("*.png"))
    assert frames and _has_unknown_cells(frames[0])  # the goal corner starts hidden
    return sidecar


# --------------------------------------------------------------------------- #
# Local runs
# --------------------------------------------------------------------------- #
def test_run_from_config_runs_the_manifest_maze_under_fog(tmp_path):
    artifacts = tmp_path / "artifacts"
    for name, render in (("full.json", None), ("fog.json", FOG)):
        run_from_config(
            run_config_path=_run_config(tmp_path, render, name), manifest_path=_MANIFEST, seeds=[0],
            artifacts_root=artifacts, run_set_id="fog", agent_factory=_replay_factory,
            difficulty_max_static_score=DMAX,
        )
    runs = artifacts / "runs" / TASK_ID
    fog = _assert_fog_run(runs / LABEL / "stub-model" / "seed_0" / "default")
    full_dir = runs / "minigrid" / "stub-model" / "seed_0" / "default"
    full = load_json(full_dir / "run_inputs.json")
    assert "render" not in full and "render" not in load_json(full_dir / "episode.json")
    assert load_json(full_dir / "episode.json")["task_spec"]["rules"]["observability"] == "full"
    assert not _has_unknown_cells(sorted((full_dir / "frames").glob("*.png"))[0])
    assert fog["inputs_hash"] != full["inputs_hash"]
    # Stage 2 scores the task file once for both runs: fog leaves the dynamics alone.
    assert len(list((artifacts / "tasks").glob("*/scored_static.json"))) == 1


@pytest.mark.parametrize("observation", ["text_only", "image_text"])
def test_fog_with_a_text_observation_is_refused_before_any_paid_call(tmp_path, observation):
    with pytest.raises(ValueError, match="image_only"):
        run_from_config(
            run_config_path=_run_config(tmp_path, FOG, observation=observation),
            manifest_path=_MANIFEST, seeds=[0], artifacts_root=tmp_path / "artifacts",
            run_set_id="fog", agent_factory=_no_paid_call, difficulty_max_static_score=DMAX,
        )
    assert not (tmp_path / "artifacts").exists()


# --------------------------------------------------------------------------- #
# Fleet: plan, serial and lockstep workers, upload verification
# --------------------------------------------------------------------------- #
def _prepare(tmp_path: Path, run_config: Path, root: str = "coordinator") -> dict:
    return prepare_job(
        run_config_path=run_config, manifest_path=_MANIFEST, seeds=[0], conditions=None,
        artifacts_root=tmp_path / root, run_set_id="dist", difficulty_max_static_score=DMAX,
    )


def test_fog_units_get_their_own_dir_hash_and_render_block(tmp_path):
    [u2] = _prepare(tmp_path, _run_config(tmp_path, None, "2d.json"), root="a2d")["units"]
    [uf] = _prepare(tmp_path, _run_config(tmp_path, FOG, "fog.json"), root="afog")["units"]
    assert "render" not in u2 and u2["backend"] == "minigrid"
    assert uf["backend"] == LABEL
    assert uf["render"] == {"backend": "minigrid", "observability": "fog_of_war", "view_size": 7}
    assert uf["run_dir"] == u2["run_dir"].replace("/minigrid/", f"/{LABEL}/")
    assert uf["episode_inputs_hash"] != u2["episode_inputs_hash"]
    assert uf["unit_id"] != u2["unit_id"]


def test_fog_text_only_is_refused_before_the_plan(tmp_path):
    with pytest.raises(ValueError, match="image_only"):
        _prepare(tmp_path, _run_config(tmp_path, FOG, observation="text_only"))
    assert not (tmp_path / "coordinator").exists()


def test_serial_worker_runs_a_fog_unit_and_the_upload_verifies(tmp_path):
    artifacts = tmp_path / "coordinator"
    _prepare(tmp_path, _run_config(tmp_path, FOG))
    store = CoordinatorStore(artifacts)
    worker_id = store.register({"worker_id": "w", "capabilities": {"model_group": "stub"}})["worker_id"]
    unit = store.assign(worker_id)["unit"]
    worker_root = tmp_path / "worker"
    run_assigned_unit(unit, artifacts_root=worker_root, agent_factory=_replay_factory)
    sidecar = _assert_fog_run(worker_root / unit["run_dir"])
    assert sidecar["inputs_hash"] == unit["episode_inputs_hash"]
    archive = package_run_archive(unit, artifacts_root=worker_root)
    store.upload(worker_id, unit["unit_id"], archive.read_bytes())  # raises unless the hash matches
    assert store.load_state()["units"][unit["unit_id"]]["status"] == "verified"


class _ReplayBatchAgent:
    def __init__(self):
        self._actions = iter(v01_empty_room_trajectory())

    def generate_batch(self, batch):
        usage = {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}
        return [Reply(text=f"FINAL_OUTPUT: {next(self._actions, 'DONE')}", usage=usage) for _ in batch]


def test_lockstep_worker_runs_a_fog_unit(tmp_path):
    artifacts = tmp_path / "coordinator"
    [unit] = _prepare(tmp_path, _run_config(tmp_path, FOG))["units"]
    agent = _ReplayBatchAgent()
    result = run_lockstep_worker(
        artifacts_root=artifacts, capabilities={"model_group": "stub"}, max_batches=1,
        client=CoordinatorStore(artifacts), agent_factory=lambda key, cfg: (agent, "stub-model"),
    )
    assert result["completed"] == 1 and not result["failed"]
    _assert_fog_run(artifacts / unit["run_dir"])


def test_a_worker_on_another_fog_render_version_refuses_the_unit(tmp_path, monkeypatch):
    import gridworld.render_settings as render_settings

    artifacts = tmp_path / "coordinator"
    _prepare(tmp_path, _run_config(tmp_path, FOG))
    store = CoordinatorStore(artifacts)
    worker_id = store.register({"worker_id": "w", "capabilities": {"model_group": "stub"}})["worker_id"]
    unit = store.assign(worker_id)["unit"]
    monkeypatch.setattr(render_settings, "FOG_RENDER_VERSION", 99)
    with pytest.raises(ValueError, match="render version or settings differ"):
        run_assigned_unit(unit, artifacts_root=tmp_path / "worker", agent_factory=_no_paid_call)


# --------------------------------------------------------------------------- #
# Archive resume rebuilds the fogged backend
# --------------------------------------------------------------------------- #
def test_resume_reads_a_fog_archive_back_into_the_same_settings(tmp_path):
    from scripts.resume_from_archive import recorded_render

    run_from_config(
        run_config_path=_run_config(tmp_path, FOG), manifest_path=_MANIFEST, seeds=[0],
        artifacts_root=tmp_path / "artifacts", run_set_id="fog", agent_factory=_replay_factory,
        difficulty_max_static_score=DMAX,
    )
    run_dir = tmp_path / "artifacts" / "runs" / TASK_ID / LABEL / "stub-model" / "seed_0" / "default"
    episode, run_inputs = load_json(run_dir / "episode.json"), load_json(run_dir / "run_inputs.json")
    render = recorded_render(episode, run_inputs)
    assert render == PROVENANCE
    # resume_from_archive's reconstruction of the settings from that block
    keys = ("backend", "camera", "resolution", "start_map", "observability", "view_size")
    settings = RenderSettings.from_run_config({"render": {k: render[k] for k in keys if k in render}})
    spec = task_spec_from_payload(episode["task_spec"])
    assert settings.provenance(spec) == render and settings.label == run_inputs["backend"]
    assert settings.apply_to_spec(spec) == spec  # the archived spec already carries the fog
