"""3D render runs on the fleet (next-run-fixes item 20).

The run-config ``render`` block reaches every distributed entry point: the
coordinator plans 3D units under the versioned 3D label (their own artifact
directory and unit hash), serial and lockstep workers draw the frames with the
same backend a local run uses, and finalize keeps the label in the rows. A 2D
plan stays byte-identical to the one before fleet 3D support (pinned below).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gridworld.render3d import RENDER3D_VERSION
from interface.agents.reply import Reply
from interface.loader import default_maze_path
from interface.smoke_tests.plans import v01_empty_room_trajectory
from scorer.io import load_json, task_spec_from_payload
from scripts import run_pipeline as pipeline
from scripts.distributed_run_pipeline import (
    CoordinatorStore,
    finalize_job,
    package_run_archive,
    prepare_job,
    run_assigned_unit,
    run_lockstep_worker,
)
from scripts.prepare_combined_job import build as build_combined_job

_REPO = Path(__file__).resolve().parents[1]
_MANIFEST = _REPO / "gridworld" / "fixtures" / "manifest.json"
THREE_D = {"backend": "mujoco3d", "camera": "top_down"}
LABEL = f"mujoco3d_top_down_grid_r{RENDER3D_VERSION}"
DMAX = 1000.0


def _run_config(tmp_path: Path, render=None, name: str = "run_config.json", **extra) -> Path:
    cfg = {
        "models": {
            "stub": {
                "provider": "claude",
                "model": "stub-model",
                "group": "stub",
                "tasks": [str(default_maze_path("V01_empty_room.json"))],
            }
        },
        **extra,
    }
    if render is not None:
        cfg["render"] = render
    path = tmp_path / name
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def _prepare(tmp_path: Path, run_config: Path, root: str = "artifacts") -> dict:
    return prepare_job(
        run_config_path=run_config, manifest_path=_MANIFEST, seeds=[0], conditions=None,
        artifacts_root=tmp_path / root, run_set_id="dist", difficulty_max_static_score=DMAX,
    )


def _no_paid_call(*_args, **_kwargs):
    raise AssertionError("agent factory reached: a paid model call would follow")


class _ReplayAgent:
    def __init__(self):
        self._actions = iter(v01_empty_room_trajectory())
        self.last_usage = None

    def __call__(self, messages):
        self.last_usage = {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}
        return f"FINAL_OUTPUT: {next(self._actions, 'DONE')}"


def _replay_factory(model_key, model_config):
    return _ReplayAgent(), model_config["model"]


def _assign(artifacts: Path) -> tuple[CoordinatorStore, str, dict]:
    store = CoordinatorStore(artifacts)
    worker_id = store.register({"worker_id": "w", "capabilities": {"model_group": "stub"}})["worker_id"]
    return store, worker_id, store.assign(worker_id)["unit"]


# --------------------------------------------------------------------------- #
# 2D stays byte-identical
# --------------------------------------------------------------------------- #
# sha256 of a 2D plan's projection (unit ids, run dirs, backend, both hashes,
# unit and plan keys), recorded on main b07e0f1 before fleet 3D support. It
# moves only with PIPELINE_VERSION or a deliberate change to the 2D recipe; a
# drift here re-pays every unit of a running 2D job.
_PINNED_2D_PLAN = "1c0c332be44cfe5d4bdec668e76af07b8aad5fd3e85d12a39e4e92c9a628ad81"


def test_2d_plan_is_byte_identical_to_the_pre_3d_plan(tmp_path, monkeypatch):
    if pipeline.PIPELINE_VERSION != "0.1.2":
        pytest.skip("pinned for PIPELINE_VERSION 0.1.2")
    monkeypatch.chdir(_REPO)  # relative manifest path: job_id must not depend on the checkout
    cfg = tmp_path / "run_config.json"
    cfg.write_text(json.dumps({"models": {"stub": {
        "provider": "claude", "model": "stub-model", "group": "stub",
        "tasks": ["validation_10_v01_empty_room", "validation_10_v04_single_key"],
    }}}), encoding="utf-8")
    plan = prepare_job(
        run_config_path=cfg, manifest_path=Path("gridworld/fixtures/manifest.json"), seeds=[0, 1],
        conditions="Observation format", artifacts_root=tmp_path / "artifacts", run_set_id="pin",
        difficulty_max_static_score=DMAX,
    )
    keys = ("unit_id", "run_dir", "backend", "episode_inputs_hash", "inputs_hash")
    projection = {
        "plan_keys": sorted(plan),
        "units": [{**{k: u[k] for k in keys}, "keys": sorted(u)} for u in plan["units"]],
    }
    digest = hashlib.sha256(json.dumps(projection, sort_keys=True).encode()).hexdigest()
    assert digest == _PINNED_2D_PLAN
    assert all(u["backend"] == "minigrid" and "render" not in u for u in plan["units"])


# --------------------------------------------------------------------------- #
# Coordinator: 3D units get their own dir and hash
# --------------------------------------------------------------------------- #
def test_3d_run_config_plans_units_with_their_own_dir_and_hash(tmp_path):
    pytest.importorskip("mujoco")
    [u2] = _prepare(tmp_path, _run_config(tmp_path, None, "2d.json"), root="a2d")["units"]
    [u3] = _prepare(tmp_path, _run_config(tmp_path, THREE_D, "3d.json"), root="a3d")["units"]
    assert u3["backend"] == LABEL
    assert u3["run_dir"] == u2["run_dir"].replace("/minigrid/", f"/{LABEL}/")
    assert u3["render"] == {**THREE_D, "resolution": "grid", "start_map": False}
    assert u3["episode_inputs_hash"] != u2["episode_inputs_hash"]
    assert u3["unit_id"] != u2["unit_id"]


def test_3d_text_only_is_refused_before_any_work(tmp_path):
    cfg = _run_config(tmp_path, THREE_D, experiment_config={"observation": "text_only"})
    with pytest.raises(ValueError, match="observation 'text_only'"):
        _prepare(tmp_path, cfg)
    assert not (tmp_path / "artifacts").exists()


def test_combined_job_prepares_2d_and_3d_batches_side_by_side(tmp_path):
    pytest.importorskip("mujoco")
    specs = [
        {"run_config": str(_run_config(tmp_path, None, "a.json")), "manifest": str(_MANIFEST),
         "conditions": None, "prompt_variant": None, "run_id": "a"},
        {"run_config": str(_run_config(tmp_path, THREE_D, "b.json")), "manifest": str(_MANIFEST),
         "conditions": None, "prompt_variant": None, "run_id": "b"},
    ]
    plan = build_combined_job(specs, artifacts_root=tmp_path / "combined", run_set_id="m",
                              seeds=[0], difficulty_max=DMAX)
    assert sorted(u["backend"] for u in plan["units"]) == sorted(["minigrid", LABEL])
    assert len({u["run_dir"] for u in plan["units"]}) == 2


def test_combined_job_refuses_a_bad_3d_batch_before_preparing_any_part(tmp_path):
    specs = [
        {"run_config": str(_run_config(tmp_path, None, "a.json")), "manifest": str(_MANIFEST),
         "conditions": None, "prompt_variant": None, "run_id": "a"},
        {"run_config": str(_run_config(tmp_path, THREE_D, "b.json",
                                       experiment_config={"observation": "text_only"})),
         "manifest": str(_MANIFEST), "conditions": None, "prompt_variant": None, "run_id": "b"},
    ]
    root = tmp_path / "combined"
    with pytest.raises(ValueError, match="observation 'text_only'"):
        build_combined_job(specs, artifacts_root=root, run_set_id="m", seeds=[0], difficulty_max=DMAX)
    assert not (root / "_parts" / "a").exists()


# --------------------------------------------------------------------------- #
# Workers draw 3D frames; the coordinator verifies and finalizes them
# --------------------------------------------------------------------------- #
def _assert_3d_run(run_dir: Path, unit: dict) -> None:
    from PIL import Image

    sidecar = load_json(run_dir / "run_inputs.json")
    assert sidecar["backend"] == LABEL
    assert sidecar["inputs_hash"] == unit["episode_inputs_hash"]
    assert sidecar["render"]["render3d_version"] == RENDER3D_VERSION
    episode = load_json(run_dir / "episode.json")
    assert episode["render"] == sidecar["render"]
    assert episode["success"] is True
    dims = max(task_spec_from_payload(unit["task_payload"]).maze.dimensions)
    frame = sorted((run_dir / "frames").glob("*.png"))[0]
    assert Image.open(frame).size == (32 * dims, 32 * dims)
    assert load_json(run_dir / "run_score.json")["backend"] == LABEL


def test_serial_worker_runs_a_3d_unit_and_finalize_keeps_its_label(tmp_path):
    pytest.importorskip("mujoco")
    artifacts = tmp_path / "coordinator"
    _prepare(tmp_path, _run_config(tmp_path, THREE_D), root="coordinator")
    store, worker_id, unit = _assign(artifacts)
    worker_root = tmp_path / "worker"

    run_assigned_unit(unit, artifacts_root=worker_root, agent_factory=_replay_factory)
    _assert_3d_run(worker_root / unit["run_dir"], unit)

    archive = package_run_archive(unit, artifacts_root=worker_root)
    store.upload(worker_id, unit["unit_id"], archive.read_bytes())  # raises unless the hash matches
    assert store.load_state()["units"][unit["unit_id"]]["status"] == "verified"
    finalize_job(artifacts_root=artifacts, run_set_id="dist")
    rows = [json.loads(line) for line in (artifacts / "episode_runs.jsonl").read_text().splitlines()]
    assert [r["backend"] for r in rows] == [LABEL]
    assert load_json(artifacts / unit["run_dir"] / "run_score.json")["backend"] == LABEL


class _ReplayBatchAgent:
    """Lockstep batch double for a single V01 unit."""

    def __init__(self):
        self._actions = iter(v01_empty_room_trajectory())

    def generate_batch(self, batch):
        usage = {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}
        return [Reply(text=f"FINAL_OUTPUT: {next(self._actions, 'DONE')}", usage=usage) for _ in batch]


def test_lockstep_worker_runs_a_3d_unit(tmp_path):
    pytest.importorskip("mujoco")
    artifacts = tmp_path / "coordinator"
    plan = _prepare(tmp_path, _run_config(tmp_path, THREE_D), root="coordinator")
    agent = _ReplayBatchAgent()
    result = run_lockstep_worker(
        artifacts_root=artifacts, capabilities={"model_group": "stub"}, max_batches=1,
        client=CoordinatorStore(artifacts), agent_factory=lambda key, cfg: (agent, "stub-model"),
    )
    assert result["completed"] == 1 and not result["failed"]
    [unit] = plan["units"]
    _assert_3d_run(artifacts / unit["run_dir"], unit)


# --------------------------------------------------------------------------- #
# A worker with another render version refuses before any paid call
# --------------------------------------------------------------------------- #
def _other_version_unit(tmp_path: Path) -> tuple[Path, dict]:
    artifacts = tmp_path / "coordinator"
    _prepare(tmp_path, _run_config(tmp_path, THREE_D), root="coordinator")
    _store, _worker_id, unit = _assign(artifacts)
    return artifacts, unit


def test_serial_worker_refuses_a_unit_planned_for_another_render_version(tmp_path, monkeypatch):
    pytest.importorskip("mujoco")
    import gridworld.render3d as render3d

    _artifacts, unit = _other_version_unit(tmp_path)
    monkeypatch.setattr(render3d, "RENDER3D_VERSION", "99")
    with pytest.raises(ValueError, match="render version"):
        run_assigned_unit(unit, artifacts_root=tmp_path / "worker", agent_factory=_no_paid_call)
    assert not (tmp_path / "worker" / unit["run_dir"]).exists()


class _RecordingFailStore(CoordinatorStore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.failures: list[tuple[str, str]] = []

    def fail(self, worker_id, unit_id, reason):
        self.failures.append((unit_id, reason))
        return super().fail(worker_id, unit_id, reason)


def test_lockstep_worker_fails_back_a_unit_planned_for_another_render_version(tmp_path, monkeypatch):
    pytest.importorskip("mujoco")
    import gridworld.render3d as render3d

    artifacts = tmp_path / "coordinator"
    _prepare(tmp_path, _run_config(tmp_path, THREE_D), root="coordinator")
    monkeypatch.setattr(render3d, "RENDER3D_VERSION", "99")
    store = _RecordingFailStore(artifacts, max_unit_attempts=1)
    result = run_lockstep_worker(
        artifacts_root=artifacts, capabilities={"model_group": "stub"}, max_batches=1,
        client=store, agent_factory=_no_paid_call,
    )
    assert result["completed"] == 0 and len(result["failed"]) == 1
    [(_unit_id, reason)] = store.failures
    assert "render version" in reason


def test_batch_smoke_prepares_a_3d_unit_with_its_render(tmp_path):
    pytest.importorskip("mujoco")
    from scripts.run_batch_smoke import _prep_and_stepper

    artifacts = tmp_path / "coordinator"
    plan = _prepare(tmp_path, _run_config(tmp_path, THREE_D), root="coordinator")
    [unit_raw] = plan["units"]
    unit, prep, stepper = _prep_and_stepper(CoordinatorStore(artifacts), plan, unit_raw, artifacts)
    assert prep.run_dir == artifacts / unit["run_dir"]
    assert type(stepper.backend).__name__ == "Mujoco3DBackend"
