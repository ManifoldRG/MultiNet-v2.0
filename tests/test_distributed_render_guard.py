"""3D render runs are local-only (scripts.run_pipeline) for now.

The distributed paths hard-code the 2D artifact dir and run hash and never
pass ``render=`` to the episode builder, so a 3D run-config sent through the
fleet would silently run 2D into the 2D directory and could reuse 2D
episodes. Every distributed / lockstep / combined-job entry point must refuse
a non-2D render before any unit is prepared or any model call is made.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from interface.loader import default_maze_path
from scripts.distributed_run_pipeline import (
    CoordinatorStore,
    prepare_job,
    run_assigned_unit,
    run_lockstep_worker,
)
from scripts.prepare_combined_job import build as build_combined_job

_MANIFEST = Path(__file__).resolve().parents[1] / "gridworld" / "fixtures" / "manifest.json"
LOCAL_ONLY = r"3D render runs are local-only \(scripts\.run_pipeline\) for now; distributed support is a follow-up"
THREE_D = {"backend": "mujoco3d", "camera": "top_down"}
DMAX = 1000.0


def _run_config(tmp_path: Path, render=None, name: str = "run_config.json") -> Path:
    cfg = {
        "models": {
            "stub": {
                "provider": "claude",
                "model": "stub-model",
                "group": "stub",
                "tasks": [str(default_maze_path("V01_empty_room.json"))],
            }
        }
    }
    if render is not None:
        cfg["render"] = render
    path = tmp_path / name
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def _no_paid_call(*_args, **_kwargs):
    raise AssertionError("agent factory reached: a paid model call would follow")


def _prepare(tmp_path: Path, run_config: Path, root: str = "artifacts") -> dict:
    return prepare_job(
        run_config_path=run_config, manifest_path=_MANIFEST, seeds=[0], conditions=None,
        artifacts_root=tmp_path / root, run_set_id="dist", difficulty_max_static_score=DMAX,
    )


@pytest.mark.parametrize(
    "render",
    [THREE_D, {"backend": "mujoco3d", "camera": "first_person", "resolution": 256}],
)
def test_coordinator_prepare_refuses_a_3d_run_config_before_any_work(tmp_path, render):
    with pytest.raises(ValueError, match=LOCAL_ONLY):
        _prepare(tmp_path, _run_config(tmp_path, render))
    assert not (tmp_path / "artifacts").exists()  # no Stage 2, no plan, no state


def test_coordinator_prepare_cli_role_refuses_a_3d_run_config(tmp_path):
    from scripts.run_pipeline import main

    with pytest.raises(ValueError, match=LOCAL_ONLY):
        main([
            "--distributed-role", "coordinator-prepare",
            "--run-config", str(_run_config(tmp_path, THREE_D)), "--manifest", str(_MANIFEST),
            "--artifacts-root", str(tmp_path / "artifacts"), "--run-set-id", "dist",
            "--difficulty-max-static-score", str(DMAX),
        ])
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize("render", [None, {}, {"backend": "minigrid"}])
def test_2d_run_configs_still_prepare(tmp_path, render):
    plan = _prepare(tmp_path, _run_config(tmp_path, render))
    assert plan["units"] and all(u["backend"] == "minigrid" for u in plan["units"])


def test_combined_job_refuses_a_3d_batch_before_preparing_any_part(tmp_path):
    specs = [
        {"run_config": str(_run_config(tmp_path, None, "a.json")), "manifest": str(_MANIFEST),
         "conditions": None, "prompt_variant": None, "run_id": "a"},
        {"run_config": str(_run_config(tmp_path, THREE_D, "b.json")), "manifest": str(_MANIFEST),
         "conditions": None, "prompt_variant": None, "run_id": "b"},
    ]
    root = tmp_path / "combined"
    with pytest.raises(ValueError, match=LOCAL_ONLY):
        build_combined_job(specs, artifacts_root=root, run_set_id="m", seeds=[0], difficulty_max=DMAX)
    assert not (root / "_parts" / "a").exists()  # the 2D batch was not prepared either


def _assigned_2d_unit(tmp_path: Path) -> tuple[Path, dict]:
    artifacts = tmp_path / "coordinator"
    _prepare(tmp_path, _run_config(tmp_path), root="coordinator")
    store = CoordinatorStore(artifacts)
    worker_id = store.register({"worker_id": "w", "capabilities": {"model_group": "stub"}})["worker_id"]
    return artifacts, store.assign(worker_id)["unit"]


_3D_UNIT_MARKS = [
    {"backend": "mujoco3d_top_down_grid"},
    {"render": THREE_D},
]


@pytest.mark.parametrize("mark", _3D_UNIT_MARKS)
def test_worker_refuses_a_3d_unit_before_the_agent_is_built(tmp_path, mark):
    """Defense in depth: a plan from elsewhere (hand-edited, newer coordinator)
    carrying a 3D unit must not be run 2D by this worker."""
    _artifacts, unit = _assigned_2d_unit(tmp_path)
    unit.update(mark)
    with pytest.raises(ValueError, match=LOCAL_ONLY):
        run_assigned_unit(unit, artifacts_root=tmp_path / "worker", agent_factory=_no_paid_call)
    assert not (tmp_path / "worker" / unit["run_dir"]).exists()


class _Marking3DStore(CoordinatorStore):
    """Hands out its units marked as 3D, as a plan from elsewhere would."""

    def __init__(self, *args, mark, **kwargs):
        super().__init__(*args, **kwargs)
        self._mark = mark
        self.failures: list[tuple[str, str]] = []

    def assign(self, worker_id, capabilities=None):
        result = super().assign(worker_id, capabilities)
        if result.get("unit"):
            result["unit"].update(self._mark)
        return result

    def fail(self, worker_id, unit_id, reason):
        self.failures.append((unit_id, reason))
        return super().fail(worker_id, unit_id, reason)


@pytest.mark.parametrize("mark", _3D_UNIT_MARKS)
def test_lockstep_worker_fails_a_3d_unit_back_without_a_paid_call(tmp_path, mark):
    artifacts = tmp_path / "coordinator"
    _prepare(tmp_path, _run_config(tmp_path), root="coordinator")
    store = _Marking3DStore(artifacts, mark=mark, max_unit_attempts=1)
    result = run_lockstep_worker(
        artifacts_root=artifacts, capabilities={"model_group": "stub"}, max_batches=1,
        client=store, agent_factory=_no_paid_call,
    )
    assert result["completed"] == 0 and len(result["failed"]) == 1
    [(unit_id, reason)] = store.failures
    assert unit_id == result["failed"][0]
    assert "3D render runs are local-only" in reason
