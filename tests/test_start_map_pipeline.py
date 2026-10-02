"""run_from_config with render.start_map: its own artifact label, the map
logged once, start_map in the render provenance, and fail-fast validation
before any agent is built."""

from __future__ import annotations

import json

import pytest

from interface.loader import default_maze_path
from interface.smoke_tests.plans import v01_empty_room_trajectory
from prompting_experiments.prompt_templates import user as user_templates
from scorer.io import load_json
from scripts.run_pipeline import run_from_config
from test_run_pipeline import _MANIFEST, _STABLE_DIFFICULTY_MAX, ReplayAgent

TASK = default_maze_path("V01_empty_room.json")


def _run_config(tmp_path, render, **extra):
    run_config = {
        "render": render,
        "models": {"stub": {"provider": "claude", "model": "stub-model", "tasks": [str(TASK)]}},
        **extra,
    }
    path = tmp_path / "run_config.json"
    path.write_text(json.dumps(run_config), encoding="utf-8")
    return path


def _run(tmp_path, cfg_path, factory):
    return run_from_config(
        run_config_path=cfg_path,
        manifest_path=_MANIFEST,
        seeds=[0],
        artifacts_root=tmp_path / "artifacts",
        run_set_id="startmap",
        agent_factory=factory,
        difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
    )


def test_start_map_run_writes_its_own_artifacts(tmp_path):
    pytest.importorskip("mujoco")
    render = {"backend": "mujoco3d", "camera": "first_person", "start_map": True}
    cfg_path = _run_config(
        tmp_path, render, experiment_config={"prompting": "minimal", "observation": "image_only"}
    )
    _run(tmp_path, cfg_path, lambda name, cfg: (ReplayAgent(v01_empty_room_trajectory()), cfg["model"]))

    run_dir = (
        tmp_path / "artifacts" / "runs" / "validation_10_v01_empty_room" / "mujoco3d_first_person_grid_map"
        / "stub-model" / "seed_0" / "default"
    )
    sidecar = load_json(run_dir / "run_inputs.json")
    assert sidecar["backend"] == "mujoco3d_first_person_grid_map"
    assert sidecar["render"] == {**render, "resolution": "grid"}
    episode = load_json(run_dir / "episode.json")
    assert episode["render"]["start_map"] is True
    assert episode["transcript"][0]["start_map_frame"] == "frames/start_map.png"
    assert (run_dir / "frames" / "start_map.png").exists()
    first_query = load_json(run_dir / "queries" / "query_001" / "query.json")
    user = first_query["agent_messages"][1]["content"]
    assert user[0] == {"type": "text", "text": user_templates.START_MAP_MINIMAL}
    assert user[1]["type"] == "image"


def _never(name, cfg):
    raise AssertionError("no agent may be built for an invalid run-config")


@pytest.mark.parametrize(
    "render, extra, message",
    [
        (
            {"backend": "mujoco3d", "camera": "first_person", "start_map": True},
            {"experiment_config": {"observation": "text_only"}},
            "text_only",
        ),
        ({"backend": "mujoco3d", "camera": "top_down", "start_map": True}, {}, "top_down"),
        ({"backend": "minigrid", "start_map": True}, {}, "start_map"),
        ({"backend": "mujoco3d", "camera": "chase", "start_map": "true"}, {}, "bool"),
    ],
)
def test_bad_start_map_configs_fail_before_any_agent(tmp_path, render, extra, message):
    with pytest.raises(ValueError, match=message):
        _run(tmp_path, _run_config(tmp_path, render, **extra), _never)


def test_a_condition_set_with_a_text_only_arm_is_rejected(tmp_path):
    render = {"backend": "mujoco3d", "camera": "first_person", "start_map": True}
    cfg_path = _run_config(tmp_path, render)
    with pytest.raises(ValueError, match="text_only"):
        run_from_config(
            run_config_path=cfg_path,
            manifest_path=_MANIFEST,
            seeds=[0],
            conditions="Observation format",
            artifacts_root=tmp_path / "artifacts",
            agent_factory=_never,
            difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
        )
