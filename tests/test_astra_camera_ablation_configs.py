"""The camera-ablation arms must differ only in how the frame is drawn."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gridworld.render_settings import RenderSettings
from scripts.run_pipeline import load_manifest, load_run_config, resolve_task_rows

FIXTURES = Path(__file__).resolve().parents[1] / "gridworld" / "fixtures"
ARMS = ["2d", "top_down", "chase", "fixed_angled", "first_person"]
# The 2026-09-18 run used the narrow eye; the preset was widened afterwards,
# so the arm keeps its original camera under its new name.
ARM_CAMERA = {"first_person": "first_person_narrow"}
MANIFEST = FIXTURES / "manifest.astra_camera_ablation.json"


def _config(arm: str) -> dict:
    return load_run_config(FIXTURES / f"run_config.astra_camera_{arm}.json")


def test_manifest_is_six_resolvable_mazes_with_switch_and_control_arms():
    rows = load_manifest(MANIFEST)
    resolved = resolve_task_rows(["all"], rows, MANIFEST)
    assert len(resolved) == 6
    for row in resolved:
        assert (Path(__file__).resolve().parents[1] / row["source"]).exists()
    switch = [r for r in resolved if "g1" in (r.get("expected_mechanisms") or [])]
    controls = [r for r in resolved if "g1" not in (r.get("expected_mechanisms") or [])]
    assert len(switch) == 4 and len(controls) == 2  # switch cliff + solved-in-2D controls
    assert not any(r["condition"] == "B" for r in resolved)  # no blind maze: colours matter here


@pytest.mark.parametrize("arm", ARMS)
def test_each_arm_resolves_the_same_six_mazes(arm):
    config = _config(arm)
    assert Path(config["manifest"]) == MANIFEST.relative_to(Path(__file__).resolve().parents[1])
    rows = load_manifest(MANIFEST)
    assert len(resolve_task_rows(config["models"]["gpt6_astra"]["tasks"], rows, MANIFEST)) == 6


def test_arms_differ_only_by_their_render_block():
    configs = {arm: _config(arm) for arm in ARMS}
    stripped = {arm: {k: v for k, v in cfg.items() if k not in ("render", "description")}
                for arm, cfg in configs.items()}
    assert len({json.dumps(s, sort_keys=True) for s in stripped.values()}) == 1
    assert "render" not in configs["2d"]  # the anchor is MiniGrid's own frame
    for arm in ARMS[1:]:
        settings = RenderSettings.from_run_config(configs[arm])
        assert settings.backend == "mujoco3d" and settings.camera == ARM_CAMERA.get(arm, arm)
        assert settings.resolution == "grid"  # same image-token budget as the 2D anchor


def test_the_five_arms_fit_the_run_budget():
    caps = {arm: _config(arm)["models"]["gpt6_astra"]["spend_cap_usd"] for arm in ARMS}
    assert sum(caps.values()) == 30  # Sean's budget; each arm stops itself at its share
