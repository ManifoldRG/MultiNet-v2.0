from __future__ import annotations

from typing import List, Tuple

from .models import (
    Backbone,
    DenseMazeParams,
    DistractorMode,
    LogicChain,
    MazeGenSpec,
    WindingCorridorParams,
)
from .orchestrator import build_valid_maze_with_retries


def _spec(
    name_backbone: Backbone,
    logic: LogicChain,
    seed: int,
    *,
    width: int = 12,
    height: int = 10,
    distractor: DistractorMode = DistractorMode.NONE,
    max_distractors: int = 0,
    params: object = None,
) -> MazeGenSpec:
    tier = 1 if logic == LogicChain.NONE else 2 if logic in {LogicChain.KD, LogicChain.SG} else 3
    return MazeGenSpec(
        backbone=name_backbone,
        logic_chain=logic,
        difficulty_tier=tier,
        grid_width=width,
        grid_height=height,
        seed=seed,
        distractor_mode=distractor,
        max_distractors=max_distractors,
        backbone_params=params,
    )


def _smoke_test_cases() -> List[Tuple[str, MazeGenSpec]]:
    winding = WindingCorridorParams()
    dense = DenseMazeParams(add_loops=False, loop_count=0)
    return [
        ("winding_none_8", _spec(Backbone.WINDING_CORRIDOR, LogicChain.NONE, 103, width=8, height=8, params=winding)),
        ("winding_kd_10", _spec(Backbone.WINDING_CORRIDOR, LogicChain.KD, 104, width=10, height=10, params=winding)),
        ("winding_sg_14", _spec(Backbone.WINDING_CORRIDOR, LogicChain.SG, 105, width=14, height=14, params=winding)),
        ("winding_ks_14", _spec(Backbone.WINDING_CORRIDOR, LogicChain.KS, 106, width=14, height=14, params=winding)),
        ("winding_sk_14", _spec(Backbone.WINDING_CORRIDOR, LogicChain.SK, 107, width=14, height=14, params=winding)),
        ("winding_kk_14", _spec(Backbone.WINDING_CORRIDOR, LogicChain.KK, 108, width=14, height=14, params=winding)),
        ("winding_ksk_14", _spec(Backbone.WINDING_CORRIDOR, LogicChain.KSK, 114, width=14, height=14, params=winding)),
        ("winding_sg_inactive_switch_10", _spec(
            Backbone.WINDING_CORRIDOR, LogicChain.SG, 115, width=10, height=10,
            distractor=DistractorMode.WRONG_SWITCHES, max_distractors=1, params=winding,
        )),
        ("winding_kk_wrong_keys_14", _spec(
            Backbone.WINDING_CORRIDOR, LogicChain.KK, 109, width=14, height=14,
            distractor=DistractorMode.WRONG_KEYS, max_distractors=2, params=winding,
        )),
        ("dense_none_8", _spec(Backbone.DENSE_MAZE, LogicChain.NONE, 110, width=8, height=8, params=dense)),
        ("dense_sg_10", _spec(Backbone.DENSE_MAZE, LogicChain.SG, 111, width=10, height=10, params=dense)),
        ("dense_ks_14", _spec(Backbone.DENSE_MAZE, LogicChain.KS, 112, width=14, height=14, params=dense)),
        ("dense_ksk_14", _spec(Backbone.DENSE_MAZE, LogicChain.KSK, 116, width=14, height=14, params=dense)),
        ("dense_kk_distractor_chain_14", _spec(
            Backbone.DENSE_MAZE, LogicChain.KK, 113, width=14, height=14,
            distractor=DistractorMode.DISTRACTOR_CHAIN, max_distractors=1, params=dense,
        )),
    ]


def run_smoke_tests() -> List[dict]:
    reports: List[dict] = []

    for name, spec in _smoke_test_cases():
        _, report, final_spec = build_valid_maze_with_retries(spec)

        item = {
            "name": name,
            "backbone": final_spec.backbone.value,
            "logic": final_spec.logic_chain.value,
            "distractor_mode": final_spec.distractor_mode.value,
            "is_valid": report["is_valid"],
            "reasons": report["reasons"],
            "optimal_cost": report["solver_result"]["optimal_cost"],
        }

        if final_spec.logic_chain != LogicChain.NONE:
            item["interactions"] = report["solver_result"].get("interactions", [])

        reports.append(item)

    return reports


if __name__ == "__main__":
    for item in run_smoke_tests():
        print(item)
