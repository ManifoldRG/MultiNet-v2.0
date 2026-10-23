import random

from gridworld.baselines import plan_bfs_path
from gridworld.task_spec import TaskSpecification
from gridworld.task_validator import TaskValidator

from .generators import generate_from_spec
from .mechanisms import place_key_door_pair, place_switch_gate_pair, place_sequential_chain, add_dead_end_distractors, add_wrong_key_distractors, add_distractor_chain
from .models import (
    Backbone,
    LogicChain,
    MazeGenSpec,
    MultiRouteParams,
    SequentialChainParams,
    SideVaultParams,
    WindingCorridorParams,
    DenseMazeParams,
    DistractorMode,
)





def sample_spec(seed: int) -> MazeGenSpec:
    rng = random.Random(seed)

    backbone = rng.choice(
        [
            Backbone.WINDING_CORRIDOR,
            Backbone.MULTI_ROUTE,
            Backbone.SIDE_VAULT,
            Backbone.SEQUENTIAL_CHAIN,
            Backbone.DENSE_MAZE,
        ]
    )


    if backbone == Backbone.WINDING_CORRIDOR:
        logic_chain = rng.choice([LogicChain.NONE, LogicChain.KD, LogicChain.SG])
        distractor_mode = rng.choice([
            DistractorMode.NONE,
            DistractorMode.DEAD_END_ROOMS,
            DistractorMode.WRONG_KEYS
        ])
        max_distractors = 0 if distractor_mode == DistractorMode.NONE else rng.randint(1, 2)
        return MazeGenSpec(
            backbone=Backbone.WINDING_CORRIDOR,
            logic_chain=logic_chain,
            difficulty_tier=1 if logic_chain == LogicChain.NONE else 2,
            grid_width=rng.choice([18, 20, 22]),
            grid_height=rng.choice([8, 9, 10]),
            seed=seed,
            backbone_params=WindingCorridorParams(
                corridor_length=rng.randint(18, 34),
                turn_count=rng.randint(3, 7),
                segment_min_length=2,
                segment_max_length=rng.randint(4, 6),
                corridor_width=1,
                allow_side_stubs=rng.random() < 0.3,
                side_stub_count=rng.randint(1, 3),
                start_goal_at_ends=True,
                self_proximity_budget=0,
            ),
            distractor_mode=distractor_mode,
            max_distractors=max_distractors,
        )

    if backbone == Backbone.MULTI_ROUTE:
        logic_chain = LogicChain.NONE
        distractor_mode = rng.choice([
            DistractorMode.NONE,
            DistractorMode.DEAD_END_ROOMS,
            DistractorMode.WRONG_KEYS
        ])
        max_distractors = 0 if distractor_mode == DistractorMode.NONE else rng.randint(1, 2)
        return MazeGenSpec(
            backbone=Backbone.MULTI_ROUTE,
            logic_chain=logic_chain,
            difficulty_tier=1,
            grid_width=rng.choice([12, 14, 16]),
            grid_height=rng.choice([10, 12, 14]),
            seed=seed,
            backbone_params=MultiRouteParams(
                num_routes=rng.randint(2, 4),
                min_route_length=rng.randint(6, 10),
                max_route_length=rng.randint(12, 20),
                allow_route_rejoin=rng.random() < 0.8,
                route_overlap_budget=rng.randint(0, 2),
                route_asymmetry=round(rng.uniform(0.2, 0.9), 2),
                dead_end_branch_count=rng.randint(0, 2),
                main_corridor_width=1,
            ),
            distractor_mode=distractor_mode,
            max_distractors=max_distractors,
        )

    if backbone == Backbone.SIDE_VAULT:
        logic_chain = rng.choice([LogicChain.NONE, LogicChain.KD, LogicChain.SG])
        distractor_mode = rng.choice([
            DistractorMode.NONE,
            DistractorMode.DEAD_END_ROOMS,
            DistractorMode.WRONG_KEYS
        ])
        max_distractors = 0 if distractor_mode == DistractorMode.NONE else rng.randint(1, 2)
        return MazeGenSpec(
            backbone=Backbone.SIDE_VAULT,
            logic_chain=logic_chain,
            difficulty_tier=1 if logic_chain == LogicChain.NONE else 2,
            grid_width=rng.choice([12, 14, 16]),
            grid_height=rng.choice([10, 12, 14]),
            seed=seed,
            backbone_params=SideVaultParams(
                foyer_size=rng.choice(["small", "medium", "large"]),
                vault_branch_depth=rng.randint(3, 6),
                vault_branch_turns=rng.randint(0, 2),
                main_route_length_before_blocker=rng.randint(6, 10),
                blocker_distance_from_goal=rng.randint(1, 3),
                vault_position_mode=rng.choice(["upper", "lower", "left", "right"]),
                mainline_shape=rng.choice(["linear", "bent"]),
                allow_small_dead_ends=False,
            ),
            distractor_mode=distractor_mode,
            max_distractors=max_distractors,
        )

    if backbone == Backbone.SEQUENTIAL_CHAIN:
        logic_chain = rng.choice([LogicChain.NONE, LogicChain.KD, LogicChain.SG, LogicChain.KS, LogicChain.SK, LogicChain.KK])
        distractor_mode = rng.choice([
            DistractorMode.NONE,
            DistractorMode.DEAD_END_ROOMS,
            DistractorMode.WRONG_KEYS
        ])
        max_distractors = 0 if distractor_mode == DistractorMode.NONE else rng.randint(1, 2)
        return MazeGenSpec(
            backbone=Backbone.SEQUENTIAL_CHAIN,
            logic_chain=logic_chain,
            difficulty_tier=1 if logic_chain == LogicChain.NONE else 2 if logic_chain in {LogicChain.KD, LogicChain.SG} else 3,
            grid_width=rng.choice([14, 16, 18]),
            grid_height=rng.choice([10, 12, 14]),
            seed=seed,
            backbone_params=SequentialChainParams(
                zone1_size=rng.choice(["small", "medium", "large"]),
                zone2_size=rng.choice(["small", "medium", "large"]),
                choke1_orientation=rng.choice(["horizontal", "vertical"]),
                choke2_orientation=rng.choice(["horizontal", "vertical"]),
                pickup1_branch_depth=rng.randint(0, 2),
                pickup2_branch_depth=rng.randint(1, 3),
                zone2_internal_branches=rng.randint(0, 2),
                main_progress_shape=rng.choice(["linear", "alternating_upper_lower"]),
                allow_local_dead_ends=False,
            ),
            distractor_mode=distractor_mode,
            max_distractors=max_distractors,
        )


    if backbone == Backbone.DENSE_MAZE:
        logic_chain = rng.choice([LogicChain.NONE, LogicChain.KD, LogicChain.SG, LogicChain.KS, LogicChain.SK, LogicChain.KK])
        distractor_mode = rng.choice([
            DistractorMode.NONE,
            DistractorMode.WRONG_KEYS,
            DistractorMode.DISTRACTOR_CHAIN,
        ])
        max_distractors = 0 if distractor_mode == DistractorMode.NONE else rng.randint(1, 2)
        return MazeGenSpec(
            backbone=Backbone.DENSE_MAZE,
            logic_chain=logic_chain,
            difficulty_tier=1 if logic_chain == LogicChain.NONE else 2 if logic_chain in {LogicChain.KD, LogicChain.SG} else 3,
            grid_width=0,   # ignored by dense_maze generator
            grid_height=0,  # ignored by dense_maze generator
            seed=seed,
            backbone_params=DenseMazeParams(
                maze_width_cells=rng.choice([5, 6, 7]),
                maze_height_cells=rng.choice([5, 6, 7]),
                add_loops=rng.random() < 0.35,
                loop_count=rng.randint(1, 3),
                pickup1_progress_min=0.15,
                pickup1_progress_max=0.35,
                blocker1_progress_min=0.40,
                blocker1_progress_max=0.55,
                pickup2_progress_min=0.55,
                pickup2_progress_max=0.75,
                blocker2_progress_min=0.80,
                blocker2_progress_max=0.92,
            ),
            distractor_mode=distractor_mode,
            max_distractors=max_distractors,
        )


def acceptance_report(task: TaskSpecification) -> dict:
    """Accept a task only when the gridworld schema and TaskValidator agree."""
    schema_ok, schema_reasons = task.validate()
    reasons = list(schema_reasons)
    validator = TaskValidator(task)
    beatable, _, message = validator.validate()
    planned = plan_bfs_path(task)
    if not beatable:
        reasons.append(message)
    else:
        reasons.extend(validator.validate_mechanism_necessity())
        if not validator.validate_chain_ordering():
            reasons.append("dependency chain order is not required")
        reasons.extend(validator.validate_distractor_safety(base_beatable=True))
        if planned.success:
            task.max_steps = max(task.max_steps, len(planned.action_labels) * 4)

    return {
        "is_valid": schema_ok and not reasons,
        "reasons": reasons,
        "solver_result": {
            "optimal_cost": len(planned.action_labels) if planned.success else None,
            "path": [tuple(pos) for pos in planned.positions] if planned.success else [],
            "interactions": list(planned.action_labels) if planned.success else [],
        },
    }


def build_maze_from_spec(spec: MazeGenSpec) -> tuple[TaskSpecification, dict]:
    """Carve a layout, place the requested chain, and validate it with TaskValidator."""
    task, slots = generate_from_spec(spec)

    if spec.logic_chain == LogicChain.KD:
        place_key_door_pair(task, slots, color="red")
    elif spec.logic_chain == LogicChain.SG:
        place_switch_gate_pair(task, slots, switch_id="1")
    elif spec.logic_chain == LogicChain.KS:
        place_sequential_chain(task, slots, "ks", {"keys": ["red"], "switches": ["1"]})
    elif spec.logic_chain == LogicChain.SK:
        place_sequential_chain(task, slots, "sk", {"keys": ["red"], "switches": ["1"]})
    elif spec.logic_chain == LogicChain.KK:
        place_sequential_chain(task, slots, "kk", {"keys": ["red", "blue"]})
    elif spec.logic_chain != LogicChain.NONE:
        raise ValueError(f"Unsupported logic chain: {spec.logic_chain}")

    return task, acceptance_report(task)


def build_maze_with_distractors(spec: MazeGenSpec):
    task, report = build_maze_from_spec(spec)
    if not report["is_valid"] or spec.logic_chain == LogicChain.NONE:
        return task, report

    if spec.distractor_mode == DistractorMode.DEAD_END_ROOMS:
        add_dead_end_distractors(task, count=max(1, spec.max_distractors), branch_length=2)
    elif spec.distractor_mode == DistractorMode.WRONG_KEYS:
        add_dead_end_distractors(task, count=max(1, spec.max_distractors), branch_length=2)
        add_wrong_key_distractors(
            task,
            colors=["yellow", "green", "purple"][: max(1, spec.max_distractors)],
        )
    elif spec.distractor_mode == DistractorMode.DISTRACTOR_CHAIN:
        add_dead_end_distractors(task, count=1, branch_length=2)
        add_distractor_chain(task, chain_type="kd", color="green")

    return task, acceptance_report(task)



def build_valid_maze_with_retries(spec: MazeGenSpec, max_retries: int = 10):
    """
    Try to build a valid maze from a spec, retrying with nearby seeds if needed.

    Returns:
        (task, report, final_spec)

    Raises:
        ValueError if no valid maze is found after max_retries attempts.
    """
    for retry_idx in range(max_retries):
        trial_seed = spec.seed + retry_idx

        trial_spec = MazeGenSpec(
            backbone=spec.backbone,
            logic_chain=spec.logic_chain,
            difficulty_tier=spec.difficulty_tier,
            grid_width=spec.grid_width,
            grid_height=spec.grid_height,
            seed=trial_seed,
            distractor_mode=spec.distractor_mode,
            max_distractors=spec.max_distractors,
            backbone_params=spec.backbone_params,
        )

        obj, report = build_maze_with_distractors(trial_spec)
        if report["is_valid"]:
            return obj, report, trial_spec

    raise ValueError(
        f"Could not generate a valid maze after {max_retries} retries "
        f"for backbone={spec.backbone.value}, logic={spec.logic_chain.value}, seed={spec.seed}"
    )