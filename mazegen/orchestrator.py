import random

from gridworld.baselines import plan_bfs_path
from gridworld.task_spec import TaskSpecification
from gridworld.task_validator import TaskValidator

from .generators import generate_from_spec, slots_for_logic
from .mechanisms import (
    place_key_door_pair,
    place_switch_gate_pair,
    place_sequential_chain,
    add_wrong_key_distractors,
    add_distractor_chain,
    add_inactive_switch_distractor,
)
from .models import (
    Backbone,
    LogicChain,
    MazeGenSpec,
    WindingCorridorParams,
    DenseMazeParams,
    DistractorMode,
)

WINDING_SIZES = (8, 10, 14)
DENSE_SIZES = (10, 14)





_SWITCH_LOGICS = {LogicChain.SG, LogicChain.KS, LogicChain.SK, LogicChain.KSK}


def _tier(logic: LogicChain) -> int:
    if logic == LogicChain.NONE:
        return 1
    if logic in {LogicChain.KD, LogicChain.SG}:
        return 2
    if logic == LogicChain.KSK:
        return 4
    return 3


def _sample_logic_and_distractors(
    rng: random.Random,
    backbone: Backbone,
) -> tuple[LogicChain, DistractorMode, int]:
    """Sample a chain and one distractor family from the benchmark suites.

    D1 is a wrong-color key on a required chain, on either layout. D2 is a
    wrong-color key plus an inactive switch, on a chain that already has a
    switch. D3 is one key-door on an existing dead end, and only on dense mazes.
    """
    logic = rng.choice(list(LogicChain))
    choices = [DistractorMode.NONE]
    if logic != LogicChain.NONE:
        choices.append(DistractorMode.WRONG_KEYS)
    if logic in _SWITCH_LOGICS:
        choices.append(DistractorMode.WRONG_SWITCHES)
    if backbone == Backbone.DENSE_MAZE:
        choices.append(DistractorMode.DISTRACTOR_CHAIN)
    mode = rng.choice(choices)
    if mode == DistractorMode.WRONG_KEYS:
        count = rng.randint(1, 2)
    elif mode == DistractorMode.NONE:
        count = 0
    else:
        count = 1
    return logic, mode, count


def sample_spec(seed: int) -> MazeGenSpec:
    rng = random.Random(seed)
    backbone = rng.choice([
        Backbone.WINDING_CORRIDOR,
        Backbone.DENSE_MAZE,
    ])
    logic, distractor_mode, max_distractors = _sample_logic_and_distractors(rng, backbone)
    tier = _tier(logic)
    size = rng.choice(WINDING_SIZES if backbone == Backbone.WINDING_CORRIDOR else DENSE_SIZES)

    if backbone == Backbone.WINDING_CORRIDOR:
        return MazeGenSpec(
            backbone=Backbone.WINDING_CORRIDOR,
            logic_chain=logic,
            difficulty_tier=tier,
            grid_width=size,
            grid_height=size,
            seed=seed,
            backbone_params=WindingCorridorParams(vertical=rng.random() < 0.5),
            distractor_mode=distractor_mode,
            max_distractors=max_distractors,
        )

    if backbone == Backbone.DENSE_MAZE:
        return MazeGenSpec(
            backbone=Backbone.DENSE_MAZE,
            logic_chain=logic,
            difficulty_tier=tier,
            grid_width=size,
            grid_height=size,
            seed=seed,
            backbone_params=DenseMazeParams(
                add_loops=rng.random() < 0.35,
                loop_count=rng.randint(1, 3),
            ),
            distractor_mode=distractor_mode,
            max_distractors=max_distractors,
        )

    raise ValueError(f"Unsupported backbone: {backbone}")


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
        # A dead-end key-door on a maze with no required chain is a distractor.
        # Necessity only applies to the required chain.
        if task.dependency_chain is not None or not task.distractors:
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
    task = generate_from_spec(spec)
    slots = slots_for_logic(task, spec.logic_chain)

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
    elif spec.logic_chain == LogicChain.KSK:
        place_sequential_chain(task, slots, "ksk", {"keys": ["red", "blue"], "switches": ["1"]})
    elif spec.logic_chain != LogicChain.NONE:
        raise ValueError(f"Unsupported logic chain: {spec.logic_chain}")

    return task, acceptance_report(task)


def build_maze_with_distractors(spec: MazeGenSpec):
    task, report = build_maze_from_spec(spec)
    if not report["is_valid"]:
        return task, report

    if spec.distractor_mode == DistractorMode.WRONG_SWITCHES and spec.logic_chain in _SWITCH_LOGICS:
        add_inactive_switch_distractor(task)
        add_wrong_key_distractors(task, colors=["yellow"])
        return task, acceptance_report(task)

    if spec.distractor_mode == DistractorMode.WRONG_KEYS and spec.logic_chain != LogicChain.NONE:
        add_wrong_key_distractors(
            task,
            colors=["yellow", "green", "purple"][: spec.max_distractors],
        )
        return task, acceptance_report(task)

    if spec.distractor_mode == DistractorMode.DISTRACTOR_CHAIN and spec.backbone == Backbone.DENSE_MAZE:
        add_distractor_chain(task, chain_type="kd", color="yellow")
        return task, acceptance_report(task)

    return task, report



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