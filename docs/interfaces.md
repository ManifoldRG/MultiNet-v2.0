# Multinet-v2.0 Interfaces

This document is the contract reference for the public interfaces used by the
current codebase. Paths are relative to the repository root.

## TaskSpecification

Module: `gridworld/task_spec.py`

`TaskSpecification` is the canonical gridworld task object. Load it with
`TaskSpecification.from_json(path)` or `TaskSpecification.from_dict(data)`.

Required top-level fields:

- `task_id: str`
- `seed: int`
- `difficulty_tier: int`
- `maze: MazeLayout`
- `goal: GoalSpec`
- `max_steps: int`

Optional top-level fields:

- `version: str = "1.0"`
- `description: str = ""`
- `mechanisms: MechanismSet = {}`
- `rules: Rules = {}`
- `dependency_chain`
- `distractors`
- `metadata`

Supported mechanisms:

- `keys`: `id`, `position`, `color`
- `doors`: `id`, `position`, `requires_key`, `initial_state`
- `switches`: `id`, `position`, `controls`, `color`, `switch_type`, `initial_state`
- `gates`: `id`, `position`, `initial_state`
- `blocks`: `id`, `position`, `pushable`, `color`
- `teleporters`: `id`, `position_a`, `position_b`, `bidirectional`
- `hazards`: `id`, `position`, `hazard_type`

Supported rules:

- `key_consumption: bool`
- `switch_type: "toggle" | "hold" | "one_shot"`
- `hidden_mechanisms: list[str]`
- `observability: "full" | "view_cone" | "fog_of_war"`
- `view_size: int`, odd and at least 3

Supported goal types:

- `reach_position`
- `collect_all`
- `push_block_to`
- `survive_steps`

Validation:

```python
is_valid, errors = spec.validate()
```

Validation checks bounds, border-wall conflicts, duplicate mechanism IDs,
position overlaps, door-key color references, switch-gate references,
hidden-mechanism references, dependency-chain references, distractor references,
goal references, `max_steps`, and `view_size`.

## Backend Interface

Module: `gridworld/backends/base.py`

All grid backends implement `AbstractGridBackend`:

```python
backend.configure(task_spec)
obs, state, info = backend.reset(seed=None)
obs, reward, terminated, truncated, state, info = backend.step(action)
image = backend.render()
mission = backend.get_mission_text()
state = backend.get_state()
backend.close()
```

Backends use the MiniGrid-compatible 7-action external interface:

| ID | Name | Meaning |
| --- | --- | --- |
| 0 | `turn_left` | Rotate counter-clockwise |
| 1 | `turn_right` | Rotate clockwise |
| 2 | `move_forward` | Move one cell in the facing direction |
| 3 | `pickup` | Pick up an object in front |
| 4 | `drop` | Drop the held object |
| 5 | `toggle` | Interact with a door, switch, or object in front |
| 6 | `done` | Wait/no-op |

`MiniGridBackend` executes these actions directly. `MultiGridBackend` translates
them to the native `multigrid.agent.Action` enum:

| External ID | Native ID |
| --- | --- |
| 0 | 2 (`TURN_LEFT`) |
| 1 | 3 (`TURN_RIGHT`) |
| 2 | 0 (`FORWARD`) |
| 3 | 4 (`PICKUP`) |
| 4 | 5 (`DROP`) |
| 5 | 6 (`TOGGLE`) |
| 6 | 8 (`WAIT`) |

## GridState

Module: `gridworld/backends/base.py`

`GridState` is the backend-independent snapshot returned by `reset`, `step`, and
`get_state`.

Fields:

- `agent_position: tuple[int, int]`
- `agent_direction: int`
- `agent_carrying: str | None`
- `step_count: int`
- `max_steps: int`
- `terminated: bool`
- `truncated: bool`
- `reward: float`
- `open_doors: set[str]`
- `collected_keys: set[str]`
- `active_switches: set[str]`
- `open_gates: set[str]`
- `block_positions: dict[str, tuple[int, int]]`
- `teleporter_cooldowns: dict[str, int]`
- `goal_reached: bool`
- `observability_mode: str`
- `visible_cells: set[tuple[int, int]]`
- `explored_cells: set[tuple[int, int]]`

Use `state.to_dict()` and `GridState.from_dict(data)` for serialization.

## TaskParser

Module: `gridworld/task_parser.py`

`TaskParser` creates a populated `CustomMiniGridEnv` from a `TaskSpecification`.
It is used by `MiniGridBackend`.

```python
parser = TaskParser(render_mode="rgb_array")
env = parser.parse(spec, seed=spec.seed)
env = parser.parse_file("gridworld/tasks/tier1/maze_simple_001.json")
env = parser.parse_dict(data)
```

Important behavior: `TaskParser.parse()` calls `env.reset()` internally before
placing task objects. Backend code must not call `env.reset()` again after parsing
or the task-specific objects will be removed.

## GridRunner

Module: `gridworld/runner/grid_runner.py`

`GridRunner` executes episodes over any `AbstractGridBackend`.

```python
runner = GridRunner(backend=backend)
result = runner.run_episode(spec, policy_fn=policy, seed=spec.seed)
results = runner.run_batch([spec1, spec2], policy_fn=policy)
demo = runner.collect_demonstrations(spec, actions=[2, 2, 1, 2])
records = runner.generate_observation_dataset([spec], output_dir="observations")
```

`policy_fn` receives `(observation, state, mission)` and returns either an action
integer or `(action, info_dict)`.

`EpisodeResult` contains `task_id`, `success`, `total_reward`, `steps_taken`,
`max_steps`, `terminated`, `truncated`, `trajectory`, `final_state`, `seed`, and
`mission`.

## ModelInterface

Module: `model_interface.py`

Model adapters implement:

```python
class MyModel(ModelInterface):
    @property
    def model_name(self) -> str: ...

    def predict(self, input: ModelInput) -> ModelOutput: ...
```

`ModelInput` fields:

- `image: np.ndarray`
- `text_prompt: str`
- `action_space: dict[int, str]`
- `step_number: int`
- `max_steps: int`
- `additional_context: str | None`
- `prior_images: list[np.ndarray] | None`

`ModelOutput` fields:

- `action: int`
- `confidence: float | None`
- `reasoning: str | None`
- `raw_output: str | None`

Built-in adapters:

- `RandomModelInterface`
- `FileBasedModelInterface`
- `adapters.ollama_vlm_adapter.OllamaVLMAdapter`
- `adapters.lmstudio_vlm_adapter.LMStudioVLMAdapter`
- `adapters.paligemma_adapter.PaliGemmaMiniGridAdapter`

## EvaluationHarness

Module: `evaluation_harness.py`

`EvaluationHarness` adapts `ModelInterface` to `GridRunner` and computes aggregate
metrics.

```python
harness = EvaluationHarness(model, backend=backend)
episode = harness.evaluate_task(spec)
tier_metrics = harness.evaluate_tier(1, task_dir="gridworld/tasks")
all_metrics = harness.evaluate_all(task_dir="gridworld/tasks", tiers=[1, 2, 3])
benchmark = harness.evaluate_task_dir("mazes/validation_10")
harness.close()
```

The harness can include prior images and rolling text summaries in `ModelInput`
via `history_images`, `history_text`, and `history_text_window`.

## CLI

Primary CLI:

```bash
python run_eval.py --benchmark validation_10 --model random
python run_eval.py --benchmark tiers --tier 1-3 --model random
python run_eval.py --benchmark directory --task-dir path/to/json_dir --model random
```

Installed entry point:

```bash
multinet-run-eval --benchmark validation_10 --model random
```
