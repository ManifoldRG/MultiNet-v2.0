# MultiNet v2.0 — System Design (Canonical)

| Field | Value |
|---|---|
| Status | Draft for review |
| Date | 2026-05-09 |
| Branch | `reporting-and-examples` |
| Scope | Target architecture + as-is delta for the MultiNet v2.0 benchmark pipeline |
| Supersedes | Pipeline summary in `RUNME.md` §10; informal owner allocation in standup notes |
| Companion docs | `docs/README.md`, `docs/technical_design.md`, `docs/gridworld_backends.md`, `docs/task_parser.md`, `docs/minigrid_backend.md`, `docs/multigrid_backend.md` (all describe the system as it is today and remain authoritative for component internals) |

This document is the single canonical source of truth for how the MultiNet v2.0 evaluation pipeline is structured. It describes the target architecture; per-component implementation details continue to live in the companion docs above.

---

## Table of Contents

1. [Overview & north stars](#1-overview--north-stars)
2. [Pipeline DAG: stages, artifacts, invalidation](#2-pipeline-dag-stages-artifacts-invalidation)
3. [Task spec contract](#3-task-spec-contract)
4. [Static scoring (13 dimensions)](#4-static-scoring-13-dimensions)
5. [Runtime scoring](#5-runtime-scoring)
6. [Backend & inference adapter contracts](#6-backend--inference-adapter-contracts)
7. [Reporting & aggregate](#7-reporting--aggregate)
8. [As-is vs target delta](#8-as-is-vs-target-delta)
9. [Open questions & calibration targets](#9-open-questions--calibration-targets)

---

## 1. Overview & north stars

**MultiNet v2.0** evaluates a model's spatial-reasoning ability on grid-based puzzle tasks across multiple **spatial topologies** (square / hex / triangle / 3-4-6-4 / 4-8-8) and multiple **observation modalities** (RGB image / textual rendering). It produces a single comparable score per (model, task suite) pair — the "MultiNet score" — analogous to ARC-AGI as a headline metric, with a per-task vector report underneath for diagnostics.

### 1.1 Design north stars

1. **Single canonical source of truth for tasks.** `TaskSpecification` (integer cell coords, typed mechanism schema). Cross-domain transport uses it directly; normalized coords are computed by adapters on demand.
2. **Composable backend × inference matrix.** Two orthogonal axes; any spatial+modality backend pairs with any inference adapter. New backends and adapters are additions, not modifications. The pipeline runs the matrix.
3. **Static and runtime scoring are separate components.** Static scoring is a property of the *task*; runtime scoring is a property of a *(task, model, seed)* run. Static feeds runtime; runtime never feeds back into static.
4. **Reproducibility by content-hash invalidation.** Every artifact is keyed off a content hash of its inputs. Re-running the pipeline only recomputes stages whose inputs changed. Same spec + same seed + same code → bit-identical outputs.
5. **Live-benchmark contamination resistance.** Procedural generation is the production mode; the static task suite is for development and regression. Both modes share the same pipeline and produce the same artifact shapes.

### 1.2 Two-axis backend / inference decomposition

```
Spatial+Modality Backend             Inference Adapter
(implements AbstractGridBackend)     (talks to a model)
─────────────────────────────        ────────────────────────
MiniGridBackend  (square+RGB)        OllamaAdapter
MultiGridBackend (5 tilings+RGB)     LMStudioAdapter
TextBackend      (any+text) Pending  PaliGemmaAdapter
                                     FileBasedAdapter
                                     ClaudeAdapter   Pending
                                     RandomBaseline (built-in)
```

Each evaluation run is a 2-tuple `(backend, adapter)` plus a task. Same task across different backends = topology / modality generalization test. Same task and backend across different adapters = model comparison.

---

## 2. Pipeline DAG: stages, artifacts, invalidation

The pipeline is a five-stage DAG. Each stage has declared inputs and outputs and is keyed by a content hash so re-runs only touch stages whose inputs actually changed.

### 2.1 Stages

1. **Generate**
   - Inputs: `generator_params` (tier, seed, knobs).
   - Outputs: `task.json` (a `TaskSpecification`).
   - Hash key: `hash(generator_v, params)`.

2. **Solve & Score-static**
   - Inputs: `task.json`.
   - Outputs:
     - `canonical_paths.json` `{ bfs: { path, steps, states_explored }, greedy: { success, path, steps }, … }`
     - `scored.json` `{ is_beatable, dimensions[13], fragility, mechanism_necessity_violations, distractor_safety_violations, message }`
   - Hash key: `hash(solver_v, scorer_v, task.json, agent_set_v)`.
   - If `scored.json.is_beatable == false`, downstream stages skip the task; it is logged as ineligible and surfaced in reports.

3. **Render-and-Run**
   - Inputs: `task.json`, `scored.json` (gate on `is_beatable`), backend choice, adapter choice, `model_id`, `seed`.
   - Outputs: `run.json` `{ trajectory, actions, tokens, terminated, success }`.
   - Hash key: `hash(backend_v, adapter_v, model_id, task.json, seed)`.

4. **Score-runtime**
   - Inputs: `run.json`, `scored.json`, `canonical_paths.json`.
   - Outputs: `run_score.json` `{ success, step_ratio, cell_overlap_*, distractor_interactions, irreversible_failures, tokens, composite }`.
   - Hash key: `hash(runtime_scorer_v, inputs)`.

5. **Aggregate**
   - Inputs: many `run_score.json` files.
   - Outputs: `leaderboard.json`, `tier_breakdown.json`, `per_model/<id>.json`, optional `comparisons.json`.
   - Hash key: `hash(aggregator_v, sorted_input_hashes)`.

Stages 1–2 are per-task and produce the **task artifact bundle**. Stage 3 is per `(task × backend × adapter × seed)`. Stage 4 is 1:1 with stage 3. Stage 5 fans in.

### 2.2 Artifact layout

```
artifacts/
├── tasks/<task_id>/
│   ├── task.json                # Stage 1
│   ├── canonical_paths.json     # Stage 2 (a)
│   └── scored.json              # Stage 2 (b) — includes is_beatable
├── runs/<task_id>/<backend>/<adapter>/<model_id>/<seed>/
│   ├── run.json                 # Stage 3
│   └── run_score.json           # Stage 4
└── reports/<run_set_id>/
    ├── leaderboard.json         # Stage 5
    ├── tier_breakdown.json
    ├── per_model/<id>.json
    └── comparisons.json         # optional, when run set covers the relevant matrix
```

Every artifact carries `inputs_hash` and `producer_version` in its header. The DAG runner reads those to detect staleness.

### 2.3 Invalidation rules

- Spec change (new `task.json`) → invalidates `canonical_paths`, `scored`, all `run`s and `run_score`s for that task.
- Solver / scorer version bump (combined stage 2) → invalidates `canonical_paths` + `scored` + all downstream.
- Backend or adapter version bump → invalidates only `run` / `run_score` rows that used that version.
- Runtime scorer change → invalidates `run_score` and `aggregate`.
- Calibration version bump → invalidates `run_score` and `aggregate` only; cached `run.json` is reused (model calls are not re-executed).

### 2.4 DAG runner

Deferred to implementation. FOSS preferred; **Snakemake** is the leading candidate.

---

## 3. Task spec contract

`TaskSpecification` (in `gridworld/task_spec.py`) is the canonical SoT. Every component in the pipeline reads or produces it. Its serialization is JSON; its in-memory form is a typed dataclass.

### 3.1 Schema overview

```
TaskSpecification
├── task_id : str               # unique identifier
├── version : str               # schema version (currently "1.0")
├── seed : int                  # deterministic-replay seed
├── difficulty_tier : int       # positive integer; coarse organizing label only
├── description : str           # optional human-readable text
├── max_steps : int             # episode budget
│
├── maze : MazeLayout
│   ├── dimensions : (int, int)         # (width, height); cell coords integer
│   ├── walls : list[Position]
│   ├── start : Position
│   ├── goal : Position
│   └── floor : Optional[list[Position]]
│
├── mechanisms : MechanismSet
│   ├── keys[]         (id, position, color)
│   ├── doors[]        (id, position, requires_key, initial_state ∈ {locked, open})
│   ├── switches[]     (id, position, controls[gate_ids], color, switch_type ∈ {toggle, hold, one_shot}, initial_state)
│   ├── gates[]        (id, position, initial_state ∈ {open, closed})
│   ├── blocks[]       (id, position, pushable, color)
│   ├── teleporters[]  (id, position_a, position_b, bidirectional)
│   └── hazards[]      (id, position, hazard_type ∈ {lava, pit, spike})
│
├── rules : Rules
│   ├── key_consumption : bool
│   ├── switch_type : default switch behavior
│   ├── hidden_mechanisms : list[mechanism_id]
│   ├── observability : {full, view_cone, fog_of_war}
│   └── view_size : odd int ≥ 3 (for non-full observability)
│
├── goal : GoalSpec
│   └── one of: reach_position(target) | collect_all(target_ids[]) | push_block_to(target_ids[], target_positions[]) | survive_steps
│
├── dependency_chain : Optional[DependencyChain]   # ordered solve sequence
├── distractors : Optional[list[Distractor]]       # type-weighted
└── metadata : Optional[dict]                       # opaque, never consumed by pipeline
```

### 3.2 Coordinate convention

- `Position(x, y)` is integer cell index. Origin is `(0, 0)` at top-left.
- Border cells (`x ∈ {0, width-1}`, `y ∈ {0, height-1}`) are implicitly walls; `walls[]` lists *interior* walls only.
- Agent direction is integer `0=right, 1=down, 2=left, 3=up` (MiniGrid convention; the validator and backends both depend on it).

### 3.3 Invariants

Enforced by `TaskSpecification.validate()`:

1. Dimensions ≥ 3×3.
2. All positions in bounds (where applicable; doors and gates may sit on walls).
3. `start` and `goal` are not walls.
4. All mechanism IDs are unique across the whole `MechanismSet`.
5. No two mechanisms share a position (except teleporter pair endpoints).
6. Every door's `requires_key` color has at least one matching key.
7. Every switch's `controls[]` references existing gate IDs.
8. `hidden_mechanisms` references existing IDs.
9. `dependency_chain.depth == len(sequence)` and step numbering is `1..N`.
10. Distractors reference existing mechanism IDs (except `distractor_chain` type, which is mechanism-set-wide).
11. Goal-type-specific consistency (e.g. `push_block_to` has equal-length `target_ids` and `target_positions`).
12. `view_size` is odd and ≥ 3 when observability ≠ full.

### 3.4 Versioning

- `version` field on every JSON file; current value `"1.0"`.
- Bumps follow semver: minor bump for backwards-compatible field additions, major bump for breaking changes.
- The pipeline's `producer_version` headers on artifacts record both schema version and code version.

### 3.5 `CanonicalTaskSpec` status

`cross_domain/canonical_task_spec.py` is **frozen as a deprecated transport-layer view** until cross-domain (physics, GUI) returns to scope. It currently has only one consumer (`cross_domain/gridworld_adapter.py` translation methods + one test).

- Mark `CanonicalTaskSpec` and `cross_domain/` as frozen in module docstrings.
- Do not add new consumers.
- When cross-domain returns, decide then whether to revive (with normalized-coord adapters) or replace.

---

## 4. Static scoring (13 dimensions)

Static scoring runs once per task at pipeline stage 2 (Solve & Score-static). It produces `scored.json`, which carries `is_beatable` plus a 13-dimension vector and supporting validation reports. The static scorer consumes `task.json` and `canonical_paths.json`.

### 4.1 Dimensions

All raw values are floats (or counts cast to float). Higher = harder *unless* explicitly marked **penalty** (in which case higher = easier-to-shortcut, which lowers the runtime composite).

1. **`optimal_path_length`** — Source: BFS canonical agent. Computation: step count of `bfs.optimal_path`.
2. **`search_space_size`** — Source: BFS canonical agent. Computation: states explored during BFS.
3. **`backtracking_required`** — Source: BFS canonical agent. Computation: revisited cells along optimal path.
4. **`fragility`** — Source: bounded BFS over irreversible transitions. Computation: `1 / min_steps_to_break` (0 if unbreakable).
5. **`dependency_depth`** — Source: spec or solver. Computation: `dependency_chain.depth` if present, otherwise inferred from mechanism interactions.
6. **`dependency_variety`** — Source: spec. Computation: distinct mechanism categories used (keys+doors, switches+gates, blocks, teleporters, hazards).
7. **`distractor_count`** — Source: spec. Computation: `len(distractors)`.
8. **`distractor_quality`** — Source: spec, type-weighted. Computation: sum of `weights[type]` per distractor. Vibe-based; **calibration target**.
9. **`grid_size`** — Source: spec. Computation: `width × height`.
10. **`wall_density`** — Source: spec. Computation: `len(walls) / grid_size`. Crude (does not separate interior vs functional walls); **calibration target**.
11. **`partial_observability`** — Source: spec rules. Computation: ordinal `{full: 0, view_cone: 1, fog_of_war: 2}` from `rules.observability`.
12. **`irreversibility`** — Source: spec rules + mechanisms. Computation: `key_consumption × #doors + #one_shot_switches + #non_bidirectional_teleporters`.
13. **`greedy_solvability`** — Source: Greedy canonical agent. Computation: `1.0 if greedy succeeds else 0.0`. **Penalty** (greedy-solvable tasks lower the runtime composite, on the rationale that they are less a test of spatial reasoning).

### 4.2 Static composite (difficulty score)

```
static_composite = Σ_i (raw_dim_i × calibration.weights[dim_name_i])
```

- `calibration.weights` lives in `calibration.yaml`; defaults to `1.0` for all dimensions until empirical tuning.
- `static_composite` is used for task ranking and live-benchmark filtering (e.g., reject tasks whose composite falls outside a tier's target range).
- It is *not* used directly in runtime scoring; runtime uses individual dimensions plus a derived "difficulty weight" (Section 5).

### 4.3 Validation reports (also in `scored.json`)

Beyond the dimension vector, `scored.json` carries the validator's structural reports:

- `is_beatable` (bool) and `message` (str) — gate for downstream stages.
- `mechanism_necessity_violations` (list of strings) — mechanisms whose removal still leaves the task solvable; flags accidental decoration.
- `distractor_safety_violations` (list of strings) — distractors that can render the task unsolvable; flags unsafe distractors.
- `chain_ordering_valid` (bool) — each dependency step actually gates the next.

These do not enter the composite but are surfaced in reports for task-quality auditing.

### 4.4 Calibration notes

- `distractor_quality` and `wall_density` are explicitly flagged as crude / vibe-based; calibration targets, not committed-correct.
- `fragility` = `1 / min_steps_to_break` is the existing implementation; document the tradeoff (sensitive to tiny changes near the boundary) as a calibration concern.
- `partial_observability` ordinal `{0, 1, 2}` is a placeholder; could be replaced with a measured visibility ratio if calibration shows the ordinal is too coarse.
- `greedy_solvability` as a 0/1 signal is the simplest form; the design leaves room to extend to a vector across additional canonical agents (random, heuristic, …) once the solver suite grows.

---

## 5. Runtime scoring

Runtime scoring runs at pipeline stage 4 (Score-runtime), once per `run.json`. It produces `run_score.json`. It consumes the run trajectory plus the static scoring artifacts (`scored.json`, `canonical_paths.json`).

### 5.1 Per-run signal vector

Recorded for every `(task, backend, adapter, model_id, seed)`:

- `success` (bool) — goal reached within `max_steps`, no terminal hazard.
- `steps` (int) — agent's actual step count.
- `terminated_reason` (str) — one of `{goal_reached, hazard, max_steps, deadlock, invalid_action_excess}`.
- `token_count` (int) — total prompt + response tokens summed over all model turns.
- `distractor_interactions` (int) — count of distractor-element interactions (any `pickup` / `toggle` / `push` on an element registered as a distractor).
- `irreversible_failures` (int) — count of irreversible actions that broke solvability, detected by re-running the validator from the post-action state.

### 5.2 Path-comparison signals (vs `canonical_paths.json`)

- `step_ratio` (float ∈ [0, 1]) — `bfs.optimal_steps / max(model_steps, bfs.optimal_steps)`. 1.0 = matched optimal; 0.5 = took 2× as long; 0 if model failed.
- `cell_overlap_bfs` (float ∈ [0, 1]) — `|model_cells ∩ bfs_optimal_cells| / |bfs_optimal_cells|`.
- `cell_overlap_greedy` (float ∈ [0, 1]) — same metric vs greedy path. Diagnostic only; not in composite by default.

### 5.3 Composite shape

```
composite = success_factor × efficiency_factor × difficulty_weight − greedy_penalty
```

- `success_factor = 1.0 if success else 0.0` — hard gate; failed runs score 0 regardless of efficiency.
- `efficiency_factor = α × step_ratio + β × cell_overlap_bfs + γ × token_efficiency` — weighted blend; default `α = β = γ = 1/3`. `token_efficiency = min(1, baseline_tokens / max(model_tokens, 1))` where `baseline_tokens` lives in `calibration.yaml`.
- `difficulty_weight = normalize(static_composite)` — harder tasks contribute more. Default normalization: `f(x) = x / max_observed_static_composite_in_suite`.
- `greedy_penalty = δ × greedy_solvability × success_factor` — applied only to successful runs; `δ` is a calibration coefficient with default 0.5.

All Greek-letter coefficients (`α, β, γ, δ`) and the normalization function live in `calibration.yaml`. The design commits to the *shape*, not the values.

### 5.4 Single-point benchmark score (ARC-AGI style)

For a task suite `T` and a model:

```
multinet_score = (1 / |T|) × Σ_{t ∈ T} composite(t, model)
```

Defaults to a uniform mean. Calibration may switch to a tier-weighted or difficulty-weighted aggregation later. The headline number is what gets reported on a leaderboard; per-task vectors stay underneath for diagnostics.

### 5.5 `run_score.json` shape

```
{
  "task_id": ..., "backend": ..., "adapter": ..., "model_id": ..., "seed": ...,
  "signals": {
    "success": ...,
    "steps": ...,
    "terminated_reason": ...,
    "token_count": ...,
    "distractor_interactions": ...,
    "irreversible_failures": ...,
    "step_ratio": ...,
    "cell_overlap_bfs": ...,
    "cell_overlap_greedy": ...
  },
  "composite": ...,
  "calibration_version": ...,
  "inputs_hash": ...,
  "producer_version": ...
}
```

### 5.6 Calibration notes

- All composite coefficients ship as `1.0` or sensible defaults; the design does not claim correctness.
- `calibration.yaml` is versioned in git; changes bump `calibration_version` and trigger stage-4 / stage-5 invalidation.
- After a calibration update, the pipeline regenerates `run_score.json` and `reports/` from cached `run.json`. Run records do **not** re-execute model calls. This is a deliberate consequence of the DAG split.

---

## 6. Backend & inference adapter contracts

### 6.1 Backend contract: `AbstractGridBackend`

Defined in `gridworld/backends/base.py`. Every spatial+modality backend implements this interface.

```
class AbstractGridBackend(ABC):
    def configure(spec: TaskSpecification) -> None
    def reset(seed: Optional[int]) -> tuple[Observation, GridState, dict]
    def step(action: Action) -> tuple[Observation, float, bool, bool, GridState, dict]
    def render() -> Observation
    def get_mission_text() -> str
    def get_state() -> GridState
    def close() -> None

    @property
    def action_space() -> ActionSpace
    @property
    def observation_modality() -> Literal["rgb", "text", "rgb+text"]
    @property
    def supported_tilings() -> set[str]
```

### 6.2 Schema generalizations from current code

Three changes from the current `AbstractGridBackend`:

1. `Observation` is now `Union[np.ndarray, str, dict]` (was `np.ndarray` only). RGB backends return arrays; text backend returns strings; future hybrid backends can return `{"rgb": ..., "text": ...}`. Adapters introspect via `observation_modality`.
2. `Action` is now `Union[int, str]` (was `int`). Discrete backends accept integer action IDs; the text backend accepts NL command strings. The backend's `action_space` describes which.
3. `ActionSpace` is a new typed object: `{kind: Literal["discrete", "text"], size: Optional[int], names: list[str], grammar_hint: Optional[str]}`. Adapters use it to format prompts and validate actions.

### 6.3 Concrete backends

- **`MiniGridBackend`** — square grid only; `observation_modality = "rgb"`; `action_space.kind = "discrete"`, size 7, names per MiniGrid (turn_left / turn_right / forward / pickup / drop / toggle / done). Wraps the gymnasium `minigrid` package.
- **`MultiGridBackend`** — `supported_tilings = {square, hex, triangle, 3-4-6-4, 4-8-8}`; `observation_modality = "rgb"`; `action_space.kind = "discrete"`, size 9, names per MultiGrid (forward / backward / turn_left / turn_right / pickup / drop / toggle / push / wait). Translates from 7-action MiniGrid for compatibility.
- **`TextBackend`** *(pending merge)* — any tiling; `observation_modality = "text"`; `action_space.kind = "text"`, with `grammar_hint` describing accepted commands ("go forward", "turn left", "pickup", …). Renders the maze as text.

### 6.4 Inference adapter contract: `ModelInterface`

Defined in `model_interface.py`. Every adapter implements:

```
class ModelInterface(ABC):
    def predict(
        observation: Observation,
        mission_text: str,
        action_space: ActionSpace,
        history: list[Turn]              # prior (obs, action) pairs, may be empty
    ) -> tuple[Action, ModelMetadata]   # action + {tokens, raw_response, latency_ms}

    @property
    def name() -> str

    @property
    def supported_modalities() -> set[Literal["rgb", "text", "rgb+text"]]
```

`supported_modalities` is checked at composition time; mismatches (`TextBackend` + RGB-only adapter) are rejected before the run starts.

### 6.5 Concrete adapters

- **`RandomBaseline`** — built-in; uniform sample from `action_space`. Sanity check.
- **`OllamaAdapter`** — local Ollama HTTP server; supports RGB+text via vision models.
- **`LMStudioAdapter`** — local LM Studio HTTP server.
- **`PaliGemmaAdapter`** — direct HuggingFace transformers; RGB-only.
- **`FileBasedAdapter`** (`FileBasedModelInterface`) — writes observation to a work-dir, polls for response file. Supports any model integrated via external process.
- **`ClaudeAdapter`** *(pending)* — Anthropic API; supports RGB+text.

### 6.6 Eval loop (pipeline stage 3 internals)

```
backend.configure(task_spec)
obs, state, info = backend.reset(seed=seed)
history = []
trajectory = [{"step": 0, "obs": obs, "state": state.to_dict()}]

while not (state.terminated or state.truncated):
    action, meta = adapter.predict(
        observation=obs,
        mission_text=backend.get_mission_text(),
        action_space=backend.action_space,
        history=history,
    )
    prev_obs = obs
    obs, reward, terminated, truncated, state, info = backend.step(action)
    history.append({"obs": prev_obs, "action": action})
    trajectory.append({
        "step": state.step_count,
        "action": action,
        "obs": obs,
        "state": state.to_dict(),
        "tokens": meta.tokens,
    })

write run.json
```

### 6.7 `run.json` shape

```
{
  "task_id": ..., "backend": ..., "adapter": ..., "model_id": ..., "seed": ...,
  "trajectory": [...],          # list of step dicts (action, obs-summary, state, tokens)
  "total_tokens": ...,
  "success": ...,
  "terminated": ..., "truncated": ...,
  "terminated_reason": ...,
  "wall_clock_seconds": ...,
  "inputs_hash": ..., "producer_version": ...
}
```

`obs` is summarized (RGB stored as a hash or thumbnail path; text stored as a hash or first-N chars) to keep `run.json` small. Full observations are kept under `runs/<...>/obs/` if needed for debugging.

---

## 7. Reporting & aggregate

Stage 5 fans many `run_score.json` files into headline reports. Aggregation is purely math over the run-score vectors; no model calls, no backend interaction.

### 7.1 Run-set concept

A **run set** is a labeled collection of runs to aggregate together. Examples:

- `daily-2026-05-08` — all runs produced today.
- `release-v0.1-suite` — fixed canonical runs for a release.
- `tier3-rgb-vs-text` — comparative subset for a specific question.

The DAG runner produces one `reports/<run_set_id>/` directory per run set. Run-set membership is declarative (a glob or filter over `runs/`).

### 7.2 Output artifacts

**`reports/<run_set_id>/leaderboard.json`** — one row per model, sorted by single-point benchmark score:

```
{
  "run_set_id": ...,
  "calibration_version": ...,
  "rows": [
    {
      "model_id": ...,
      "multinet_score": ...,            # uniform mean of composite over the run set
      "task_count": ...,
      "success_rate": ...,              # mean of success_factor
      "step_efficiency": ...,           # mean of step_ratio over successful runs
      "path_overlap": ...,              # mean of cell_overlap_bfs over successful runs
      "token_efficiency": ...,
      "by_tier": { "1": {...}, "2": {...}, ... },
      "by_backend": { "minigrid": {...}, "multigrid_hex": {...}, "text": {...} }
    },
    ...
  ]
}
```

**`reports/<run_set_id>/tier_breakdown.json`** — per-tier mean composite, mean success rate, count of tasks. Used to surface where models break down.

**`reports/<run_set_id>/per_model/<model_id>.json`** — full per-task table for one model (every `run_score.json` for that model in the run set, plus aggregates). For deep-dive analysis.

### 7.3 Cross-axis comparisons (optional)

The aggregator computes generalization-gap metrics when the run set covers the relevant matrix:

- **Topology gap**: `score(model, MultiGrid_square) − score(model, MultiGrid_hex)` (square-grid overfitting).
- **Modality gap**: `score(model, MiniGrid) − score(model, Text)` (RGB vs text reliance).
- **Tier slope**: regression of composite over tier (gentle slope = robust; cliff = brittle).

These live in `reports/<run_set_id>/comparisons.json` if computable, omitted otherwise.

### 7.4 What's *not* in scope for this design doc

- Visualization (HTML dashboards, plots) — downstream tools consume the JSONs.
- Public-leaderboard hosting — orthogonal concern.
- Statistical significance testing — reports surface raw numbers; significance testing is a research-side analysis on top of the per-task vectors.

---

## 8. As-is vs target delta

What exists in `reporting-and-examples` today against the target design.

Status legend:
- ✅ matches target as-is
- ⚠️ partial / requires modification
- 🚧 pending (not yet merged or not yet started)

### 8.1 Component-by-component

**1. Procedural maze generator** — Stage 1
- 🚧 Pending merge. Emits `TaskSpecification` JSON.
- Delta: schema-conformance check on merge.

**2. Validator** — folded into Stage 2
- ✅ `gridworld/task_validator.py::TaskValidator` does exhaustive BFS over the full mechanism state space, plus `compute_fragility`, `validate_mechanism_necessity`, `validate_chain_ordering`, `validate_distractor_safety`.
- Delta: surface validation reports into `scored.json` instead of emitting a separate `validity.json`.

**3. Solver suite (canonical agents)** — Stage 2
- ⚠️ BFS exists inside `TaskValidator._find_solution`. Greedy does not yet exist as a separate canonical agent.
- 🚧 Multi-tier solver suite pending; Greedy is the next addition, then heuristic, then random.
- Delta: extract BFS path emission as one canonical agent, add Greedy as a peer, write combined output to `canonical_paths.json`.

**4. Static scorer** — Stage 2
- ⚠️ `gridworld/scoring.py::compute_12d_score` exists with 12 dimensions matching dimensions 1–12 of §4 (modulo formula calibration).
- Delta: add dimension 13 (`greedy_solvability`), restructure output to `scored.json` sidecar, move composite weights to `calibration.yaml`, include validation reports.

**5. `MiniGridBackend`** — backend axis
- ✅ `gridworld/backends/minigrid_backend.py` implements `AbstractGridBackend` for square grids with discrete actions + RGB rendering.
- Delta: adopt new `ActionSpace` typed object and `observation_modality` property (§6.2 schema generalization).

**6. `MultiGridBackend`** — backend axis
- ✅ `gridworld/backends/multigrid_backend.py` + `multigrid/` package supports square / hex / triangle / 3-4-6-4 / 4-8-8.
- Delta: same `ActionSpace` / `observation_modality` adoption as `MiniGridBackend`.

**7. `TextBackend`** — backend axis
- ⚠️ `nl_domain/nl_env.py` is a wrapper around `MiniGridBackend`, not a peer backend. Placeholder only.
- 🚧 Full `AbstractGridBackend` implementation pending.
- Delta on merge: deprecate the wrapper; the new `TextBackend` takes its place. NL-as-action-mode collapses into the backend's text `ActionSpace`.

**8. Inference adapters** — adapter axis
- ✅ `OllamaAdapter`, `LMStudioAdapter`, `PaliGemmaAdapter`, `FileBasedModelInterface`, `RandomModelInterface` exist in `model_interface.py` + `adapters/`.
- 🚧 `ClaudeAdapter` pending.
- Delta: extend `predict()` signature to accept the new `ActionSpace` and to return a `ModelMetadata` object (tokens, latency, raw_response).

**9. Evaluation harness** — Stage 3
- ⚠️ `evaluation_harness.py` + `run_eval.py` exist; the loop is roughly the right shape but produces ad-hoc result dicts, not pipeline artifacts.
- Delta: emit canonical `run.json`; remove inline scoring (move to Stage 4); add per-step trajectory recording.

**10. Runtime scorer** — Stage 4
- 🚧 Does not exist as a component. Some scoring logic lives inside `evaluation_harness.py`.
- Delta: new module that consumes `run.json` + `scored.json` + `canonical_paths.json` and produces `run_score.json`.

**11. Aggregator / reporter** — Stage 5
- ⚠️ Partial. `evaluation_harness.py` produces some summary dicts; nothing matches the per-run-set artifact layout.
- Delta: new module emitting `leaderboard.json`, `tier_breakdown.json`, `per_model/<id>.json`, `comparisons.json`.

**12. Pipeline DAG runner**
- 🚧 Does not exist.
- Delta: new component; FOSS preferred (Snakemake leading candidate); content-hash invalidation.

**13. `cross_domain/` package**
- ⚠️ `CanonicalTaskSpec` exists but has effectively one consumer (`gridworld_adapter.py` translation methods).
- Delta: mark module **frozen** in docstrings; do not extend until cross-domain returns to scope.

### 8.2 Summary

The pipeline is **roughly 60% present in tree** — backends, validator, 12 of 13 static dimensions, and four inference adapters are all there. The structural work is mostly **consolidation and contracts**: emitting canonical artifacts, separating Solve/Score from validate-and-everything-else, extracting runtime scoring into its own component, and adding the DAG runner.

The most net-new components are the **DAG runner**, **runtime scorer**, **aggregator**, and the **`ClaudeAdapter`** + **`TextBackend`** that close the pending-merge gaps.

---

## 9. Open questions & calibration targets

Items the design intentionally defers. None block initial implementation.

### 9.1 Implementation choices
- DAG runner technology — Snakemake leading candidate; final pick deferred to implementation.
- Token-efficiency baseline (`baseline_tokens`) — per-task vs global constant; needs a sensible default once a few model runs exist.

### 9.2 Calibration coefficients (live in `calibration.yaml`, default to placeholders)
- Runtime composite blend weights `α, β, γ` (step ratio / cell overlap / token efficiency).
- Greedy penalty coefficient `δ`.
- `difficulty_weight` normalization function (currently `x / max_observed`; may switch to a percentile or log normalization).
- Static composite per-dimension weights (currently all 1.0).
- Aggregation weighting for `multinet_score` (currently uniform mean; may become tier- or difficulty-weighted).

### 9.3 Dimension fidelity (calibration targets)
- `distractor_quality` — type-weighted is vibe-based; possible empirical upgrade (baseline-VLM distractor-interaction rate).
- `wall_density` — interior vs functional wall accounting; navigability-aware metric.
- `fragility` — `1 / min_steps_to_break` may be sensitive near boundaries; consider smoother alternatives.
- `partial_observability` — ordinal `{0, 1, 2}` could become a measured visibility ratio.

### 9.4 Future extensions (out of scope for v2.0; design accommodates)
- Multi-canonical-agent vector — replace `greedy_solvability` (single 0/1) with `solvability_by[agent]` once heuristic / random / A* solvers land.
- Cross-domain (physics, GUI) — `cross_domain/` is frozen; revival is a v3 concern.
- Live-benchmark task lifecycle — retirement policy, contamination detection, production task-pool governance.
- Generalization-gap thresholds — empirical thresholds for "topology overfitting" / "modality overfitting" need real eval data.

---

## Appendix A — Original component diagram (for reference)

The original kickoff sketch is preserved here for traceability against this design:

```
JSON generator → Outputs: json files/strings
    ↓
Task spec / Validator / BFS-greedy agents
    Outputs: maze validity (bool), canonical agent path, score calculation
    ↓
Backend Generator (Gridworld / Multigrid / Text)
    Outputs: AbstractGridBackend-derived backends
    ↓
Inference scripts (ollama, lmstudio, claude api, ...)
    Outputs: success (bool), steps (int), token count (int), other metrics
    ↓
Scoring code
    Outputs: final score, comparison with agent paths
```

Mapping to the canonical pipeline:

| Original term | Canonical term in this doc | Section |
|---|---|---|
| JSON generator | Stage 1 (Generate) | §2.1 |
| Task spec / Validator | folded into Stage 2 (Solve & Score-static) | §2.1 |
| BFS-greedy agents | Multi-tier canonical agent suite (Stage 2) | §2.1, §4 |
| Score calculation (static) | Static scoring (13 dimensions) (Stage 2) | §4 |
| Backend Generator | Backend axis: `MiniGridBackend` / `MultiGridBackend` / `TextBackend` | §6 |
| Inference scripts | Adapter axis: `ModelInterface` implementations | §6 |
| Scoring code (final score, comparison) | Runtime scoring (Stage 4) + Aggregate (Stage 5) | §5, §7 |

The original twelve-dimension list maps 1:1 onto §4 dimensions 1–12; **dimension 13 (`greedy_solvability`)** is added based on the multi-tier solver decision (see §4.1 and §8 row 3).
