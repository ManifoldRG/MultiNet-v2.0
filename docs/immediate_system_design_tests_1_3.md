# MultiNet v2.0 Immediate System Design for Tests 1-3

| Field | Value |
|---|---|
| Status | Draft for merge planning |
| Date | 2026-05-17 |
| Branch basis | `origin/reporting-and-examples` |
| Scope | Minimal system needed to run Experimental Evaluation Draft tests 1, 2, and 3 |
| Companion target design | `docs/system_design.md` (read first for canonical architecture) |

This document narrows the canonical target design to the system needed for the
current merge stack and the first scoring experiments. The canonical document
remains the long-term architecture; this is a merge and execution plan for the
immediate slice.

The source experiment plan is
`/home/sean/AI_training/Manifold/MultiNet/docs/plans/EXPERIMENTAL_EVALUATION_DRAFT.md`.
Tests 1-3 are:

1. Scoring calibration over the existing task set.
2. Complexity-vs-distance validation with short mechanistic and long open paths.
3. Mechanism-ordering consistency checks over matched task pairs.

## 1. Executive Summary

The tests 1-3 pipeline is roughly 80-90% present across merged foundation work
and open PRs. The remaining work is integration shape, not research code:
emitting standard artifacts, splitting the curated fixture generator, cleaning
the NLP backend and baseline-agent branches, and adding thin aggregation
reports.

**Status at a glance:**

- Foundation (task schema, validator, BFS, static scoring): merged to `main`.
- Runtime backend, model adapters, reporting/docs scaffolding: in PRs #3, #4,
  #5 — awaiting land.
- Baselines, NLP backend, maze generator: in PRs #10, #8, #1 — need cleanup
  before they fit the standard.

**Remaining effort:** <1 week of scoring/integration work, plus 2-3 days per
API endpoint across five adapters — about 2-3 weeks of active engineering once
the merges land.

**Headline:** once the next push of merges (#3, #4, #5) lands, we'll be ready
to start the final buildout for tests 1-3.

**Decisions needing leadership input** (see §7):

1. Whether `greedy_solvability` is a separate canonical-agent feature or part
   of the calibration vector.
2. Whether PR #10's generated validation result JSON files are committed.
3. The implementation shape of the NLP backend.
4. Whether OGBench is part of the current merge basis.

**Merge order:** #3 → #4 → #5, then revise #8, then rebase #10. Details in §6.

## 2. Goal and Non-goals

### 2.1 Immediate Goal

The immediate system must answer one question:

> Can we produce reliable task, solver, run, and report artifacts for the first
> three scoring experiments without merging the full target DAG?

The merge slice therefore needs:

- A canonical `TaskSpecification` for every fixture.
- A BFS validator/solver that emits correct optimal plans.
- Static scoring rows for the draft's 12 dimensions.
- Optional canonical-agent features such as greedy solvability, kept separate
  until calibration decides whether they belong in the point vector.
- A MiniGrid runtime whose termination/reward semantics match the validator.
- Baseline/model episode rows with success, steps, optimality, path choice,
  mechanism interaction order, and failure point.
- JSON/CSV summaries simple enough to inspect and stable enough to feed
  notebooks or later report scripts.

### 2.2 Explicit Non-goals

These remain part of the canonical target design but should not block the
current merge slice:

- A full content-hash DAG runner.
- Production procedural generation for live benchmark tasks.
- MultiGrid parity across exotic tilings.
- Public leaderboard hosting or dashboard UI.
- Final runtime composite scoring.
- Final static point weights and tier thresholds.
- Cross-domain physics/GUI revival.

The NLP/text backend is a desired backend axis, but the current open PR needs
revision before it should join this merge basis.

## 3. Minimal Pipeline

The canonical design has five DAG stages. For tests 1-3, use the same concepts
but implement them as simple, inspectable artifact steps. Full artifact field
schemas are in Appendix A.

### 3.1 Fixture Selection and Generation

Inputs:

- Existing task set for test 1.
- Curated shortcut-maze variants for test 2.
- Curated mechanism-ordering pairs for test 3.

Outputs:

- `task.json` files in the existing `TaskSpecification` schema.
- A manifest with `task_id`, `experiment`, `condition`, `variant`, `source`,
  `expected_mechanisms`, and `notes`.

Required behavior:

- Every fixture must pass `TaskSpecification.validate()`.
- Fixture IDs encode the experiment and variant, for example
  `T2_short_mech_open`, `T2_long_open_only`, `T3_A1_key_switch`.
- Test 2 must make path choice measurable when both routes are open.
- Test 3 must hold path length and layout topology constant within each pair.

### 3.2 Static Solve and Score

Inputs: `task.json`.

Outputs: `canonical_paths.json` (BFS trace) and `scored_static.json` (12
calibration dimensions, beatable flag, validation messages, optional
canonical-agent features). See Appendix A for field shapes.

Required behavior:

- The BFS path must be replayable in the runtime backend.
- Switch activation semantics must match between validator and runtime. The
  validator activates switches from the agent's current cell, so the runtime
  must check the agent cell for switches.
- Non-`reach_position` goals must terminate in runtime when complete and
  return a positive reward so downstream success detection works.

The draft uses 12 scoring dimensions. The canonical target design
adds `greedy_solvability` as a 13th dimension. For the immediate experiments,
keep the 12 draft dimensions as the calibration vector and record greedy/random
baseline outcomes as separate canonical-agent features. After test 1, decide
whether `greedy_solvability` becomes a calibrated dimension.

### 3.3 Runtime Runs

Inputs: `task.json`, `canonical_paths.json`, backend choice, adapter or
baseline agent, seed.

Default immediate backend: `MiniGridBackend`.

Default immediate agents/adapters:

- BFS canonical replay.
- Greedy baseline, once PR #10 is cleaned up.
- Random baseline, once PR #10 is cleaned up.
- Ollama/LM Studio VLM adapters from PR #4 for model runs.

Outputs: `episode_runs.jsonl`, one row per `(task, backend, agent_or_model,
seed)`. See Appendix A for required row fields.

`path_choice` is required for test 2. `mechanism_interaction_order` and
`failure_point` are required for test 3.

### 3.4 Runtime Metrics and Reports

Inputs: `scored_static.json`, `canonical_paths.json`, `episode_runs.jsonl`.

Outputs:

- `scoring_calibration_summary.json` for test 1.
- `complexity_distance_summary.json` for test 2.
- `mechanism_ordering_pairs.json` for test 3.
- Optional CSV mirrors for notebooks.

Required report fields:

- Success rate by task, condition, and agent/model.
- Mean and median optimality ratio.
- Path choice counts for test 2.
- Paired success deltas for test 3.
- Scoring dimension correlation matrix for test 1.
- Draft 12-dimension point-weight candidates and tier-boundary candidates.

These reports do not claim a final MultiNet score. They are calibration
evidence used to update the scoring system.

## 4. Readiness Assessment

The tests 1-3 pipeline is roughly 80-90% present in tree and open PRs.

**Binary readiness:** not ready. The pipeline cannot run end-to-end today
because the runtime, adapter, and reporting layers (PRs #3, #4, #5) have not
landed.

**Effort to ready, post-merge:** <1 week of scoring/integration work, plus
2-3 days per API endpoint across five adapters — about 2-3 weeks of active
engineering once the basis merges land.

**Blocker:** review and merge cadence on the basis stack (#3 → #4 → #5). Once
those merge, the remaining work is integration shape rather than new research.

**What is already close:**

- Task schema, validation, BFS solving, and static scoring are largely in
  `main` via the merged foundation work.
- MiniGrid runtime and backend support are largely in #3.
- Model adapters and run harness concepts are largely in #4.
- Documentation, examples, and reporting scaffolding are largely in #5.
- NLP/text backend and maze-generation ideas exist in #8/#1, though they need
  reshaping before joining the standard pipeline.
- Baseline-agent work exists in #10 and should become the canonical-agent
  suite extension after cleanup.

**Remaining 10-20% (integration shape, not missing research code):**

- Emit standard artifacts or rows for `canonical_paths`, `scored_static`, and
  `episode_runs`.
- Build or split the curated fixture generator for tests 2-3.
- Clean and rebase #8 so text/NLP uses the same task/backend/adapter contracts.
- Clean and rebase #10 so BFS/greedy/random baselines are code, not generated
  result dumps.
- Generate the three experiment summaries with `multinet-aggregate-scores`
  after fixture runs are available.

**Scope assumption:** the immediate scope stays MiniGrid-first and does not
require full MultiGrid parity, public dashboards, final runtime composite
scoring, or the full DAG runner before tests 1-3.

Per-test readiness detail is in Appendix B.

## 5. PR Status — Consolidated

GitHub PR state is the source of truth for ownership and assignment. The
tables below consolidate component role, current GitHub state, landing effort,
and must-fix items.

### 5.1 Basis PRs (critical path for tests 1-3)

| PR | Branch | State | Role | Effort | Must-fix |
|---|---|---|---|---|---|
| #3 | `maze-realization-and-backends` | `CHANGES_REQUESTED` | MiniGrid runtime, MultiGrid code, backend abstractions, task parser | Medium | Confirm MiniGrid switch/goal-completion fixes in `2b82e81`; resolve duplicate-files/scope concern; clarify MultiGrid is not critical path for tests 1-3; rerun focused backend/runtime tests. |
| #4 | `model-scaffolding-and-runs` | `CHANGES_REQUESTED` | Model adapters, evaluation harness, runner scripts | Medium | Rebase onto current #3 without losing runtime fixes; keep release-1 adapter exposure consistent; align run output with `episode_runs.jsonl` fields. |
| #5 | `reporting-and-examples` | `REVIEW_REQUIRED` | Canonical docs, examples, render/reporting scripts | Small-to-medium | Rebase onto #4; add this immediate design; scrub removed release-1 adapter docs; preserve canonical target design; avoid making scripts/docs the source of truth for schemas. |

### 5.2 Related PRs

| PR | Branch | State | Role | Effort | Must-fix |
|---|---|---|---|---|---|
| #1 | `maze_gen_and_interface` | `REVIEW_REQUIRED` | Historical large drop: generator, NLU code, 200 generated mazes, PNGs, smoke outputs, terminal logs | Do not land as-is | Extract generator ideas into a small standards-based PR that emits `TaskSpecification` fixtures/manifests for tests 2-3. |
| #6 | `ogbench` | `CHANGES_REQUESTED` | Older OGBench submodule attempt | Replace/close | Prefer #11 or a successor with a clear submodule update policy. |
| #8 | `pr1/nlu-interfacing-minimal-mazegen` | `CHANGES_REQUESTED` | NLP/text runner plus maze-generation ideas; parallel to the standard system (custom env, custom maze dataclasses, no `TaskSpecification`, differing switch semantics) | Large | Rebase after #5; split generator from NLP backend; use `TaskSpecification`; implement as `TextBackend` or adapter-compatible runner; remove generated PNG/log artifacts from source. |
| #9 | `pr/benchmark-solver-artifacts` | Clean against #8 | Solver PNG/CSV artifacts for #8 review | Do not land independently | Wait for #8 rewrite; move artifacts to ignored/generated outputs unless explicitly promoted to fixtures. |
| #10 | `baseline-agents` | `DIRTY`, approved | BFS/greedy/random baseline code; dirty against #5 (removes reporting docs, resurrects a removed release-1 adapter, commits large generated result JSON; forward-cell switch semantics conflict with #3) | Medium-to-large | Clean rebase after #8/#5; keep only baseline code/tests; remove generated result dumps; update switch semantics to match validator/runtime; avoid resurrecting removed adapter files. |
| #11 | `codex/add-ogbench-submodule` | `REVIEW_REQUIRED` | Cleaner OGBench submodule PR (`.gitmodules` plus `ogbench` only) | Small | Define ownership/update policy before merge; confirm tests do not require recursive submodule checkout surprises. |
| #12 | `prompts` | `REVIEW_REQUIRED` | Prompt condition definitions for later protocol experiments | Small-to-medium | Remove generated/editor files (`__pycache__`, `.vscode`) and unrelated submodule changes; rebase after the core stack; clarify which prompt conditions are required before tests 1-3. |

## 6. Merge Order

Recommended order for the immediate merge slice:

1. Land #3 first as the runtime/backend layer on top of the merged task schema,
   validator, and scorer foundation. This is where runtime/validator parity and
   duplicate-file scope are settled.
2. Land #4 second as the model-adapter and run-harness layer. It should rebase
   onto #3 and align run outputs with the immediate row schema.
3. Land #5 third as the reporting/docs/examples layer. It should preserve the
   canonical design while adding this immediate merge-scope design.
4. Revise and land #8 fourth as the NLP/text backend and minimal maze-generation
   follow-up. It should rebase onto #5, move into the backend/adapter standards,
   use `TaskSpecification`, and stop committing build/render artifacts.
5. Rebase and land #10 fifth as the baseline-agent suite. The baseline code is
   needed before running test 1 at scale, but generated validation result
   artifacts should be separated from the code review unless the team decides
   they are committed fixtures.
6. Split or revise any remaining maze-generator work after the core stack. The
   generator should produce canonical `TaskSpecification` fixtures and manifests
   for tests 2-3 without pulling in the full historical #1 drop.
7. Decide between #6 and #11 for OGBench handling. Prefer one maintained
   submodule PR, not both.
8. Clean #12 and merge prompts when prompt sensitivity/protocol experiments
   become active.

## 7. Open Decisions

1. Should `greedy_solvability` remain a separate canonical-agent feature during
   test 1, or be promoted immediately into the calibration vector?
2. Should PR #10's generated validation result JSON files be committed, or
   regenerated as local artifacts during calibration runs?
3. Should the NLP backend be implemented as a `TextBackend`, an adapter layer
   around MiniGrid, or a separate smoke-test package that feeds the canonical
   runner?
4. Do tests 1-3 require any OGBench dependency now, or can OGBench be kept out
   of the current merge basis?

## 8. Mapping to Canonical Design

Traceability against `docs/system_design.md`. Useful for cross-checking
component intent against the long-term architecture; not status-critical for
leadership.

| Canonical component | Immediate tests 1-3 version | Current source |
|---|---|---|
| Stage 1 Generate | Curated fixtures plus existing task set | Merged foundation + future cleaned generator work |
| Stage 2 Solve and Score-static | BFS validation, replayable path, 12-dimension score rows | Merged foundation + PR #10 for extra canonical agents |
| Canonical agent suite | BFS validator now; greedy/random after cleanup | Merged foundation, PR #10 |
| Stage 3 Render-and-Run | MiniGrid runtime plus model/baseline runners | PR #3, PR #4, PR #10 |
| Stage 4 Score-runtime | Per-run metric rows, no final composite yet | PR #4 plus small reporting code |
| Stage 5 Aggregate | JSON/CSV summaries for tests 1-3 | PR #5 plus immediate doc work |
| Backend axis | MiniGrid only for this slice | PR #3 |
| MultiGrid backend | Present but not critical path | PR #3 |
| Text/NLP backend | Desired follow-up, needs revision | PR #8 |
| Prompt condition system | Useful later; not blocking tests 1-3 | PR #12 |
| OGBench submodule | External dataset/input dependency; needs ownership decision | PR #11 or successor |

## Appendix A. Artifact Field Reference

Full field schemas for the per-stage artifacts referenced in §3.

### A.1 `canonical_paths.json` (Stage 3.2)

- `bfs.success`
- `bfs.actions`
- `bfs.positions`
- `bfs.optimal_steps`
- `bfs.states_explored`

### A.2 `scored_static.json` (Stage 3.2)

- `is_beatable`
- `dimensions_12`
- `static_score_unweighted`
- Validation messages and structural warnings
- Optional `canonical_agent_features`

### A.3 `episode_runs.jsonl` row fields (Stage 3.3)

- `task_id`
- `experiment`
- `condition`
- `backend`
- `agent_or_model`
- `seed`
- `success`
- `terminated`
- `truncated`
- `reward`
- `steps`
- `optimal_steps`
- `optimality_ratio`
- `path_choice`
- `mechanism_interaction_order`
- `failure_point`
- `tokens`
- `raw_output_ref`

`path_choice` is required for test 2. `mechanism_interaction_order` and
`failure_point` are required for test 3.

## Appendix B. Per-Test Readiness Detail

The binary readiness call in §4 hides per-test variation. For engineers
planning execution:

**Test 1 — Scoring calibration over existing task set**

- Closest to ready. The task set exists; no new fixture generation required.
- Blocked on: #3 (runtime), #4 (adapters), #5 (reporting), #10 (greedy/random
  baselines for calibration runs).
- Remaining engineering after merges: run the calibration suite and inspect
  the generated correlation matrix and weight candidates.

**Test 2 — Complexity-vs-distance validation**

- Requires curated shortcut-maze variants where both routes are open and path
  choice is measurable.
- Blocked on: basis merges plus a small fixture generator (split out of #1/#8).
- Remaining engineering: fixture authoring/generation and `path_choice`
  recording in runs.

**Test 3 — Mechanism-ordering consistency**

- Requires curated mechanism-ordering pairs with path length and topology held
  constant within each pair.
- Blocked on: basis merges plus the same fixture generator work as test 2.
- Remaining engineering: paired-pair fixture authoring. Runtime scoring
  reconstructs `mechanism_interaction_order` and `failure_point`, and Stage 5
  emits paired deltas.
