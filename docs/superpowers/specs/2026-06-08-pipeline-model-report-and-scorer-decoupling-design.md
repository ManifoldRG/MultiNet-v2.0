# Pipeline: per-model report + scorer/interface decoupling

**Date:** 2026-06-08
**Branch:** `feature/run-pipeline`
**Status:** Approved design — pending implementation plan

## Goal

Two small, related changes to the bare-bones run pipeline so it supports the
intended spine **(pre-made mazes → solvers → interface/API to a target model →
run → per-model report)** with the scoring infrastructure kept out of the
solver path:

1. **Scorer/solver decoupling** — make the solver + static-score path runnable
   without importing the `interface` (model/runner) stack, so `scorer` is its
   own module and Stages 1–2 never drag in the heavy interface dependencies.
2. **Per-model report** — emit one machine-readable JSON report per model,
   separate from the scorer-calibration ("tuning") artifacts.

## Context

- `scripts/run_pipeline.py` drives Stages 1–5 sequentially. Stage 2
  (`_score_suite → score_tasks → scorer.score_task_file`) runs the BFS/greedy
  solvers and writes `canonical_paths.json` **before** Stage 3 builds any agent
  or calls a model. So solvers already run before the interface *logically*.
- However, **importing the solver path pulls in `interface`**:
  `scorer/runtime.py` and `pipeline/episode_metrics.py` do
  `from interface.telemetry import token_count_from_record` at module load, and
  `interface/__init__.py` eagerly imports `interface.runner`, so that single
  import drags in `pygame`, `PIL`, `interface.runner/renderer/loader`. This is
  the contamination to remove.
- Tasks are sourced via a **manifest** (one row per maze JSON: `source` +
  metadata such as `experiment`, `condition`, `expected_mechanisms`, `pair_id`,
  route cells) and a **run-config** (model → task selection). Sourcing needs no
  code change; the user provides manifests and symlinks the maze JSONs in under
  `gridworld/tasks/`. `source` paths resolve through symlinks transparently.

## Non-goals (explicitly out of scope)

- **No maze-generation stage.** Generation lives upstream and is kept separate
  to minimize contamination risk. Mazes are plain JSON files supplied to the
  pipeline.
- **No relocation of shared telemetry** to a neutral module yet. We use lazy
  imports (Approach 3) to keep the PR footprint small. Extracting telemetry and
  other shared cross-module components into a neutral home is **deferred** to a
  later PR.
- **No scorer tuning.** The report is intentionally shipped before calibration;
  its scored fields are provisional.

---

## Section 1 — Scorer / solver decoupling (Approach 3: lazy imports)

Three lazy-import edits. No file moves, no public-API changes.

1. `scorer/runtime.py` — move `from interface.telemetry import
   token_count_from_record` from module scope into the function that uses it
   (the per-record token sum, currently ~line 83).
2. `pipeline/episode_metrics.py` — move the same import into
   `episode_token_count` (currently ~line 157).
3. `scripts/run_pipeline.py` — defer the interface imports into the functions
   that need them:
   - `from interface.config import ExperimentConfig` → into `_condition_configs`
     (the only runtime use; annotations already rely on
     `from __future__ import annotations`, so type hints need no import).
   - `from pipeline.run_stage3 import run_episode` → into `_run_one_model`
     (Stage 3).
   - `from prompting_experiments import CONDITION_SETS, iter_condition_configs`
     stays eager — verified interface-free at import time.

**Outcome:** `import scorer` and importing/running `run_pipeline` Stages 1–2
load no `interface` code; `interface` loads lazily only when Stage 3 runs a
model.

### Verification

A new import-isolation test asserts:
- `import scorer` → `interface` not in `sys.modules`.
- Importing `scripts.run_pipeline` and importing `pipeline.episode_metrics` →
  `interface` not in `sys.modules` (must run in a fresh interpreter per case,
  e.g. via `subprocess`, because other tests import `interface`).

---

## Section 2 — Per-model report

### Component

New pure function in `pipeline/reports.py`:

```python
def model_report(
    run_rows: list[dict],
    composites: dict[tuple, float | None],
    model_id: str,
    run_set_id: str,
) -> dict[str, Any]: ...
```

`backend` is read from the filtered rows (`row["backend"]`); `run_set_id` is
passed through from `_write_aggregate`.

- Filters `run_rows` to `row["agent_or_model"] == model_id`.
- Reuses a private `_summary(rows, composites)` helper to compute the metric
  block, applied to the overall set and to each `by_experiment` / `by_prompt_variant`
  group.
- Reads per-run composites via the existing `_run_key(row)` (now the 5-tuple
  including `prompt_variant`), skipping `None`.

`_summary` block:
`{ success_rate, optimality_ratio_mean, optimality_ratio_median (successful runs
only), steps_mean, tokens_mean, tokens_total (None tokens skipped),
composite_mean (provisional) }`.

### Wiring

In `_write_aggregate`, after writing the tuning artifacts: collect the distinct
`agent_or_model` values from `run_rows` and write one report per model to
`artifacts/reports/<run_set_id>/models/<model_id>.json` (model_id already
sanitized upstream; re-sanitize defensively for the filename). Add the reports
to the returned `payloads` under `"model_reports": {model_id: payload}` for
programmatic callers/tests. The tuning artifacts are unchanged and remain at
`artifacts/reports/<run_set_id>/*.json`.

### Schema (`schema_version: "0.1.0"`)

```jsonc
{
  "schema_version": "0.1.0",
  "model_id": "...", "run_set_id": "...", "backend": "minigrid",
  "seeds": [0], "task_count": N, "run_count": M,
  "provisional": true,
  "overall": { "success_rate", "optimality_ratio_mean", "optimality_ratio_median",
               "steps_mean", "tokens_mean", "tokens_total", "composite_mean" },
  "by_experiment":     { "test1": { …same block… }, "test2": {…}, "test3": {…} },
  "by_prompt_variant": { "default": { …same block… } },
  "tasks": [
    { "task_id", "experiment", "condition", "prompt_variant", "seed",
      "success", "steps", "optimal_steps", "optimality_ratio",
      "path_choice", "tokens", "composite" }
  ]
}
```

Notes:
- `run_set_id` is threaded into `_write_aggregate`/`model_report` (already a
  parameter of `_write_aggregate`).
- `provisional: true` is a constant flag until the scorer is tuned. Raw metrics
  (success/steps/optimality/tokens) are meaningful now; `composite*` fields are
  placeholders.
- One standalone report per model, single shared schema. No separate
  cross-model comparison file — comparison is an external tool's job.

### Testing

- **Unit** (`tests/test_reports.py`): `model_report` over run_rows for two
  models → assert per-model `overall`, `by_experiment`, `by_prompt_variant`,
  and per-task rows; the two models yield two independent reports; provisional
  flag present.
- **E2E** (`tests/test_run_pipeline.py`): existing pipeline test asserts
  `artifacts/reports/<run_set_id>/models/<model>.json` exists and carries the
  schema keys; `payloads["model_reports"]` keyed by model.

---

## Risks / edge cases

- Empty groups: `_summary` must return `None` (not crash / not NaN) for means
  over zero successful runs or all-`None` tokens — reuse existing `_mean`/`_median`.
- `tokens` may be `None` for runs without query telemetry — skip in
  `tokens_mean`/`tokens_total`, do not coerce to 0.
- Import-isolation tests must use fresh interpreters; the broader suite imports
  `interface`, which would pollute `sys.modules` within a single process.
- Filename safety: derive the report filename from a sanitized `model_id`.

## Deferred (future PRs)

- Extract `telemetry` (and other shared cross-module helpers) into a neutral
  module so `scorer` has no `interface` import at all (replacing the lazy-import
  stopgap).
- Optional `--solve-only` entry that stops after Stage 2 for fully standalone
  solver runs.
- Scorer calibration/tuning (makes the report's scored fields meaningful).
