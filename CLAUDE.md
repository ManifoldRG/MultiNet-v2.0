# MultiNet-v2.0 — working notes for Claude

Gridworld/multigrid benchmark: VLM/LLM agents navigate procedurally generated
mazes (keys, doors, switches, gates) under controlled prompt/observation/query
conditions. The 2026-07 conditional sweep (Claude Opus 4.8, Kimi k2.6, local
Qwen3.6-27B on A100 vLLM; 540 episodes) is complete — findings in
`analysis/FINDINGS.md`. Setup/commands: `RUNME.md`.

## Layout (the parts that matter)

- `gridworld/` — task specs (`task_spec.py`), runtime env (`custom_env.py`,
  built by `task_parser.py`), beatability + difficulty (`task_validator.py`),
  executable-action planners (`baselines.py`), fixtures (manifests +
  run-configs) in `gridworld/fixtures/`
- `interface/` — episode runner (`runner.py`), prompt assembly
  (`prompt_strategies.py`, `observation.py`), reply parsing (`parser.py`,
  `querying.py`), model agents in `interface/agents/`
- `prompting_experiments/prompt_templates/` — every prompt string lives here,
  not inline
- `scorer/` — canonical paths + static/runtime scoring; writes per-run
  `run_score.json` and `episode_runs.jsonl`
- `scripts/` — `run_pipeline.py` (local runs, run-config guards),
  `distributed_run_pipeline.py` (coordinator/worker jobs),
  `prepare_combined_job.py` (run-massive: many batches, one saturated job)
- `sweep_run.sh` + `launch_distributed.sh` + `lib/cost_safety.sh` — GCP fleet
  driving; `deploy/` — VM setup
- `analysis/` — pandas loaders (`data.py`), findings, figures (untracked)
- `ogbench/` — **git submodule** (vendored; pytest ignores it via pyproject)
- `mazes/`, `artifacts*/`, `.runs/`, `Multinet-v2-results/` — data, mostly
  untracked/ignored

## Invariants that have bitten us (do not re-break)

- **One goal source.** The env scores `reach_position` against `goal.target`;
  `TaskSpecification.resolved_goal()` is the single source for solvers, the
  rendered goal tile, and prompts. `validate()` rejects specs where
  `maze.goal` ≠ `goal.target`.
- **Difficulty counts executable actions.** `compute_difficulty` prefers
  `plan_bfs_path` (a locked door costs TOGGLE + MOVE_FORWARD; same-cell key
  pickup costs 1 PICKUP). Runtime, solver, and validator must agree on
  mechanics or paid runs mis-score.
- **FINAL_OUTPUT is the answer line.** All query modes instruct
  `FINAL_OUTPUT:`; `QueryingMode.parse_actions` parses it first — never let a
  mid-reasoning "Actions:" mention preempt it.
- **Multiturn turns stay lean.** With `chat_history=rolling/full`, the chat is
  the history: no one-shot example and no `context_window` sections inside
  stored turns (duplication collapsed Claude 53%→7% once).
- **Equal token caps across models** in one run config, or declare
  `"allow_unequal_max_tokens": true` (unequal caps confounded
  baseline_thinking).
- **Kimi/Moonshot temperature is mode-forced**: thinking ON → 1.0, OFF → 0.6;
  Opus 4.7+ rejects sampling params (never send temperature).
- **Coordinates:** specs/solvers use (x, y); prompts use (row, col) — convert
  only through `interface/coords.py`.
- **Cost safety:** `MAX_RUN_DURATION` and `BATCH_CAP` are required, no
  defaults. STOP VMs, don't delete; pull artifacts before spindown. On-VM code
  must match the local committed SHA (git-archive push + verify, fail-closed).

## Commands

- Tests: `pytest` (repo root; `--ignore=ogbench` is automatic via pyproject)
- One local run: `python -m scripts.run_pipeline --run-config <rc.json>
  --manifest <manifest.json> --conditions <cond> --seeds 0 ...`
- Fleet sweep: `sweep_run.sh` subcommands (provision / next-batch /
  run-massive / finalize-batch / publish / teardown); batch-starting
  subcommands require `BATCH_CAP`
- Episode artifacts land under `<artifacts-root>/runs/<task>/<backend>/
  <model>/seed_<n>/<variant>/episode.json`, aggregated to `episode_runs.jsonl`

## Conventions

- Feature work on branches; never merge/push to `main` directly
- Small PR footprints; prefer lazy imports over moving shared modules
- Tests colocate in `tests/`; e2e runner tests script an agent and drive
  `build_runner` (see `tests/test_cardinal_runner.py`)
- `docs/superpowers/` (dev planning) is gitignored — don't `git add -f`
- Design proposals for the next iteration: `docs/design-proposals-2026-07.md`
