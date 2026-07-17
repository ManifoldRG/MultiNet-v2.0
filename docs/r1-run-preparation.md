# R1 run preparation — preflight verification, footguns, launch checklist

**Date:** 2026-07-16. **Scope:** the R1 fixed-cell run (`gridworld/fixtures/run_config.r1.json`
on `manifest.r1_balanced_03.json`, 50 mazes × 1 seed × 3 models), its 5-maze
validation smoke, and the two supporting designs:
`docs/batch-api-lockstep-runner-design.md` (Claude/Kimi batch lockstep) and
`docs/qwen-two-tier-rerun-design.md` (Qwen phase-1/phase-2). Both specs were
verified line-by-line against the codebase and provider docs on 2026-07-16;
corrections are integrated into the specs themselves. This document holds
everything else: verification results, the footgun catalog, and the launch
sequence.

## Preflight verification results (all clean unless noted)

- **Manifest ↔ panel:** `manifest.r1_balanced_03.json` matches
  `analysis/candidate_mazes/sets/balanced_03/SUMMARY.md` **exactly 1:1** — 50
  tasks, every source path and `optimal_steps` value matches the walkthrough,
  strictly ascending 23–106, provenance recorded in the manifest's `selection`
  block. The other three R1 manifests (long_tail_02, mechanism_rich_01,
  pairwise_01) exist and parse at 50 tasks each.
- **Submodule-path safety:** all R1 manifests point into the `ogbench`
  submodule. Safe since commit `36ebf73`: `launch_distributed.sh` ships each
  submodule tree at the superproject-pinned sha and **fail-closes** if a
  submodule is missing/empty on a VM. Vendoring into `mazes/` is no longer
  required. The S4 goal defect is fixed at the pinned sha (`31a0549`).
- **Beatability/validation:** all 50 mazes audit clean (beatable, BFS-solvable,
  `optimal_steps` consistent). Re-validate any new/edited manifest with
  `python scripts/validate_fixtures.py --manifest <path>`.
- **Provider batch APIs:** both confirmed real and sufficient (Anthropic 50%
  off, Moonshot 40% off, thinking + images supported, results carry
  `stop_reason`/`finish_reason`). Details and request-shape requirements live in
  the batch-runner spec.

## Footgun catalog

Each item: the trap, and where the mitigation lives.

1. **Stale Qwen block in `run_config.r1.json`** (user-flagged). The committed
   config had qwen at `max_tokens=64000 / max_model_len=96000` — phase-2 values
   with no two-tier mechanism. Worse: on the served path `max_model_len` (and
   `gpu_memory_utilization`, `enforce_eager`) are **inert** — `vllm serve` args
   are hard-coded in `lib/distributed_start.sh:150-157` at 16384 context, so the
   old config would have requested 64k outputs from a 16k-context server.
   *Mitigation:* two-tier implementation parameterizes the serve line; config
   carries phase-1 values (`max_tokens=8000`, `allow_unequal_max_tokens: true`);
   phase 2 gets its own config. See the two-tier spec.
2. **Uncommitted working tree vs fail-closed code-sync.** The early-terminate
   branch has ~728 uncommitted lines (runner stall-K hardening, coordinator
   inputs-hash verification, tests) plus untracked candidate-maze tooling. VM
   code-sync pushes the **committed** sha and fail-closes on mismatch — nothing
   runs until this work is committed. *Mitigation:* commit (including
   implementation work) before any smoke/run; verify `git status` clean and the
   ogbench submodule pointer pushed.
3. **`max_in_flight=1` for Claude/Kimi vs the lockstep working set.**
   `max_in_flight` is a fleet-wide per-group throttle checked at `assign`; at 1
   it would starve the batch runner to a single active maze. *Mitigation:* R1
   config raises Claude/Kimi `max_in_flight` to ≥ 50 (= MAX_BATCHES /
   `worker_concurrency`). Invariant documented in the batch-runner spec.
4. **No kill layer between unit staleness and VM shutdown.** A wedged batch
   round is kept "alive" by our own heartbeats; the only backstop is the
   `BATCH_CAP` VM shutdown (the 6h-watchdog incident, amplified by 24h batch
   expiry). *Mitigation:* per-round deadline in the batch runner (spec), and
   size `MAX_RUN_DURATION`/`BATCH_CAP` from smoke-measured round latency —
   batch rounds have minutes-to-unbounded variance, not sync-API latency.
5. **Unit-hash churn re-pays paid units.** `unit_id` folds in scorer identity
   (`SCORER_VERSION`, scorer weights, `PIPELINE_VERSION`, task rows). Bumping
   any of these between prepare calls of the same campaign orphans
   verified/uploaded units; fresh workers re-pay. *Mitigation:* **freeze
   scorer/pipeline versions and weights for the entire R1 campaign** (smoke →
   phase 1 → phase 2). The deeper design question (should unit identity cover
   scoring inputs at all) stays open but is explicitly deferred past R1.
6. **Phase collision / double-count.** Same run-dir for both Qwen phases would
   silently clobber; separate roots double-count in finalize and
   `analysis/data.py`. *Mitigation:* phase-labeled artifacts roots + `pass`
   field + later-pass-wins dedup at aggregation (two-tier spec).
7. **Truncation trigger reads the wrong field if implemented naively.** Usage
   is on **query records** (`transcript[kind=="query"].usage.output_tokens`),
   not step records; `episode_runs.jsonl`'s `tokens` is a total-tokens sum and
   can't drive the trigger; the cap lives in `run_inputs.json`, not
   `episode.json`. *Mitigation:* spelled out in the two-tier spec; unit tests
   pin it.
8. **`truncated` name collision.** `truncated`/`end_reason=="truncated"` mean
   *environment* truncation today. Token-cap truncation uses distinct names
   (`token_truncated`, `truncated_at_ceiling`, recorded `stop_reason`).
9. **Kimi specifics.** (a) `max_tokens` defaults to 32768 if unset — always
   explicit; (b) sampling params are mode-forced (thinking→1.0) — omit or match,
   anything else 400s; (c) Kimi thinking-runaway truncated 5.6% of queries at
   16k with **empty content** — rate at 64k unknown; the smoke measures it and
   `stop_reason` capture makes it visible. Kimi remains the likeliest
   parse-fail artifact source; judge from smoke data before the full run.
10. **Claude Opus 4.8 request shape.** `thinking: {"type":"adaptive"}` +
    `output_config.effort`; `budget_tokens`/`temperature`/`top_p`/`top_k` are
    hard 400s; thinking display defaults to omitted (set `summarized` to log
    traces; billing identical). Already honored by the sync agent; batch
    implementation must match.
11. **Batch-runner uploads must reuse the run-inputs machinery.** Upload
    verification rejects archives whose `run_inputs.json` `inputs_hash` ≠ the
    unit's `episode_inputs_hash` (new check on this branch). The lockstep worker
    goes through the same `run_inputs` writer as `pipeline._run_one_unit`.
12. **last3 + text-summary prompt comparability.** The reworked last3 history
    (`FINAL_OUTPUT:` + Position-after/Feedback shape) AND the new persistent
    start-pose anchor in the text summary (`"You started at (r, c) facing DIR."`,
    added 2026-07-17 to ground image_only) mean R1's `text_summary_and_last3`
    data is **not prompt-comparable** with previously-collected corpora (incl.
    the kimictx run). Deliberate. Never pool old and new arms silently in analysis.
13. **Panel interpretation caveats** (from the balanced_03 SUMMARY, restated so
    they survive into analysis): mechanism comparisons begin at 26 actions (no
    shorter mechanism fixtures exist); the single D3 maze's BFS estimate may be
    inflated vs legal runtime execution; matched pairs/triplets are the causal
    units, not the marginal curve.
14. **Cosmetic:** the balanced_03 walkthrough PNG for `S4/10x10_dense_1` still
    renders the pre-fix goal position. Re-render + re-package when convenient;
    does not affect runs (runtime reads `goal.target` via `resolved_goal()`).
15. **Kimi transcripts now include thinking traces.** The Reply-based agents
    capture Kimi's `reasoning_content` (previously discarded), so paid Kimi
    runs record `thinking` on every query record — intended (it's how
    truncation-vs-runaway is diagnosed) but transcript/artifact size grows
    accordingly.
16. **Order-dependent test flake:**
    `tests/test_launch_distributed.py::test_sync_ships_and_verifies_submodule_content`
    failed once in full-suite order (passes in isolation and in most full
    runs). Not caused by the feature work; diagnose before treating a red
    full-suite as a launch blocker.

## Launch sequence (in order; nothing runs before its predecessor)

Steps 1–2 are **DONE** (2026-07-16, commits `d1196a6..00757f3` on
`feature/early_terminate`): both designs implemented with per-task review
(batch lockstep runner incl. Anthropic/Moonshot batch clients, `EpisodeStepper`
extraction, query-boundary checkpoint/resume, `lockstep-worker` coordinator
role; two-tier serve-arg env knobs + `reload-qwen-phase2`, truncation scanner,
phase provenance, later-pass-wins merge, both run configs, operator runbook
`docs/qwen-two-tier-rerun.md`); full suite green (1006+ tests); smoke fixtures
validated. Remaining steps are operator-run (paid).

1. ~~Implement~~ **DONE.** Fixtures: `manifest.r1_smoke_batch.json` +
   `run_config.r1_smoke_batch.json`; validate any edited manifest with
   `python -m scripts.validate_fixtures --manifest <path>` (must be invoked
   with `-m`; bare-path invocation lacks sys.path).
2. ~~Commit everything~~ **DONE** (submodule pointer at the S4-fixed sha).
3. **Validation smoke** (5 mazes, Claude + Kimi via `generate_batch`, plus a
   serial sync control; ~$5–25):
   `SMOKE_BUDGET_ACK=1 python -m scripts.run_batch_smoke --models claude_opus,kimi_k26 --max-batches 5 --control-episodes 1 --max-usd 25 --artifacts-root .runs/r1_smoke_batch`
   (use `--dry-run` first). Verifies custom_id mapping, ragged termination,
   checkpoint/resume, price delta; **measures queries/episode, output/query,
   per-round batch latency, and Kimi/Claude truncation rates at 64k** into
   `smoke_report.json`. Also the first live confirmation of the Moonshot batch
   wire shapes (implemented from docs, unverified against the live endpoint).
4. **Re-budget from smoke numbers** (prior central ~$435 sync → ~$229 with
   batch discounts; smoke collapses the $200–1025 range). Set
   `MAX_RUN_DURATION` / `BATCH_CAP` from measured round latency — count the
   full worst-case round: `round_deadline_s` (default 2h) **+ cancel grace
   (default 5 min)**, times a multi-round tail. **Decision gate:** proceed /
   adjust caps / Kimi thinking-off fallback if its 64k truncation rate is
   still pathological.
5. **Full run, phase 1:** Claude + Kimi through the lockstep batch worker —
   fleet-launched by exporting `API_WORKER_ROLE=lockstep-worker
   API_WORKER_CONCURRENCY=50` before starting the API workers
   (`lib/distributed_start.sh::start_worker`; defaults unchanged = old serial
   role), or manually via `--distributed-role lockstep-worker
   --worker-concurrency 50`. Exactly ONE lockstep worker per API model group;
   coordinator serves with `--stale-after-seconds 9000` (default 300 is wrong
   for batch rounds); the model groups' `max_in_flight` is already 64 ≥
   MAX_BATCHES. Qwen runs wide at 8k on the A100 fleet (serve env unset =
   phase-1 defaults). Cost-safety env vars are required-no-default; STOP VMs,
   never delete; pull artifacts before spindown.
   - **Coordinator staleness for lockstep groups:** start `coordinator-serve`
     with `--stale-after-seconds 9000` (must be ≥ `batch_deadline_s` +
     `batch_cancel_grace_s` + slack; R1 default 7200 + 300). The lockstep
     worker only heartbeats between rounds (`on_round`), and a single batch
     round can run for the whole 2 h deadline with no heartbeat — the default
     `stale_after_seconds=300` would flip every held unit stale mid-round.
   - **Exactly ONE lockstep worker per API model group.** With one worker,
     a stale bounce is harmless (the coordinator re-hands the same worker's
     units and refill dedups them, costing only an `attempts` increment). With
     two workers on a group, a stale unit is re-assigned to the *other* worker
     → double-run → double-pay. Size `MAX_RUN_DURATION` / `BATCH_CAP` to the
     same worst-case round (`batch_deadline_s` + `batch_cancel_grace_s`) × the
     multi-round tail (see step 4).
6. **Scan → phase 2 → merge:** exact commands in `docs/qwen-two-tier-rerun.md`
   (scanner is fail-closed on unresolvable caps; `reload-qwen-phase2` requires
   `BATCH_CAP`; merge is fail-closed on missing flagged tasks and stamps
   `pass`/`truncated_at_ceiling` on merged copies only).
7. **Finalize + analysis:** merged `episode_runs.jsonl`, `run_score.json`,
   budget/actuals reconciliation, `truncated_at_ceiling` cases reported
   explicitly. Analysis note: rows now carry additive `pass`/`max_tokens`
   columns; query records carry additive `stop_reason`/`token_truncated`.

## Budget (smoke-measured, 2026-07-17)

The pre-smoke central estimate (~\$229 batch across all 3 models) assumed higher
queries/episode than reality: under image_only the episodes **stall at ~30–80
steps** (stall-K=30), well below a solve, so Claude comes in far cheaper than the
old \$165 line. Smoke-measured (Claude, 50 mazes × 1 seed):

| Model | R1 batch | R1 sync | Basis |
|---|--:|--:|---|
| **Claude Opus 4.8** | **~\$23.4** | ~\$46.8 | measured: 49.8 q/episode × 50, mean 574 out-tok/step (`smoke_report.json`) |
| **Kimi k2.6** | ~\$50–100 (uncertain) | ~\$85–170 | single-round datapoint only: ~16.5k reasoning tok/step — measure with a Kimi smoke before trusting |
| **Qwen** | \$0/token | \$0/token | local vLLM |

Full tables + provenance: `analysis/r1_smoke_batch_findings.md`. `token_truncated`
was 0/249 for Claude at 64k. Cost lever ranking unchanged: output cap runaway
tail > batch discount > prompt caching (input minor at xhigh). **Kimi is the cost
*and* wall-clock long pole** (heavy per-step thinking + Moonshot's ~17 min/round).

## Lessons learned from the batch smoke (2026-07-17)

The 5-maze Claude batch smoke (`analysis/r1_smoke_batch_findings.md` has the full
tables) surfaced several things that will bite the live R1 launch if not checked.

### Smoke result summary (Claude, image_only, thinking-on, batch)
- All 5 mazes **stalled** (0 solved) at exactly `stall_gap = 30` → stall-K=30 is
  the terminator. Queries/episode 32–83 (mean 49.8). `token_truncated = 0`.
- Per-step tokens: output mean **574** / median 252 / max 21426; input mean 886.
- **Cost:** batch $2.34 vs sync-equiv $4.68 (**50% discount confirmed**). Official
  projection: **Claude batch ~$23.4 / sync ~$46.8** for 50 mazes × 1 seed.
- **Latency:** Anthropic batch **median 6–7 min/round** (max 31); Moonshot batch
  **median 17 min/round** (max 67) — Kimi-via-Moonshot is the wall-clock long pole.

### Pre-launch checks (do these on the LIVE coordinator run, early)
1. **Confirm thinking is actually ON** — the #1 trap. The smoke driver built its
   batch agent from the stripped `plan["models"][key]` (no `enable_thinking`/
   `effort`) and silently ran **thinking-OFF** (uniform ~24-token outputs, all
   stalled). The real `run_lockstep_worker` uses `unit["model_config"]` and is
   correct, **but verify it live**: within the first 1–2 rounds, pull a query
   record's `usage.output_tokens`. **~24 tokens = thinking-OFF (STOP, investigate);
   hundreds–thousands = thinking-ON.** Also check `thinking` non-empty on Claude
   (display=summarized) and `reasoning_content` on Kimi.
2. **Check token use early** to catch runaway before it burns budget — median is
   ~250–570 out-tok/step for Claude; Kimi thinks *heavily* (~16.5k tok for one
   step in the confirmation). A first-round spot check bounds the projection.
3. **Qwen is slow to start — don't panic.** Qwen's first solve takes **~15–20 min**
   (vLLM load + first decode on the served A100 path); an early "no progress" is
   startup, not a hang (see the qwen-fp8 smoke-stall note). Check after ~20 min.

### If the batch coordinator breaks
- **At launch (first ~hour, no real work done):** kill the job, fix the bug,
  relaunch clean. Cheap — nothing to salvage.
- **After ~an hour (real batches in flight / episodes on workers):** **monkeypatch
  live** rather than restart. Completed episodes live on the workers as
  `episode.json`; a coordinator restart risks mid-flight batches and re-pays. The
  lockstep worker checkpoints per round (`checkpoint.json` in the run dir) and the
  Anthropic/Moonshot batch history is queryable directly (`GET /v1/messages/batches`
  / `GET /v1/batches`) to inspect in-flight state without touching the process.

### Known behaviors / gotchas
- **Two config sources:** `plan["models"]` is a routing/topology view (stripped of
  runtime params); `unit["model_config"]` carries `enable_thinking`/`effort`/
  `temperature`. Always build agents from the unit config (the smoke fix + the
  worker both do).
- **image_only gives NO action-outcome feedback by design** ("PNGs + inventory/
  action labels, no text history"). The model can't tell a move was BLOCKED, so it
  repeats it until stall-K — this is the fast-stall mechanism, not a bug. The
  text-summary now carries a persistent **"You started at (r,c) facing DIR."**
  anchor to partially ground it.
- **Temperature:** Claude Opus 4.8 **rejects** `temperature`/`top_p`/`top_k` (400) —
  never send it; it samples via adaptive thinking. Kimi is mode-forced to 1.0.
  Qwen raised 0.6 → 1.0 to match Kimi (0.6 was Kimi's *non-thinking* value).
- **Batch API honors thinking** for both providers (verified by direct sync-vs-batch
  A/Bs: Anthropic text+image; Moonshot single round, no 64k truncation) — but this
  is a per-run-path property, so item (1) above still applies to the live coordinator.
- **Smoke ≠ worker on checkpoints:** the smoke driver runs the lockstep runner
  WITHOUT a checkpoint path (episodes flush only at leg end); the real worker
  checkpoints per round. Don't infer worker resume behavior from smoke disk state.
