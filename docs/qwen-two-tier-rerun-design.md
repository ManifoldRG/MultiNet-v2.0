# Qwen two-tier truncation-requeue — design spec

**Status:** design, approved 2026-07-16; **amended 2026-07-16 after full codebase
verification** (corrections integrated in place; the biggest one is the serve-args
finding below).
**Motivation:** Qwen3.6-27B runs locally on A100 via served vLLM. A large output
budget forces a KV-cache tradeoff: `max_model_len=96000` (needed for a 64k output
cap) collapses concurrency from ~13-16 episodes/server to ~2-3, wrecking the
"~4-5 hour, 26-parallel" plan. But Qwen's *observed* output demand is tiny
(median 142, p90 ~2815, max ~4921; ~4.8% of image_only queries hit the buggy 4k
cap). So: run wide at a small cap, and only re-run the few mazes that actually
truncate, at 64k.

## Verified reality: serve args are hard-coded, not config-driven

The served-vLLM path **ignores the run-config's `max_model_len` /
`gpu_memory_utilization` / `enforce_eager`** — those keys only feed the
in-process `qwen_vllm` provider. The server is launched with hard-coded args in
`lib/distributed_start.sh:150-157`
(`--max-model-len 16384 --max-num-seqs 64 --gpu-memory-utilization 0.9`), with
twin copies in the backfill launcher (kept with the run records) and `launch_qwen_smoke.sh:171`. A
reuse guard (`distributed_start.sh:150`) skips relaunch when a server is already
up. Consequences:

- `run_config.r1.json`'s previous qwen block (`max_tokens=64000`,
  `max_model_len=96000`) was **inert-and-broken**: the server would still be
  16384-context while the agent requested 64k outputs. This is fixed as part of
  this work (phase-1 values in the config; see Config summary).
- The implementation must **parameterize the serve line** (env knobs, e.g.
  `QWEN_MAX_MODEL_LEN` / `QWEN_MAX_NUM_SEQS`, defaulting to today's values —
  `launch_qwen_smoke.sh:101` already does this pattern for the smoke path) and
  apply them in all three launch sites.
- The phase-2 **reload** uses the existing teardown primitive
  `stop_gpu_worker` / `lib/gpu_teardown.sh` (`lib/distributed_start.sh:215-222`,
  fail-closed GPU-free polling), then relaunches with phase-2 args. Serve-args
  cannot change at runtime; this is a real ~14-min reload (see the earlier
  served-vLLM concurrency investigation).

## Scope

- **Qwen-only.** Claude/Kimi are API with no KV constraint — they run at 64k from
  the start, no two-tier.
- A general two-tier mechanism, used by R1.

## Two phases, one fleet, one reload

- **Phase 1 (fast/wide):** `max_model_len=16384`, `max_tokens=8000`, full
  concurrency. Runs **all** mazes. (8k vs the prior proven 4k gives headroom for
  the harder balanced panel; because Qwen's median output is ~142, most sequences
  stay tiny and the concurrency hit from the higher cap is modest — only the rare
  long generations pay the extra KV.)
- **Reload (same fleet):** `stop_gpu_worker` per GPU VM, then restart the vLLM
  servers with `max_model_len=96000`, reduced `--max-num-seqs`, and reduced
  worker/coordinator concurrency (`WORKER_CONCURRENCY`, `max_in_flight` ~2-3 per
  server — the cost of the big KV cache).
- **Phase 2 (deep/narrow):** `max_tokens=64000`, reruns **only the flagged
  mazes**. The flagged set is small (~5%), so phase 2 is short despite low
  concurrency.

## Two run configs, not a runtime toggle (verified constraint)

There is no per-phase max_tokens override mechanism, and two pipeline guards
force separate configs:

- The **equal-token-caps guard** (`scripts/run_pipeline.py:831-850`) fires on
  phase 1's mixed caps (qwen 8000 vs Claude/Kimi 64000) → the phase-1 /
  combined R1 config sets `"allow_unequal_max_tokens": true` with a comment
  explaining the two-tier rationale (this is the documented exception the
  invariant allows).
- The **manifest-match guard** (`scripts/run_pipeline.py:811-822`) pins a
  config's declared `manifest` to the CLI `--manifest` → phase 2 needs its own
  run config (`run_config.r1.qwen_phase2.json`, qwen-only, single-model so the
  caps guard auto-passes) whose `manifest` points at the **generated rerun
  manifest** path; the scan tool writes the manifest to exactly that path.
- Phase-2 qwen `timeout` must be raised: measured served decode at low
  concurrency is ~67-100 tok/s per stream, so a genuine 64k generation is
  ~640-955 s — at or over the current 900 s, and `max_attempts=3` would then
  *retry* multi-minute timed-out decodes. Phase 2: `timeout: 2400`,
  `max_attempts: 2`. Phase 1: a cap-hitting 8k decode at concurrency ~13-16 is
  ~1140 s > 900 s; set phase-1 qwen `timeout: 1800` so cap-hitters truncate
  server-side and return, rather than timing out client-side and retrying.
- Hash behavior (verified): `max_tokens` is in the episode/unit hash, so
  phase-2 units are legitimately distinct from phase-1 units (intended re-pay);
  the server knobs (`max_model_len`, `max_num_seqs`, `max_in_flight`,
  `gpu_memory_utilization`) are in `_NON_RUNTIME_MODEL_KEYS`, so the reload
  itself churns nothing.

## Detection & scan

- **Trigger:** an episode is flagged if **any query record has
  `usage.output_tokens >= cap`** OR `finish_reason == "length"` once the agent
  captures it (the batch-runner work adds `stop_reason` capture; use it when
  present, cap-compare as fallback), where `cap` is that phase's `max_tokens`
  (8000 in phase 1).
  This is deliberate and load-bearing: Qwen's truncated steps mostly still report
  `parse_ok=True` (the lenient parser salvages an action from the cut-off
  reasoning), so a parse-failure trigger would miss ~98% of Qwen truncations. The
  cap-hit trigger is the only one that catches the silent, degraded decisions.
- **Exact field paths (verified):** token usage lives on **query records**, not
  step records: `episode.json → transcript[i where kind=="query"].usage.output_tokens`
  (equivalently `queries/query_NNN/query.json → usage.output_tokens`). The
  aggregate `episode_runs.jsonl` `tokens` field is an episode-wide
  *total_tokens sum* and **cannot** drive this trigger. The cap is not in
  `episode.json`; the scan reads it from the sibling
  `run_inputs.json → model_config.max_tokens` (or takes `--cap` explicitly).
- **When:** live as episodes complete in phase 1. Flagged `task_id`s accumulate
  into a **phase-2 rerun manifest** (a subset of the original manifest,
  same task schema, written to the path the phase-2 run config declares).
  Reruns are deferred to phase 2 regardless (the server reload forces a phase
  boundary), so a live scan and a post-phase-1 sweep are functionally
  equivalent; live is chosen for earlier visibility.
- **Where:** `scripts/scan_truncations.py` reading `episode.json` files under
  the phase-1 artifacts root; also usable post-hoc on any past run.

## Result reconciliation & provenance (verified mechanics)

- Each `episode.json` records the `pass` (1 or 2) and the `max_model_len` /
  `max_tokens` it ran under (net-new fields — nothing records these today).
- **Run-dir separation:** the run-dir convention
  (`runs/<task>/<backend>/<model>/seed_<n>/<variant>/`) has no phase segment, so
  without intervention phase 2 **silently clobbers** phase-1 episodes on disk,
  while separate roots double-count in `finalize_job` (`run_rows.append` never
  dedups) and in `analysis/data.py::load_episodes`. Implementation: a phase
  token in the run-dir path (phase-labeled artifacts roots
  `.runs/<run>/qwen_phase1/`, `.runs/<run>/qwen_phase2/` — i.e. separate
  artifacts roots per phase) so both phases persist legibly.
- **Merge = explicit later-pass-wins:** final results = phase-1 episodes for
  unflagged mazes + phase-2 episodes **overwrite phase-1** for flagged mazes
  (keyed by `task_id`). Implemented at the aggregation choke point: a `pass`
  field added to `build_run_row` (`pipeline/episode_metrics.py:245-266`) and a
  dedup (prefer higher `pass` per `(task_id, model, seed, condition, variant)`)
  in the merge/finalize step that produces the combined `episode_runs.jsonl`.
  Never rely on the accidental same-path clobber.

## Terminal case

64k is the ceiling — there is no third tier. If a maze **still** hits
`output_tokens >= 64000` (or `finish_reason=="length"`) in phase 2, keep the
episode and mark it `truncated_at_ceiling: true` rather than looping. (Distinct
name on purpose: `truncated`/`end_reason=="truncated"` already mean
*environment* truncation.) Report these explicitly so the results are
interpretable (a genuinely unbounded Qwen thinking loop is a finding, not a bug
to retry forever).

## Documentation (explicit requirement — clear & replicable)

- The operational runbook lives with the run records: what/why/how, the
  KV-vs-parallelism rationale, the exact reload step and phase-2 serve args,
  how to re-run.
- Inline comments at the phase-transition points in the run scripts
  (`launch_distributed.sh` / `lib/distributed_start.sh` / the sweep driver).
- Phase-labeled artifacts (above) as self-documenting output.
- Goal: a new person can see and reproduce the two-tier flow from the files.

## Testing

- Unit: truncation-flagging — `output_tokens >= cap` at various **query-record**
  positions flags the episode; all-under does not flag; `parse_ok=True`
  truncations are still flagged; `finish_reason=="length"` flags even when
  output_tokens < cap (defensive).
- Scan → rerun-manifest: correct subset of `task_id`s emitted; manifest passes
  the same schema/validation as the source manifest.
- Merge/provenance: phase-2 overwrites phase-1 by `task_id`; `pass` and caps
  recorded on each episode; no double rows in the merged `episode_runs.jsonl`.
- Terminal case: `truncated_at_ceiling` set when phase-2 still hits 64k.
- Serve-arg plumbing: launch scripts render the phase-1/phase-2 `vllm serve`
  lines from the env knobs (bash-level test or a rendered-command assertion).

## Config summary

| Phase | max_model_len (serve env) | max_tokens | timeout / attempts | concurrency | mazes | run config |
|---|---|---|---|---|---|---|
| 1 | 16384 | 8000 | 1800 s / 3 | full (~13-16/server) | all | `run_config.r1.json` (qwen block; `allow_unequal_max_tokens: true`) |
| 2 | 96000 | 64000 | 2400 s / 2 | reduced (~2-3/server, small `--max-num-seqs`) | flagged only | `run_config.r1.qwen_phase2.json` |

Claude/Kimi (for reference, not two-tier): 64000 cap from the start, via the
batch lockstep runner (`docs/batch-api-lockstep-runner-design.md`).

**Freeze during the campaign:** `SCORER_VERSION`, `PIPELINE_VERSION`, and scorer
weights must not change between phase 1 and phase 2 (or between the smoke and
the full run) — unit identity folds them in, and churned unit_ids on a re-prepare
orphan verified (paid) units; fresh workers would re-pay episodes.
