# R1 full run (`r1-20260717`) — live incidents, fixes, and operator actions

Running log of everything addressed during the supervised R1 production run
(launched 2026-07-18 ~00:02 CEST, sha `5941806`, fleet `r1-20260717` in
us-central1-a). Written so nothing is lost to context compaction; each entry
notes what happened, the evidence, the action taken, and any follow-up owed.

## 1. Launch-time: `DIFFICULTY_MAX` default would have aborted provision

- **Trap:** `sweep_run.sh` defaults `DIFFICULTY_MAX=1000`; R1's largest static
  score is **2688.35**. Provision's `coordinator-prepare` would have failed
  after VM creation (fleet auto-STOPped, launch dead on arrival).
- **Caught:** local dry-run of the exact prepare command pre-launch.
- **Fix:** launched with `DIFFICULTY_MAX=3000`. Dry-run and live prepare both
  produced the identical job (`job_e0133acc7ebd`, 150 units).

## 2. Launch-time: coordinator staleness flag is not plumbed

- **Trap:** `lib/distributed_start.sh::start_coordinator` starts
  `coordinator-serve` with **no** `--stale-after-seconds`; argparse default is
  300 (`scripts/run_pipeline.py:1152`). Lockstep batch rounds run 7–67 min with
  no heartbeat → every held unit would flap stale (double-pay risk with >1
  worker; the launch-blocking check from the handoff).
- **Fix (operator, no code change):** exploited the launcher's serial worker
  start (GPU vLLM loads first, ~45 min before API workers exist) to kill and
  restart `coordinator-serve` manually with `--stale-after-seconds 9000` inside
  that window. Verified in the live process line before any worker registered.
- **Follow-up owed (post-campaign):** plumb a `STALE_AFTER_SECONDS` env through
  `start_coordinator`; every manual serve restart must remember the flag until
  then.

## 3. Kimi leg dead at launch: Moonshot batch balance precheck (the big one)

- **Symptom:** round 1, all 50 units → "no valid actions parsed" ×3 attempts,
  then finalize failures ("Runtime scoring requires positive token telemetry").
  Worker stopped per the 4xx-storm playbook after diagnostic capture.
- **Root cause:** Moonshot rejects a batch **wholesale** at submission if the
  account balance can't cover the batch's worst case (~50 × 64k max_tokens ≈
  $8–12). Account balance was **$7.09**. All 7 submitted batches show
  `status=failed, completed=0` with `failed_precondition: user has insufficient
  balance` → **$0 was billed**. The 5-maze smoke never caught this because its
  single-request batches passed the precheck.
- **Cascade:** each rejected batch fails in ~38 s and surfaces as per-item
  parse failures, so episodes burn all 3 parse attempts in ~2 min, finalize
  fails, the coordinator re-hands the unit, and within ~13 min all 50 units hit
  `max_unit_attempts=3` → dead-lettered (assignment skips them).
- **Recovery (2026-07-18 ~01:0x, after user recharged to $32 + auto-refill
  below $15):**
  1. Backed up `job_state.json`
     (`artifacts/r1-20260717/distributed/job_state.json.bak-kimi-reset-*` on the
     coordinator), killed serve, reset **only** the 50 kimi units
     (`worker_1831564f4bfe`) to `pending/attempts=0/progress=0`. Deliberately
     did NOT use the global re-prepare: it would also reset mid-flight
     Claude/Qwen units, and Qwen's 3-worker group makes that a double-run risk.
  2. Restarted serve with `--stale-after-seconds 9000` (see #2), verified
     status: 98 running untouched + 52 pending.
  3. Cleared the kimi VM's poisoned worker state (`runs/`, `tasks/`,
     `worker_state.json` — checkpoints were `finished=true` from parse
     failures, and the lockstep worker resumes existing checkpoints on start).
     Old log kept as `worker.log.failed-round1`.
  4. Restarted the lockstep worker (`API_WORKER_ROLE=lockstep-worker`,
     `API_WORKER_CONCURRENCY=50`). It re-took all 50 units; first Moonshot
     batch after recharge (`batch_6a5aae8a…`) went **`in_progress`** — precheck
     passed, leg live.
- **Follow-ups owed (post-campaign, do NOT hot-fix mid-run):**
  - Batch-level rejection must not masquerade as per-item parse failures — it
    burns all unit attempts in minutes and dead-letters the whole panel. Detect
    batch `status=failed` and fail loudly/pause instead.
  - Add a provider **balance/quota preflight** to the launch checklist and/or
    the lockstep runner (Moonshot: `GET /v1/users/me/balance`).
  - Smoke lesson: a batch smoke must include at least one **full-size** batch
    to exercise the balance reservation, not only 1-request batches.
  - Operational note: with the $15 auto-refill floor, a worst-case round
    reservation (~$8–12) still clears, but a raised `max_tokens` or panel size
    would not — recheck if either changes.

## 4. Claude thinking-ON gate: PASSED (with a budget note)

- First 100 query records: 73% carry a thinking block; output tokens
  min 23 / median 442 / p75 1154 / max 19,621; `stop_reason` 100% `end_turn`
  (zero `max_tokens`). The 27% ~24-token replies are adaptive thinking
  legitimately skipping trivial moves — NOT the smoke's uniform-24
  thinking-OFF bug (worker builds agents from `unit["model_config"]`,
  confirmed live).
- **Budget note:** mean output 1374 tok/query vs smoke's 574 → Claude
  projection revised ~$23 → **~$40–55** (batch). Still far under the account
  cap. Track at each check-in.

## 5. Transient annoyances (no action needed)

- Local DNS resolution to `compute.googleapis.com` failed twice, transient,
  immediate retry succeeded both times.
- `worker_count=6` on the coordinator: the dead kimi worker's registration
  lingers alongside its replacement. Harmless (registrations are not
  assignment targets; units were explicitly reset).

## 6. Kimi content gate after restart: PASSED (~02:45)

- Round 1 post-recharge: 50/50 completed; all 50 query records carry
  `reasoning_content`; out-tokens min 1955 / median 8191 / p90 19,613 /
  max 36,476 (heavy thinking, as the smoke datapoint suggested). Round 2 in
  flight. No parse failures.
- Measured cadences at ~02:45: Qwen ~7 q/unit/hr (first episode completions
  expected ~04:00; leg end early-to-mid afternoon incl. the 2-unit tail).
  Claude ~4 q/unit/hr (~15 min/round — slower than smoke's 6–7 min median).
  Kimi ~50 min/round effective → confirmed long pole, projected to end 07-19;
  **kimi+coord watchdog extension beyond 11:02 07-19 will be required**
  (user pre-approved extending only those two, ceiling = GCP 72h bound
  ~23:15 07-19).

## 7. Qwen token spot-check (user-requested): RED FLAG CONFIRMED (~03:40)

- First 4 completed Qwen episodes: **~70–90% of queries hit exactly
  `output_tokens=8000` with `finish_reason="length"`** (per-episode: 29/32,
  26/30, 27/31, 17/34; median = the cap; `token_truncated` stamped correctly).
  Not "self-limiting near the cap" — genuine hard truncation: Qwen's thinking
  at R1 settings blows straight through 8k on most steps.
- The two-tier premise (prior campaign: median 142 / p90 ~2815 / ~4.8% over
  4k) is off by ~an order of magnitude under R1 conditions. Plausible cause to
  investigate: R1 raised Qwen temperature 0.6 → 1.0 (`0bcfffe`, matching
  Kimi's thinking mode); the old distribution was measured at 0.6 — hotter
  sampling plus thinking = far longer traces.
- Mechanically the episodes survive: truncated queries still parse actions
  (29/29 in the sampled episode; end_reason=stalled at 32 steps). So phase-1
  data is complete-but-reasoning-truncated — exactly the population phase 2
  exists to redo.
- **Consequence / MORNING DECISION GATE:** the scanner will flag ~all 50 mazes,
  so designed phase 2 (64k, `max_num_seqs=3`, ~2–3 streams/GPU) becomes a
  near-full rerun — realistically **multi-day** GPU work that cannot fit the
  current fleet's GCP 72h bound (qwen VMs die 23:15 07-19). Options to discuss:
  (a) run designed phase 2 on fresh VMs (new 72h budget, real days of A100);
  (b) intermediate tier instead, e.g. 16–24k cap at moderate concurrency —
  needs a NEW run config (allowed post-phase-1? it's a new pass with its own
  units — but deviates from the frozen phase-2 config, user must approve);
  (c) accept truncated phase-1 as the Qwen finding ("cannot operate in 8k
  thinking budget") and skip/limit phase 2;
  (d) any of the above on a flagged SUBSET (e.g. rank by how often truncation
  changed behavior) to bound GPU time.
  No overnight action taken beyond letting free phase-1 finish (its data is
  the scanner input in every scenario).

## 8. Qwen phase 1 complete → artifacts pulled, scanner confirms 50/50, VMs stopped (~09:00)

- All 50 Qwen units verified by ~08:50 (leg wall-clock ≈ 9h at full 3×16
  fan-out — slower than the 4–5h plan, consistent with 8k-cap decodes on most
  queries).
- **Artifacts pulled** to `.runs/r1-20260717/qwen_phase1_pulled/` (50
  episode.json + run_inputs, 144M, PNGs excluded — full archives incl. frames
  remain on the coordinator's uploaded copies and the preserved VM disks).
- **Scanner dry-run (no `--out`): 50 flagged / 0 not-flagged / 0 cap-unresolved**
  — the §7 red flag is scanner ground truth now. No phase-2 manifest emitted;
  that emission is part of the morning decision.
- **Qwen A100 VMs STOPPED** (~$14/hr saved while phase 2 is undecided). Two
  operational notes: (a) `gcloud compute instances stop` now requires
  `--discard-local-ssd` on a2-ultragpu — used `=true` after verifying the local
  SSD is unmounted/unused (weights = 52G HF cache on the persistent boot disk);
  `lib/cost_safety.sh::cs_stop_vms` will silently fail on these VMs
  (`|| true` swallows the 400) — **fix post-campaign**; (b) restart for any
  phase-2 path: `gcloud compute instances start` all three, wait for ssh, then
  the chosen reload path (serve reuse guard will relaunch vLLM since servers
  died with the stop).

## Pending overnight tasks (as of ~01:30)

- **Qwen token spot-check (user request):** once the first Qwen episodes
  complete (expected ~02:00–04:00), pull per-query `usage.output_tokens` from
  episode transcripts: distribution, count > 4000, count at/near the 8000 cap,
  any `finish_reason=="length"`. Red flag = a large fraction pinned at/near the
  cap (model self-limiting its thinking because the cap is visible/small).
  Result to be logged here.
- Qwen phase boundary when all 50 qwen units verify: pull phase-1 artifacts →
  scanner dry-run (no `--out`) → emit `manifest.r1_qwen_phase2.json` →
  `reload-qwen-phase2` (re-asserts per-kind watchdog deadlines afterwards:
  qwen abs ~07-19 06:00, api/coord abs ~07-19 11:00) → phase 2 at reduced
  concurrency, parallel with the Kimi tail (user-approved sequencing).
- Kimi round-1 content gate: verify `reasoning_content` non-empty and positive
  usage when `batch_6a5aae8a…` completes (~17 min median).
