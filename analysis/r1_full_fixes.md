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

## 9. Phase 2 launched: full-64k rerun, all 50 mazes (user decision, ~11:00 07-18)

- **User call:** fairness requires the full 64k think budget even if multi-day;
  the 8k tier was only ever a performance optimization built on stale ablation
  stats.
- **KV sizing experiment (qwen-0, A100-80):** KV cache 286,476 tokens →
  2.98 worst-case streams at `max_model_len=96000`; the 96k reservation is
  padded (real need ≈ ≤2k in + 64k out), so phase 2 serves at
  **`QWEN_MAX_MODEL_LEN=72000` → 3.98 ≈ 4 streams/GPU (12 fleet-wide)**.
  Final args: 72000 / `max_num_seqs=5` (mild oversubscription, vLLM preemption
  absorbs bursts) / worker concurrency 4 per VM. Est. wall-clock 24–36h.
- **Deviations from the runbook, and why:**
  - `reload-qwen-phase2` NOT used: (a) the phase-1 coordinator still occupies
    :8765 serving Kimi/Claude, and `start_worker` hard-codes port 8765 — the
    runbook's fleet path implicitly assumed the fleet was free; (b) reload
    re-arms ALL VM watchdogs, which would have clobbered the per-kind caps.
  - Instead: **second coordinator on :8766** (same coord VM), job
    `job_21dc689df529`, 50 units, own artifacts root
    `artifacts/r1-20260717-qwen2`, `--stale-after-seconds 9000`; workers
    hand-started with `--coordinator-url http://10.128.0.64:8766`,
    `--worker-concurrency 4`, artifacts root
    `~/multinet-worker-artifacts/r1-20260717-qwen2` per VM.
  - `max_model_len` 72000 instead of the design's 96000 — env-only change on
    the manual path (`reload_gpu_worker` hard-codes 96000; NOT edited).
- Phase-2 manifest committed + pushed (`gridworld/fixtures/manifest.r1_qwen_phase2.json`,
  50 tasks) and scp'd to the coordinator (prepare reads it there; workers get
  task rows via units, so only the coordinator needed the file).
- GCP `maxRunDuration` confirmed to reset per boot → restarted qwen VMs each
  have a fresh 72h ceiling. Watchdogs: qwen 40h, coord extended to 48h
  (hosts both jobs). Kimi/claude watchdogs untouched (kimi+coord push tonight,
  sized from Claude's final step distribution, per user).
- Ops gotcha hit twice: remote commands whose literal text contains "vllm
  serve" self-match `pkill -f`/`pgrep -f` and kill their own ssh shell (exit
  255) — always feed such scripts via stdin `bash -s` (the repo's
  `reload_gpu_worker` comment documents this; it got me anyway).

## 10. Phase-2 throughput halved by baked-in `max_in_flight=6` (~13:00 07-18)

- Phase 2 ran at 6/12 streams: the phase-2 config's `max_in_flight: 6`
  (sized for the design's 2-streams/GPU pessimism) throttled assign fleet-wide.
  At the observed ~13 min per 64k query, 6 streams ≈ 60h — over the watchdog.
- **Gotcha within the gotcha:** editing `plan["models"][...]["max_in_flight"]`
  + serve restart did NOTHING — `_below_max_in_flight` reads the value stamped
  into **each unit record** at prepare (`distributed_run_pipeline.py:805`,
  stamped at :357). Fix = atomic edit of all 50 units' top-level
  `max_in_flight` → 12 in `job_plan.json` (`model_config` left byte-identical
  so `episode_inputs_hash` verification is unaffected); no serve restart needed
  (assign re-loads the plan per call). Fleet filled to 12 running within a
  minute. Backup: `job_plan.json.bak-maxinflight-*`.
- Also this window: Claude leg COMPLETE (50/50 verified: 3 solved — r1_S4_10x10_dense_1,
  r1_M1_8x8_corridor_kr_0, r1_M1_8x8_corridor_kr_1 — 45 stalled, 2 late tails;
  0/1876+ queries token-truncated; ~$31). All 50 episodes pulled locally;
  Claude VM STOPPED. qwen-2's first `instances start` failed SILENTLY
  (transient A100 capacity; output was tail-swallowed) — always check start
  output per-VM.

## 11. Afternoon 07-18: first phase-2 results, Kimi round-16 cancel, watchdog extensions

- **Phase-2 behavioral shift confirmed:** with the full 64k budget Qwen episodes
  run PAST the stall-30 wall that killed phase-1 (leaders 30+ turns and still
  progressing). First completion: `r1_B1_8x8_corridor_swg_1` — the one maze
  Qwen solved *truncated* in phase 1 — **stalled** at 39 queries under full
  thinking (out-tok median 14.7k / max 21.8k, ZERO truncation). Analysis note:
  later-pass-wins records the stall; the phase-1 solve stays in the phase-1
  root (`pass=1`). "Solved truncated, stalled with full thinking" is a finding.
  Longer episodes ⇒ phase-2 may stretch past the ~33h projection; watchdog 40h,
  re-check at midpoint.
- **Kimi round 16 cancelled at the 2h round deadline (~$2.2 paid, lost):** the
  batch's 50 requests completed but Moonshot's `finalizing` stage sat 90+ min;
  deadline+cancel fired per design; the ~50 attempt-1/3 parse failures in the
  worker log are that round's mazes retrying. Round 17 (also slow-finalizing,
  103 min) resolved `completed` in time and applied — one-off Moonshot
  slowness, not systemic; no plan surgery. Parse failures otherwise benign:
  69×1/3, 6×2/3, 0×3/3 (no episode deaths).
- **Watchdogs extended 15:54:** kimi + coord → +49h (deadline ~16:54 07-20,
  inside the immovable GCP stop 23:20 07-20). Note kimi's 71-round worst case
  ends ~midnight 07-20/21 — if the tail is that long the GCP stop wins; the
  lockstep per-round checkpoints + on-disk episodes make that recoverable, but
  flag it to the user ~evening 07-20 if >5 kimi mazes still run.
- Cost note: two consecutive slow finalizations suggest Moonshot's batch
  finalizing latency is variable at 64k-thinking scale — the `batch_deadline_s`
  knob (hash-safe by design but baked per-unit at prepare, same trap as
  `max_in_flight`) is the lever if cancels become frequent.

## 12. ⚠️ INCIDENT: Moonshot outage terminated ALL 50 Kimi episodes (07-19 ~10:45)

- Moonshot batch processing stopped completing requests ~05:45 CEST (0/50 for
  hours). Rounds 35–37 cancelled at the 2h deadline ($0 billed each). The
  runner records a cancelled round as a **parse-failure step per maze** —
  three consecutive cancels = `consecutive_failures=3` for the whole panel =
  every episode terminated at query #34 with `end_reason=parse_failed` and
  finalized "successfully" (33 real rounds ⇒ positive telemetry ⇒ scoring
  passed). Coordinator now shows the phase-1 job 150/150 verified. Checkpoints
  were cleaned up at finalize — no on-disk resume state.
- **Paid exposure:** ~$60 of Kimi rounds (33/maze) now capped by infra-caused
  "failures". Episodes pulled locally (`.runs/r1-20260717/kimi_pulled/`, 50
  episodes) + coordinator archives.
- **Options presented to user:** (1) resume surgery — synthesize
  `checkpoint.json` from each episode transcript minus the poisoned query-#34
  records, reset units verified→pending, restart worker (RECOMMENDED; needs a
  converter script = user sign-off); (2) full re-run (~$60–100, 20–40h);
  (3) accept right-censored-at-33 data (damages solve-rate science — Claude's
  solves landed at 30/35/47). All options gated on Moonshot recovery.
- **Headline post-campaign fix (now proven twice):** batch-level failure
  (rejection OR deadline-cancel) must NOT convert to per-maze parse failures /
  episode attempts. It should pause the round and retry, alerting the
  operator. This one design flaw caused both the launch-night dead-letter
  cascade and this episode-truncation incident.

## 13. Kimi resume kit built + validated (07-19 ~11:45); executes on Moonshot recovery

- Converter (`scratchpad/synthesize_kimi_checkpoints.py`): strips the exactly-3
  trailing poisoned query records per episode (fail-closed asserts: 3 stripped,
  transcript ends at a step record), rebuilds checkpoint.json per the
  `episode_checkpoint._FIELDS` schema with `query_count=31,
  consecutive_failures=0, finished=False`. All 50 converted; outputs in
  `.runs/r1-20260717/kimi_resume_checkpoints/`.
- **Validated locally**: `resume_stepper` + `build_episode_runner` on
  `r1_S5_14x14_corridor_0` → 31 queries restored, 29/29 steps replayed, clean
  boundary. task_sources pulled from the VM for the validation.
- Execution procedure (on MOONSHOT-RECOVERY + user go):
  1. scp the 50 checkpoint.json into the kimi VM run dirs; delete episode.json
     + run_score.json there (locals + coordinator archives retain the
     originals).
  2. Coordinator surgery (same pattern as launch-night): 50 kimi units
     verified→pending, attempts=0 (backup job_state first). NOTE: phase-1 job
     shows 150/150 verified until then — do not finalize/aggregate before the
     resume completes (kimi rows would be the parse_failed stubs).
  3. Restart lockstep worker (`API_WORKER_ROLE=lockstep-worker`,
     `API_WORKER_CONCURRENCY=50`); `_ensure_unit` resumes each from its
     checkpoint.
  4. Gate: first round shows query #32 issued (not #1), usage>0,
     reasoning_content present.
- Also: 23/50 qwen phase-2 done → 1 solve, 21 stalls, 1 env-truncation at 87
  steps; Qwen episode lengths (med ~47–66, max 87) ≈ 2× Claude's (med 35) —
  progress-aware stall-K rewards 64k-thinking exploration; solve conversion
  stays thin (1/23 vs Claude 3/50).

## 14. Kimi RESUMED on sync transport (07-19 ~12:20) — outage was batch-only

- Moonshot's **sync** endpoint was healthy all along (1.6s liveness call) —
  the outage was batch-infrastructure-only. Per user directive (no batch mode,
  accept full pricing), shipped commit `e0542b5`: env-gated
  `KIMI_BATCH_TRANSPORT=sync` in `KimiK26Agent.generate_batch` — bounded
  ThreadPool of the tested sync `generate()` under the same lockstep round
  contract; per-item failures isolated as `sync_error` stubs (a transport
  failure can no longer masquerade as panel-wide parse failures). 17 kimi
  tests green + behavioral checks; default transport unchanged.
- Executed the §13 resume: code-sync to kimi VM @ `e0542b5` (fail-closed
  verify), 50 checkpoints staged, stubs cleared, units reset
  verified→pending (backup `job_state.json.bak-kimi-resume-*`), worker
  restarted with `KIMI_SYNC_FANOUT=15`, concurrency 50.
- **Gate PASSED:** all 50 assigned; first sync round completed <7 min;
  checkpoints advanced 31→32 across the panel; zero parse failures. Round
  cadence now ~7 min vs ~55 (batch) — Kimi projected to finish TODAY.
  Extra sync cost ≈ $35–70 over batch for the remaining rounds.
- Ops footnotes: gcloud ssh env-prefix does not survive the hop — use the
  lib's unquoted-heredoc `printf %q` pattern for secrets; and a prior `cd` in
  a compound command left the tool shell outside the repo, so `source .env`
  silently loaded nothing — always source by absolute path.

## 15. Phase-2 ran shortest-first, not LPT (caught by user, 07-20 early AM)

- Supervisor error: I earlier claimed units were handed out "hardest-first
  (LPT)". Wrong — `_order_units_lpt` is applied only by
  `prepare_combined_job` (run-massive). Plain `coordinator-prepare` (used for
  phase 2) preserves manifest order, and the scanner's rerun manifest is a
  verbatim subset of the deliberately-ascending balanced_03 ⇒ phase 2 ran
  optimal_steps 23→106, so the four LARGEST mazes (caps 288–318) queue last —
  the anti-LPT worst-case tail.
- **Fix for future runs (order is hash-free):** LPT-order units in plain
  coordinator-prepare (reuse `_order_units_lpt`), or have scan_truncations
  emit the rerun manifest in descending optimal_steps.

## 16. ⚠️ INCIDENT #2 on Kimi: project consumption budget hit mid-round (07-20 ~05:00)

- Kimi marched rounds 32→65 on sync transport (26 more stalls + **1 SOLVE:
  `r1_M5_8x8_corridor_kr_kb_1` @ 62 queries**; 1 legitimately token-truncated
  query at 64k). At round #63 the Moonshot **project consumption budget**
  (the user's $150 account-side ceiling) tripped: cash balance fine ($54.76)
  but every call returns `exceeded_current_quota_error`. Three all-`sync_error`
  rounds → the attempt-counting flaw killed the remaining **23 episodes** at
  query #65 (`end_reason=parse_failed`; poisoned stop_reasons 69× sync_error).
  Third bite of the same design flaw (fixes §3, §12).
- Spend at trip: ~$60 batch + ~$90 sync ≈ $150 ✓ the ceiling worked as
  configured; the runner turned it into data damage.
- **Secured/prepared:** final 50 episodes pulled
  (`.runs/r1-20260717/kimi_pulled_final/`); 23 resume checkpoints synthesized
  (`kimi_resume_checkpoints_2/`, ~62 clean queries each, fail-closed asserts
  passed). Worker left DOWN (any restart just burns attempts on the quota
  wall).
- **BLOCKED ON USER:** raise the Moonshot project consumption budget (suggest
  ≥$250 total: remaining ~10-25 rounds × ~$2-3 sync + headroom). Then: push
  checkpoints → clear the 23 stub episodes worker-side → reset those 23 units
  verified→pending → restart worker (sync env) → gate on query #63 issuing.
- Deadline note: kimi+coord GCP hard stop 23:20 TODAY (07-20); at ~30-45
  min/round the 23 mazes need ≈ max(remaining) ≈ 5-20 rounds ≈ 3-12h —
  resume must start by ~mid-afternoon to be safe; watchdog 16:54 needs its
  final extension at resume time.

## Cost / VM-hours ledger (GCP-authoritative timestamps, us-central1-a)

Machine type a2-ultragpu-1g (1×A100-80GB) on-demand ~$4.70-5.10/hr.
**CAVEAT:** GPU `lastStartTimestamp` = MOST RECENT boot only. GPUs were
stopped after phase 1 and restarted for phase 2, so the pairs below are the
PHASE-2 window; phase 1 (~9-9.7h/VM, created Fri 07-17 ~23:39 CEST, stopped
Sat ~09:00) is a separate window those fields overwrote.

| VM | phase-2 start (CEST) | stop (CEST) | phase-2 h |
|---|---|---|---|
| qwen-0 | 07-18 08:39 | 07-20 02:28 | ~41.8 |
| qwen-1 | 07-18 06:54 | 07-20 12:13 | ~51.3 |
| qwen-2 | 07-18 09:10 | 07-20 04:43 | ~43.6 |
| claude-api | 07-17 23:24 | 07-18 09:59 | ~10.6 (single boot) |
| kimi-api | 07-17 23:23 | RUNNING | — |
| coord | 07-17 23:23 | RUNNING | — |

GPU-hours: phase 2 ~137 + phase 1 ~28 ≈ **165 A100-h ≈ ~$775-840**.
Higher than the mid-run ~$650-725 estimate — phase 2 ran a full ~2 days
(42-51h/VM) vs the projected ~24-36h. Token spend: Claude ~$31 (batch),
Kimi ~$60 batch + ~$90+ sync (2 resumes), Qwen $0. **A100 hours dominate
total cost — the real lever for the next run, not token discounts.**

## 17. Supervisor error: watchdog under-sized, STOPped kimi+coord mid-run (07-20 19:19)

- I armed kimi+coord watchdogs to 12h at 07:19 CEST intending "~evening" but
  12h→19:19 (mis-stated as 20:20), and the Kimi timeout-throttled tail ran
  past it. Both VMs STOPped at 19:19 with 9 mazes still running.
- **Zero data lost** — cost-safety worked as designed (STOP not delete): 41
  finalized episodes + 9 per-round checkpoints (q82) preserved on the kimi
  disk; coordinator job_state (141 verified) preserved on coord disk.
- Recovery (handoff playbook, ~20:30): restart coord+kimi VMs (coord kept
  10.128.0.64), pull all kimi data, restart coordinator-serve (:8765, stale
  9000), reset the 9 running units→pending (backup), restart worker (sync
  env) → resumed from q81-82 checkpoints, ~0 rounds lost. Re-armed 18h
  watchdogs, **verified fire-time via /run/systemd/shutdown/scheduled**
  (2026-07-21 13:34 UTC) instead of trusting mental math.
- **Lesson:** after every arm_watchdog, read back the actual scheduled
  shutdown from the VM; never rely on computed wall-clock. Applies to the
  final kimi watchdog and any future re-arm.

## 18. Reporting clarification: query_count ≠ env step_count (07-21, user-flagged)

- During Kimi's tail, two 8x8 mazes appeared to run PAST their 3×BFS caps
  (reported "step 89 / cap 84 → -5"), which looked like a broken env
  truncation. **It was a reporting error, not a bug.** I had been reporting
  `query_count` as "step" throughout live monitoring; the two counters diverge
  by exactly a maze's parse-failure total, because a parse-failed query
  advances `query_count` but executes NO env action (no step).
- Ground truth for `r1_M1_8x8_corridor_kr_1`: env `step_count=83`,
  `max_steps=84` (=3×optimal 28) — i.e. 1 step from truncation, not 5 past it.
  Its `query_count` was 89; the 5-6 gap = the 5 outage/timeout parse-failed
  queries. Env truncation fires correctly AT max_steps regardless of query
  count; a maze cannot run "several rounds past" its cap.
- **Implication for analysis:** every per-maze "step"/"turn"/"q" figure quoted
  in this doc and in live updates is `query_count`, which OVERCOUNTS true env
  steps by that maze's parse-failure count. Clean mazes: ~0-2 (negligible).
  Outage/timeout-scarred Kimi mazes: 5-6. When computing exploration-length
  stats (esp. the "explores 2-4x longer" cross-model claim), use
  transcript `env_step_count` / `state.step_count`, NOT `query_count`, or the
  outage-affected episodes will read a few steps long. `token_truncated`,
  end_reason, and solve counts are unaffected (those key off real state).
- Detail worth keeping: that maze's last action was `DONE` with
  `event_type=WRONG_DONE` (called DONE at (5,1), goal (6,6)) — Kimi declaring
  victory at the wrong cell, a qualitative failure mode for the writeup.

## 19. Final maze — infra death + Moonshot overload incident (07-21 ~late AM)

- Last unit `r1_M6_14x14_dense_kr_sg_kb_1` (hardest, opt 106) died
  `parse_failed` at env_step 130 — but the tail was 2× `sync_error` +
  1× **`engine_overloaded`** (NEW Moonshot failure mode), not a real outcome.
  133 clean queries preserved.
- Position analysis (user asked): last 30 steps hit only 3 cells
  (5,1)/(5,2)/(6,2); last 8 pinned at (6,2) hammering MOVE_FORWARD→BLOCKED,
  TOGGLE→NOTHING — wedged at a door it lacks the key for (task kr_sg_kb, holds
  only red). BUT only ~7 steps into the 30-step stall window when infra killed
  it — NOT a legitimate stall. User: resume it (7 ≠ stall).
- Salvage staged: checkpoint q133 pushed, stub cleared, unit reset
  verified→pending (backup). BUT **Moonshot is currently overloaded**
  (engine_overloaded on a trivial call) — worker start HELD behind a
  moonshot-recovery monitor; fires + starts worker when sync endpoint clears.
  Worker had exited (normal, all 50 done) — restart needed on recovery.
- Do NOT finalize until this unit completes (149/150 verified, 1 pending).

## 20. Final maze root cause = OUR 600s timeout, not model overrun (07-21, user-gated)

- User bar: token-cap truncation (model spends all 64k thinking, no move) is a
  fair enforced outcome; a timeout from insufficient client-side budget vs
  Moonshot's (new-model-launch) slow infra is on us and must be fixed. Gate:
  confirm no trace exceeds 64k first.
- **Confirmed:** across this maze's 132 completed turns, max output = 44,167
  tokens (median 18k, p90 29k), ZERO at/over 64k, ZERO finish=length / token_
  truncated. The 2 failures were pure `sync_error` timeouts on valid sub-64k
  replies Moonshot was just slow to return. Not model overrun — our timeout.
- Fix (commit `9aca849`): `KIMI_TIMEOUT_OVERRIDE` env raises the per-call socket
  timeout (hash-safe, doesn't touch model_config). Resumed at 2400s. Also
  re-synthesized the checkpoint with `parse_failures=0` — verified ALL 3 counted
  failures were infra (`sync_error`/`engine_overloaded`), genuine model parse
  failures = 0, so the 4/3 termination was 100% infra-caused.
- Restarted worker (sync, fanout 15, timeout 2400); unit resuming from q133.
- **Post-campaign:** the parse-failure terminator must not count infra
  (batch/sync/overload) failures toward the model's parse-retry budget — same
  root flaw as §3/§12/§16, now also manifesting via timeout. And default the
  Kimi timeout higher than 600s for deep-thinking runs.

## 21. THE fix: infra failures no longer terminate episodes (commit 8b8e7e5, 07-21)

- The recurring villain (§3/§12/§16/§20) fixed at the source:
  `EpisodeStepper.apply_reply` counted ANY empty-action reply as a model parse
  failure, so transport/provider failures (`sync_error`, `batch_expired`,
  `batch_errored`, `engine_overloaded`) spent the model's parse-retry budget and
  terminated valid episodes on provider flakiness. Now `_INFRA_STOP_REASONS` are
  excluded from the terminator and retried next round. **Token-cap truncation
  (finish_reason=length / token_truncated) is deliberately STILL counted** —
  exhausting 64k is the enforced model outcome per user. Full suite green (1020).
- Synced to kimi VM @ 8b8e7e5. Final maze resume is HELD behind a
  **sustained-window monitor** (3 consecutive healthy Moonshot pings, not 1 —
  the single-ping recovery re-failed last time) per user ("wait for a bit of a
  window"). On fire: restart worker (sync, fanout 15, KIMI_TIMEOUT_OVERRIDE=2400)
  from the pf=0 checkpoint at q133; episode can now only end on a legit outcome
  (stall / solve / real 64k token-cap).
- Final maze context (user asked): still wedged at position (6,2) — never moved
  off it (env_step 130, MOVE_FORWARD→BLOCKED at a door it lacks the key for,
  ~8/30 into the stall window). Expected to legitimately STALL ~22 steps after
  resume unless it finds the missing key.
- Watchdogs: kimi+coord 15:34 CEST 07-21 — re-arm/extend at resume time.

## 22. Watchdog fired AGAIN mid-recovery — extend proactively, not last-minute (07-21 15:34)

- 2nd watchdog-timing miss. The crash-recovery 18h arm fired at 15:34 CEST while
  I was waiting for the Moonshot sustained-window; I tried to extend only AT
  15:34, but the OS had already begun `shutdown -h` and refused SSH logins
  ("System is going down"), so the extend + `shutdown -c` were impossible.
  coord + kimi STOPped (data preserved, new code 8b8e7e5 already synced).
- **Rule (reinforced):** extend watchdogs with WIDE margin the moment a long
  wait begins — never let a blocking wait (Moonshot window, slow tail) run up
  against a watchdog deadline. Re-arm as soon as a new open-ended phase starts,
  not when the deadline is imminent. (First miss: §17.)
- Recovery in flight: wait for TERMINATED → restart coord+kimi → re-arm 12h+
  (verify via /run/systemd/shutdown/scheduled) → coordinator-serve :8765 stale
  9000 → reset final unit pending → restart worker (sync, fanout 15,
  KIMI_TIMEOUT_OVERRIDE=2400, code 8b8e7e5 with the infra-failure fix) → resume
  from q133 checkpoint.

## BUG (user-flagged, 07-21): transcript step position is (row,col), should be (x,y)

- `EpisodeStepper` step records log `position_before`/`position_after` in
  **(row, col)** (they equal `state.position_row_col`), while the co-located
  `state.agent_position` is **(x, y)**. Confirmed on the final maze: step
  `position_after=[6,2]` == `position_row_col=[6,2]`, but
  `state.agent_position=[2,6]`. User: the transcript should be (x,y) to match
  the spec/solver convention (CLAUDE.md) — the row,col logging is a bug.
- **Impact:** any position read from `position_before`/`position_after` (in this
  fixes doc, my live commentary, and downstream trajectory analysis) is
  TRANSPOSED vs spec coords. Use `state.agent_position` for (x,y), or transpose.
  My first hand-render of the final maze placed the agent at spec (6,2) — wrong;
  true cell is spec (x=2,y=6), a pre-red-door corridor consistent with the agent
  still holding the (unconsumed) red key. Pulled the ACTUAL query-136 frame to
  confirm (agent = blue triangle, left edge mid-height, facing EAST into wall).
- **Fix (post-campaign):** log step positions in (x,y) — or rename the fields to
  make the row,col convention explicit — and audit any analysis that consumed
  `position_after` as (x,y).

## FINAL MAZE RESOLVED (07-21 eve) — 150/150 COMPLETE

- `r1_M6_14x14_dense_kr_sg_kb_1` finalized `end_reason=parse_failed`* at
  q147/env_step 143/pos (6,8). **ASTERISK: this is an INFRA outcome, NOT a model
  failure.** It was actively PROGRESSING — solved the red-key→red-door stage
  (kick1 got it through: TOGGLE OPENED, key consumed, `open_doors=['DR']`) — then
  the Moonshot stream hung (bytes_received frozen; server-side stall behind
  Cloudflare) every ~28min. Kick budget (3) exhausted per user; kick2/kick3
  re-hung at q147 without advancing. Per user pre-commit, terminated as a
  parse-failure-type outcome (forced checkpoint finished + end_reason, worker
  finalized: scored + uploaded).
- **For analysis: exclude/asterisk this episode's end_reason** — it is
  Moonshot-launch-instability truncation after real sub-goal progress, not a
  stall or model parse failure. It reached the red door (1 of ~4 chain stages),
  the deepest single-episode progress of the run for Kimi.
- **RUN IS 150/150 verified.** Remaining: pull final artifacts, merge_two_tier
  (qwen p1+p2), finalize/aggregate to exactly 150 rows, STOP VMs, egress+publish
  on user ask.

## FINAL-MAZE KICK BUDGET (user decision, 07-21 eve) — HARD LIMITS

- Moonshot stream hangs ~every 25-30min; each kick (kill worker + reset unit
  pending + restart from checkpoint) breaks it and has yielded real progress
  (solved the red door). Allowed BECAUSE it is genuinely progressing.
- **Kick budget = 3 total. Used: 2** (kick1 → through red door; kick2 → current
  worker, running from q147). **ONE kick remaining (kick3)** for the next hang.
- **After kick3: STOP kicking.** If it hangs again, treat as terminal — a
  parse-failure-type / infra outcome (asterisk). If it STALLS naturally, it
  stalls. Do NOT kick a 4th time.
- **HARD DEADLINE: the ~04:38 CEST watchdog.** If it has NOT gotten through the
  GATE (switch→gate stage, after red door / before blue key) by then, KILL it
  (let it finalize as-is). Through-gate-by-4AM is the bar for any further
  investment.
- Chain remaining: red door DONE → find switch (s1 @ (1,9)) → gate g1 @ (10,11)
  → blue key kB @ (12,12) → blue door DB @ (11,1) → goal (12,1).

## Operator flags (user decisions, 07-21)

- **TEMPORARY: `KIMI_TIMEOUT_OVERRIDE=2400`** on the final-maze resume is a
  stopgap for Moonshot's new-model-launch slowness. **Remove after R1** — the
  proper default-timeout fix belongs in run_config/KimiK26Config, not an env
  override carried forward silently.
- **PRE-COMMITTED CALL on the final maze (`r1_M6_14x14_dense_kr_sg_kb_1`):** if
  it reaches **10 stalled steps** (no new tile) AND Kimi infra goes down again
  as we approach VM end-of-life, we **call it a STALL** (record with an
  asterisk noting the early call). Rationale: it is repeating MOVE_FORWARD into
  a BLOCKED door at (6,2); 10 identical no-progress steps → overwhelmingly
  likely to repeat to the full stall-K=30. Do not burn more VM hours / fight
  more Moonshot outages for the last ~20 mechanical steps. The asterisk keeps
  it honest in analysis (early-terminated stall, not a natural stall-K fire).

## 🔴 MAJOR BUG (user-caught, 07-21): image_only does NOT render door open/closed state

- User noticed the opened red door still looked closed in the frames. VERIFIED
  by pixel forensics: door cell xy(6,8) in query-136 (env_step 132, door CLOSED,
  open_doors=[]) vs query-145 (env_step 141, door OPEN, open_doors=['DR'], key
  consumed) is **PIXEL-IDENTICAL** — mean abs diff 0.0, same 'locked door' glyph
  (dark-red fill + bright border + keyhole dash) in BOTH states.
- **The image_only observation never encodes door open/closed.** A model that
  correctly opens a door sees an identical frame and cannot perceive that its
  TOGGLE succeeded — it can't visually distinguish a passable open door from a
  blocking closed one.
- **Severity: HIGH — run-level confound for ALL image_only episodes (the entire
  R1 run).** Mechanism-interaction tasks (kr/sg/kb chains, doors, gates) are
  fundamentally handicapped: the agent must track door state in its head with no
  visual feedback. This likely depresses solve rates across all 3 models and
  interacts with the rotation off-by-one below. Text-summary arms partially
  mitigate (feedback text), but image_only mechanism results are suspect.
- **BUT models DID solve mechanism tasks despite it (validity check):** of the 6
  total solves, 4 were key-door mazes — Claude on M1_8x8_corridor_kr_0/kr_1 and
  M1_10x10_corridor_kr_1 (red key+door each), and **Kimi on M5_8x8_corridor_kr_kb_1
  which required opening TWO locked doors (red+blue) with no visual confirmation.**
  Only 2 solves were plain navigation. So the bug DEPRESSES mechanism solve rates
  but does not fully block them — models that track door state mentally can still
  solve. Mechanism results are depressed-but-real, not noise. Strengthens the case
  a fixed-renderer re-run would raise solve rates.
- **BROADER THAN DOORS — it's a general dynamic-mechanism render gap (07-21
  deeper check):** `Switch.render()` (custom_env.py:101) draws a fixed-color
  circle with NO branch on `is_active` → **switch on/off renders identically
  (code-confirmed)**. `Gate` extends `Door` (custom_env.py:119), and doors
  render identically open/closed (empirically confirmed above) → **gates almost
  certainly don't render open/closed either.** So image_only hides the runtime
  state of doors AND switches AND (by inheritance) gates — the agent sees only
  the INITIAL mechanism glyphs, never their post-interaction state. This makes
  the confound systemic across the whole mechanism taxonomy, not just doors.
- **Only 1 row in the final 150 is infra-asterisked:** kimi
  `r1_M6_14x14_dense_kr_sg_kb_1` (`end_reason=parse_failed`, the Moonshot-outage
  termination). All resumed/salvaged episodes completed with legit end_reasons
  (46 stalled / truncations / solves per model). Final solves = 6 (Claude 4:
  S4 nav + M1_8x8_kr_0/kr_1 + M1_10x10_kr_1; Kimi 1: M5_8x8_kr_kb; Qwen 1: S4
  nav) — scorer-authoritative (my mid-run "Claude 3" predated a late finisher).
- **Repro note for the post-mortem:** a bare `load_task_from_file(spec).reset()`
  showed only Wall+Goal in the grid (mechanisms not placed via that path) — use
  the runtime's full build path to reproduce the render bug, or drive an actual
  episode. Pixel-forensics method that proved it: crop the mechanism cell
  (cell_px = frame_w // grid_w) from a pre- vs post-interaction frame and diff.
- **Post-mortem actions:** (1) fix the renderer to draw open doors distinctly
  (MiniGrid convention: hollow/outline for open). (2) re-examine whether other
  dynamic state (gates open/closed, switch on/off) is rendered — likely the same
  bug class. (3) caveat all image_only mechanism-task results in the writeup.
  (4) consider whether this + rotation off-by-one warrant an image_only re-run.

## ⚠️ RUN-LEVEL ASTERISK (user, 07-21): possible off-by-one rotation bug (POST-MORTEM)

- User read the actual query-136 frame and concluded **Kimi is "off by 1 on its
  rotation"** — the agent's rotation/facing appears offset by one, which would
  explain the wall-ramming navigation failures. User suspects a **regression of
  a rotation bug we believe was fixed previously**.
- Concrete symptom in hand for the post-mortem: agent state `facing=EAST` at
  spec (x=2,y=6); it issues MOVE_FORWARD and hits BLOCKED (wall at (3,6), due
  east). Open neighbors are N (2,5) and S (2,7). If facing render/semantics are
  rotated by one, the model's chosen action maps to a different real direction
  than intended → systematic navigation failure that is partly ARTIFACT, not
  model limitation.
- **Impact:** if real and general, this puts a small asterisk on the WHOLE run's
  navigation/solve interpretation (all image_only episodes), not just this maze.
  Interacts with the coordinate (row,col vs x,y) bug above — check whether the
  facing-vs-frame or the action→direction mapping is where the offset lives.
- **Decision: DEFERRED to post-mortem. Nothing to do now but watch** (user).
  Do NOT investigate/fix mid-campaign.

## ACTION ITEM (user, 07-21): document exact timestamps of all outages

- Compile a precise timeline of every infrastructure outage/incident during R1
  with UTC+CEST timestamps and durations. Sources: coordinator/worker logs,
  Moonshot batch `created_at`/`failed_at`, VM `lastStart/lastStopTimestamp`,
  and the incident sections here (§3 launch balance, §12 first batch outage,
  §16 quota/consumption trip, §19-20 final-maze engine_overloaded + timeouts,
  §22 watchdog stops, plus the ~3h hung-stream wedge on 07-21 eve). Purpose:
  quantify how much wall-clock the Moonshot new-model-launch instability cost
  the run, and support the post-mortem's infra-resilience recommendations.
  (Not now — post-campaign.)

## Post-run analysis questions (user-queued)

- **Claude's thinking depth vs `effort: xhigh`:** median output was only ~442
  tok/turn (27% of turns ~24 tok, no thinking block) despite xhigh adaptive
  thinking — vs Kimi ~18–20k and Qwen ~15k per turn. Is adaptive thinking
  genuinely deciding these moves are trivial, or is the image_only prompt
  failing to trigger deliberation (and would effort have engaged under
  text_summary)? Cross-check thinking depth vs step outcome (BLOCKED repeats
  vs novel moves) and vs the exploration-length gap: Claude stalls at median
  35 while the heavy thinkers run 50–130+ — thinking depth correlates with
  exploration length across all three models but NOT with solves (Claude 3,
  Kimi 1, Qwen 1).

- **How did Qwen reach env-truncation without solving or stalling?** (3
  episodes ran the full step budget — up to 132 queries — never tripping
  stall-K.) Characterize those trajectories: were stall-counter resets driven
  by genuinely novel tile visits (systematic exploration of large mazes), or
  does the progress signature admit "wandering" that keeps discovering new
  cells without ever converging on the goal (i.e., can the per-tile progress
  rule be gamed by non-goal-directed coverage)? Compare coverage-over-time vs
  distance-to-goal-over-time for the env-truncated vs stalled vs solved
  episodes (`r1_M3_8x8_corridor_kr_sg_1` 87q + the two 105q/132q cases vs
  `r1_S4_10x10_dense_1` 69q solve). Bears directly on the per-tile step-cap
  redesign planned for the next iteration.

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
