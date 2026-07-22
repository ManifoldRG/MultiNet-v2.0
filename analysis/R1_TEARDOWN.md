# R1 (`r1-20260717`) Post-Mortem Teardown

**Author:** post-mortem branch `postmortem/r1-teardown` · **Date:** 2026-07-22
**Scope:** bug-squashing teardown of the R1 production run — root-cause every
defect by reproduction, fix it with a regression test, remove the one-time
operational hacks, and confirm (not theorise about) the open behavioural
questions. This document is the single source; it supersedes the scattered
claims in `analysis/r1_full_fixes.md` where they conflict (each such correction
is called out explicitly).

Companion checklist: `analysis/POSTMORTEM_TODO.md` (P0–P3). Every item there is
resolved or dispositioned below (see §9 status table).

---

## 0. Executive summary

Two real code defects, one confirmed-benign behaviour, and two refuted alarms.

| # | Item | Verdict | Fix / status |
|---|------|---------|--------------|
| 1 | **Switch dynamic-state render** | **REAL BUG** (since 2026-04-25) | Fixed `Switch.render`/`encode` + 4 regression tests |
| 2 | **Transcript `position_*` = (row,col)** | **REAL BUG** (since 2026-05-22) | Renamed to `position_*_row_col` + 3 regression tests |
| 3 | Door/gate render "systemic bug" | **REFUTED** | Reproduced: doors & gates render state correctly |
| 4 | Rotation off-by-one | **REFUTED** | Reproduced: facing↔move↔render consistent; guard test added |
| 5 | Claude thinking-ON/xhigh | **CONFIRMED CORRECT** | Config uniform, thinking present, adaptive — no code bug |
| 6 | Render bug run-level confound | **~ZERO measured impact** | 0/50 mazes/model solve-relevant-affected |
| 7 | `KIMI_TIMEOUT_OVERRIDE` env hack | **REMOVED** | Folded into run_config (`timeout: 2400`) + guard test |
| 8 | Qwen two-tier (8k→64k) | **DROP the tier** | Commit to single 64k (or data-driven 32k) stage |
| 9 | **image_only frames absent from published bundle** | **DATA-INTEGRITY GAP** | Frames (the raw observation input) exist only on VM/coordinator archives; re-egress with frames + completeness gate (§8a) |

Test suite after all changes: **1029 passing** (was 1020; +9 new regression
tests, 0 regressions).

---

## 1. BUG #1 — image_only never rendered switch on/off state

### 1.1 What the bug is
`Switch.render()` drew a fixed-colour circle with **no branch on `is_active`**,
so a switch looked pixel-identical whether on or off. `Switch.encode()` also
omitted `is_active`, so even MiniGrid's tile cache (keyed on `encode()`) would
not refresh on a state change. In an `image_only` observation the agent could
not visually tell whether its TOGGLE flipped a switch.

### 1.2 When it entered the codebase
- **2026-04-25** — the gridworld v2.0 core (`ad6091c`/`79f72a5`) introduced the
  `Switch` class with **no `render()` override**, inheriting `Ball.render()` (a
  plain circle, no state). So switch state was invisible from day one.
- **2026-05-01** (`4c0ea41`, "Preserve switch colors across backends") added an
  explicit `Switch.render()` — but still **without an `is_active` branch**.
- Present for ~2.7 months, through the entire R1 run.

### 1.3 Root cause, by reproduction (not inference)
Built the real runtime env (`TaskSpecification.from_json` →
`TaskParser.parse` → `env.render()`) and diffed the mechanism cell before/after
a state change:

| Mechanism | mean abs pixel diff | verdict |
|---|---:|---|
| Door open↔closed (r1_M5_8x8_kr_kb, Kimi's solve) | **46.8** | renders correctly |
| Gate open↔closed (r1_M2_14x14_sg_0) | **17.4** | renders correctly |
| **Switch on↔off** | **0.0** | **invisible — the bug** |

The single confirmed defect is the switch glyph. (A 19.5 "switch diff" seen in
one intermediate check was **agent-triangle contamination** — the agent standing
on the switch cell — not switch state; the clean isolated diff is 0.0.)

### 1.4 Correction to the incident log (important)
`analysis/r1_full_fixes.md` §MAJOR BUG and commit `0fbde63` claimed the render
gap was **"systemic across doors + switches + gates … a run-level confound for
ALL image_only episodes."** **That is refuted.** Doors and gates render their
open/closed state correctly through the production path. The incident's
door-frame forensics were done on the final maze
`r1_M6_14x14_dense_kr_sg_kb_1` — a pathologically checkpoint-resumed episode
whose frame PNGs were excluded from the local pull, so the exact frames are not
re-inspectable. Resume is deterministic replay (re-seed + replay actions), so a
toggled door **is** reopened on the rebuilt grid and would render open; there is
no resume artifact. The most likely explanation for the diff-0.0 observation is
a frame-selection error compounded by the query_count≠env_step_count confusion
(§18) and the (row,col)/(x,y) coordinate bug (BUG #2) — i.e. a measurement
artifact, not a renderer defect. The switch gap is the only true render bug.

### 1.5 Measured effect on R1: ~zero
A per-episode census over all 150 image_only episodes (parallel forensic agent):
- **31 door mazes/model.** 15 doors were actually opened (Claude 4, Kimi 7,
  Qwen 4); **all 15 were subsequently traversed.** Zero "opened door then stalled
  at it." All 11 TOGGLE-at-door events that did *not* open had `carrying == None`
  (no key) — correct engine refusal. The door mechanic + render are sound.
- **35 switch mazes/model.** **Every switch in R1 is `switch_type == "toggle"`**
  (zero one_shot, zero hold), and every solve-relevant switch (`s1 → g1`) has its
  gate `g1` co-present. Because the **gate renders correctly**, the model gets
  visible feedback that the toggle worked. The switch's own invisible glyph never
  determined an outcome. The only truly-invisible switches are the `inactive_sb`
  **decoy** switches in the 7 D2 mazes/model — they control nothing.
- **Bottom line, per model:** solve-relevant mazes affected by the render bug =
  **0**. Door-scope (the original hypothesis) = **0**. The dominant switch-maze
  failure mode was models never *reaching* the switch (`toggled == False`) — a
  navigation/planning failure, not rendering.

So the render bug is a genuine correctness defect worth fixing, but it did **not**
depress any R1 solve rate. The P0 "caveat ALL image_only mechanism results"
concern is largely moot; see §9.

### 1.6 The fix
`gridworld/custom_env.py`:
- `Switch.render()` now draws a state core: **bright white when active, dark when
  inactive**, over the coloured circle.
- `Switch.encode()` now encodes `is_active` in the state field (`+2`) while
  preserving the custom-colour low-bit flag, so the tile cache refreshes on a
  state flip.

### 1.7 Regression tests — `tests/test_render_mechanism_state.py`
`test_switch_render_changes_with_is_active`,
`test_switch_encode_reflects_is_active`,
`test_door_render_changes_with_open_state`,
`test_gate_render_changes_with_open_state`. The door/gate tests also **pin the
ground truth** so a future "systemic render bug" claim can be checked in seconds.

---

## 2. BUG #2 — transcript step positions logged as (row,col), named as if (x,y)

### 2.1 What the bug is
`EpisodeStepper` step records logged `position_before` / `position_after` in
**(row, col)** (they equalled `agent_row_col(state)`), while the field name and
the co-located `state.agent_position` are **(x, y)**. Any downstream analysis
that read `position_after` as (x, y) got **transposed** coordinates. This bit the
live operator: their first hand-render of the final maze placed the agent at spec
(6,2) instead of the true (2,6).

### 2.2 Scope: analysis-facing only — the model was never misled
All prompt/text_summary rendering is internally consistent in (row, col) via
`interface/coords.py` (`to_row_col`, `agent_row_col`, `goal_row_col`): goal,
keys, doors, switches, gates, current pose, start-pose anchor, and the movement
trail are **all** (row, col) and **all labelled** row/col. There is **no
model-facing coordinate bug** — the model saw a coherent world. The defect was
purely the ambiguous transcript field name colliding with the (x,y) convention of
`agent_position` / the spec / the solvers (CLAUDE.md invariant).

### 2.3 When it entered the codebase
- **2026-05-22** (`210d4b1`, "Add interfacing with backend") — first logged
  `position_before`/`position_after` via the (row,col) helper.
- **2026-07-16** (`8c8093f`) — moved verbatim into `interface/episode_step.py`
  during the stepper-seam refactor (no semantic change).

### 2.4 The fix
Renamed the ambiguous fields to **`position_before_row_col`** /
**`position_after_row_col`** across the writer and every consumer
(`episode_step.py`, `observation.py`, `episode_checkpoint.py`, `scorer/runtime.py`
and their tests). Semantics are unchanged (the prompt path and checkpoint-replay
validation both depend on (row,col)); the authoritative (x, y) remains
`state_after.agent_position`. Chosen over "switch to (x,y)" because (row,col) is
load-bearing in the prompt + replay paths — the rename removes the ambiguity that
caused the harm at the source without touching those contracts.

### 2.5 Audit note for downstream analysis
Any prior R1 trajectory analysis that read `position_after` as (x,y) is
transposed. Re-read via `state_after.agent_position` (x,y) or transpose. The
forensic census in §1.5 was done on `agent_position`, so it is unaffected.

### 2.6 Regression tests — `tests/test_transcript_coordinates.py`
`test_step_position_field_is_explicitly_row_col` (new field == `to_row_col(xy)`;
bare `position_after`/`position_before` keys are **gone**),
`test_row_col_field_is_transpose_of_xy`.

---

## 3. REFUTED — "rotation off-by-one" (§RUN-LEVEL ASTERISK)

Reproduced across all four facings: FORWARD moves exactly one cell in the
matching (x,y) direction (EAST→+x, SOUTH→+y, WEST→−x, NORTH→−y), the facing
label matches `_DIR_TO_FACING`, and the rendered agent triangle points the right
way (EAST-facing centroid at x≈0.35 → apex right/east). **No off-by-one exists.**
The operator's suspicion was the same transposed-coordinate misread as BUG #2:
reading the agent at (6,2) instead of (2,6) makes an EAST-facing MOVE_FORWARD
into a wall look like a rotation error. Guard test added:
`test_facing_forward_movement_has_no_off_by_one`.

---

## 4. CONFIRMED — Claude was thinking-ON at effort xhigh (no code bug)

The task was to *confirm the code/config is correct* (not to theorise about *why*
depth is shallow — that is explicitly deferred). Verdict: **CONFIRMED, no bug.**
The `086a2e1` stripped-config regression (uniform ~24-token, thinking-OFF) did
**not** recur. Evidence across all 50 Claude episodes / 1,876 queries
(`claude_pulled`):

1. **Config uniform:** every `runtime_model_config` is byte-identical —
   `{"effort":"xhigh","enable_thinking":true,"max_tokens":64000,…}` (1 distinct
   config over 50 files). This is the real fleet-worker path, not the stripped
   smoke path.
2. **Thinking present:** 1,584/1,876 (84.4%) turns carry a non-empty,
   readable-plaintext thinking block (median 330 chars, max 18,190). Not redacted
   on this run, so the 292 blockless turns genuinely had no thinking.
3. **Stop reasons:** 100% `end_turn`, **0** `max_tokens` — zero truncation.
4. **Output shape is a broad long tail** (median 289, mean 1,097, max 24,723; 376
   turns >1,000 tok), **not** a flat ~24-token spike. The smoke bug had a ~24
   spike with nothing above; here there is a ~24 spike **and** a long right tail.
5. **Decisive cross-tab:** perfect separation — all 285 low-output (≤30 tok) turns
   have **zero** thinking; all 1,584 thinking turns produce >30 tok. 36/50
   episodes are **mixed** (heavy-thinking and skip turns under the same
   thinking-ON config). A config/build bug cannot produce that mix; only the model
   choosing to skip thinking on trivial turns can.

Conclusion: the low mean (~1,097 out-tok/query total; ~442 median/turn in the
incident's first-100 window) is **genuine adaptive behaviour**, not a code
defect. *Why* it under-thinks (image_only prompt vs "maze looks easy") is a
behavioural question for the next iteration, out of scope here.

---

## 5. Token & runtime census (per model × maze)

Source: per-query `query.json` (`usage`, `llm_latency_s`) under the authoritative
pulls, reconciled **exactly (diff 0)** against
`Multinet-v2-results/r1-20260717/R1_FINAL_episode_runs.jsonl`. Full per-maze
tables: `scratchpad/postmortem_tokens.md`.

| model | total tokens | input | output | queries | mean out/query | out min/med/p90/p99/max |
|---|---:|---:|---:|---:|---:|---|
| **Claude Opus 4.8** | **3,835,785** | 1,776,799 | 2,058,986 | 1,876 | 1,097 | 17 / 289 / 3,062 / 12,317 / 24,723 |
| **Kimi k2.6** | **69,207,190** | 2,775,851 | 66,431,339 | 3,207 | 20,714 | 0 / 20,112 / 33,246 / 45,256 / 64,000 |
| **Qwen3.6-27B (phase2, FINAL)** | **50,032,265** | 2,015,488 | 48,016,777 | 3,045 | 15,769 | 33 / 16,444 / 22,398 / 26,843 / 64,000 |
| Qwen3.6-27B (phase1, 8k) | 20,361,008 | 1,644,520 | 18,716,488 | 2,472 | 7,571 | 26 / 8,000 / 8,000 / 8,000 / 8,000 |

**Headline: Kimi burns ~18× Claude's tokens** and ~19× its mean output/query.
Qwen phase-1's flat 8,000 across p90/p99/max is the cap-truncation that forced
the 64k rerun (§6).

### Runtime — summed `llm_latency_s` (aggregate model-wait, **NOT** wall-clock)
Calls ran concurrently (Kimi batch/sync fan-out 15–50; Claude via Batch API;
Qwen 3× A100 in parallel), so these sums are an upper-bound wait ledger.

| model | summed latency | (h) | queries >600 s |
|---|---:|---:|---:|
| Claude | 1,003,575 s | 278.8 | 460 (Batch-API turnaround, not compute) |
| **Kimi** | 7,824,206 s | 2,173 | **2,999 (infra-slow / 600s-timeout tail)** |
| Qwen p2 | 1,801,896 s | 500.5 | 1,582 (legit 64k local decodes) |
| Qwen p1 | 772,763 s | 214.7 | 0 |

### Kimi real wall-clock
The 2,173 h sum massively overstates elapsed time. Real Kimi span ≈ **3.5 days
(07-18 → 07-21)**, dominated by outages and stop/resume gaps: batch cadence
~50–55 min/round, sync cadence ~7 min/round (later 30–45 min under load), ~65–70
rounds. Three infra incidents (all the *same* "infra-failure counted as a
parse-failure" flaw: launch-night balance precheck; 07-19 batch outage killing
all 50 at q34; 07-20 $150-budget trip killing 23 at q65), plus a watchdog
mis-fire and the final-maze 600s-timeout death. Spend: Kimi ≈ $60 batch + $90+
sync ≈ the $150 ceiling; Claude ~$31; Qwen $0 tokens. **~165 A100-h (~$775–840)
dominate total run cost — not tokens** (the real lever for the next run).

---

## 6. Qwen two-tier (stage 1 / stage 2) — DROP the tier

Full analysis: `scratchpad/postmortem_qwen.md`.

**Does Qwen use `max_tokens` to guide its thinking effort? No — it is a pure
ceiling.**
- Phase 1 (8k cap): **89.9%** of queries hard-truncated at exactly 8,000 tokens;
  **49/50** mazes had their *median* pinned at the cap. Phase-1 data is
  scientifically unusable (≈90% of reasoning cut mid-thought).
- Phase 2 (64k cap): natural median **16,444** tokens; only **0.03%** (1/3,045)
  reach 64k. Qwen does **not** inflate toward the larger budget.
- Same-maze test: all 50 mazes have a phase-2 median (12.6k–19k) far below 64k
  and far above 8k; **90.3%** of phase-2 queries exceed 8k. The distribution
  clusters at an intrinsic ~16k regardless of the cap → the cap clips, it does
  not guide. (Temperature is not a confound; both phases ran temp=1.0.)

**Runtime cost of the tier:** two-tier = ~28 (p1) + ~137 (p2) ≈ **165 A100-h**;
stage-2-only = **~137 A100-h**. The screen added **~28 A100-h (~$130–145, ~20%
overhead) for zero usable data** and still flagged 50/50 mazes → phase 2 re-ran
everything.

**Recommendation:** future runs use a **single 64k stage**. If an efficiency
lever is wanted, a **data-driven ~32k cap** covers 99.9% of queries (only 0.1%
exceeded 32k) without censoring reasoning. Any future tiering must gate on a
**live probe** of the actual model+temp+thinking config — the stale 0.6-temp
ablation under-estimated reasoning length by ~2 orders of magnitude.

---

## 7. One-time operational hacks — cleanup

### 7.1 `KIMI_TIMEOUT_OVERRIDE` env — REMOVED
- **Why it existed (2026-07-21, `9aca849`):** R1 shipped Kimi `timeout: 600`,
  too small for 64k deep-thinking under Moonshot's launch-week slow infra; valid
  sub-64k replies timed out client-side. The env raised the per-call socket
  timeout mid-run (hash-safe) as a stopgap; resumed at 2400 s.
- **Removed (2026-07-22):** deleted the `os.environ.get("KIMI_TIMEOUT_OVERRIDE")`
  read in `interface/agents/kimi_k26.py`; per-call timeout now comes **only** from
  `KimiK26Config` (set via run_config `model_config["timeout"]`). Folded the
  proven value into `gridworld/fixtures/run_config.r1.json` (Kimi
  `timeout: 600 → 2400`).
- **Guards:** `test_kimi_timeout_override_env_is_not_consulted` (env is ignored);
  `test_kimi_deep_thinking_timeout_above_600s` (deep 64k row must use >600 s).

### 7.2 `KIMI_BATCH_TRANSPORT=sync` — KEEP (decision)
The env-gated sync transport (`e0542b5`) was **load-bearing resilience**: it
finished the run after Moonshot's batch infrastructure failed repeatedly, and it
isolates per-item failures as `sync_error` stubs so a transport failure can no
longer masquerade as panel-wide parse failures. It is covered by 17 Kimi tests.
**Decision: keep it as a supported fallback — do NOT remove.** Recommended
follow-up (not done here, to keep the PR focused): promote it from an
undocumented env to a first-class `model_config["transport"]` option and document
it in RUNME. This is the opposite call from the timeout override (a genuine
stopgap) — sync transport is a feature that happened to be env-gated.

No other one-time code hacks were found in the model/agent paths (scan of
`TEMPORARY|stopgap|hack|workaround` markers and `os.environ.get` gates). The
remaining env vars (`API_WORKER_ROLE`, `SWEEP_TOPO`, `QWEN_WORKER_COUNT`,
`SMOKE_BUDGET_ACK`, `KIMI_SYNC_FANOUT`) are legitimate operational knobs, not
hacks. Live-surgery levers (`--stale-after-seconds`, per-unit `max_in_flight`
stamping, `QWEN_MAX_MODEL_LEN`) are operational, not tree hacks; their plumbing
gaps are tracked as P2 in §9.

---

## 8. THE recurring villain (context) — infra-failure terminator

Not re-fixed here (already fixed mid-run, `8b8e7e5`), but it is the single root
cause behind the launch dead-letter, both Moonshot outages, the budget trip, and
the final-maze false termination. `EpisodeStepper.apply_reply` counted **any**
empty-action reply as a model parse failure, so transport/provider failures
(`sync_error`, `batch_expired`, `batch_errored`, `engine_overloaded`) spent the
model's parse-retry budget and killed valid episodes. `8b8e7e5` excludes
`_INFRA_STOP_REASONS` from the terminator (token-cap truncation is still counted
— exhausting 64k is a real model outcome). **Verified present on this branch.**
Remaining gap (P1): the *batch-submission* path can still masquerade as per-item
parse failures — see §9.

---

## 8a. DATA-INTEGRITY — image frames missing from the published bundle

The published results bundle `Multinet-v2-results/r1-20260717/` contains, for all
150 episodes, the correct `episode.json` + `queries/*/query.json` (text, usage,
actions) + `run_inputs.json` + `run_score.json` — token totals reconcile
**exactly** with `R1_FINAL_episode_runs.jsonl` (Kimi = final-resumed, Qwen =
phase-2 `pass=2`, all match). **But every `frames/` directory is empty, and there
are ZERO decision-frame PNGs anywhere locally** (not in the bundle, not in
`claude_pulled`/`kimi_final`/`qwen_phase2_pulled`).

For an **image_only** benchmark the rendered decision frames *are* the model's
raw observation input. Their absence means:
- The bundle is **not self-contained** — no one can re-inspect what a model
  actually saw. This is precisely the wall that blocked re-verifying the render
  bug forensics on the final maze (§1.4).
- The frames survive **only on the coordinator's uploaded archives and the
  preserved VM disks** (per `r1_full_fixes.md` §8, PNGs were excluded from every
  pull to save space).

**Action (owed):** before those VM disks are reclaimed, pull the frame PNGs from
the coordinator uploads / preserved disks and re-egress the bundle with `frames/`
populated (or ship a separate frames archive keyed by `raw_output_ref`). Add a
"frames present" assertion to the publish/egress step so an image_only bundle can
never again be published frame-less. Note the earliest snapshot of this bundle
showed *only* the aggregate JSONLs (no `runs/` tree at all) — the raw traces were
added late and still without frames; the egress step needs a completeness gate.

## 9. POSTMORTEM_TODO status (P0–P3)

**P0 — data-validity**
- image_only render bug — **FIXED** (switch); doors/gates were never broken (§1).
- Caveat image_only mechanism results — **largely moot**: measured render-bug
  impact is 0/50 per model (§1.5). Keep a one-line footnote for switch-glyph
  decoys only.
- image_only re-run — **not warranted by rendering** (0 impact). Any re-run is a
  behavioural choice, not a bug remediation.
- Make `text_summary_and_last3` the primary mechanism read — still reasonable, but
  the image_only handicap it was meant to dodge is ~nonexistent for solves.

**P1 — correctness**
- Rotation off-by-one — **REFUTED + guard test** (§3).
- Transcript (row,col)/(x,y) — **FIXED** (§2); downstream audit note in §2.5.
- Infra failures vs parse budget — **FIXED `8b8e7e5`, verified present** (§8).
- Batch-level failure masquerading as per-item — **effectively CLOSED by
  `8b8e7e5`** (traced this run). The launch-night mode (batch created → status
  `failed` → empty results map → every item stamped `batch_errored`, which is in
  `_INFRA_STOP_REASONS` → excluded from the parse-retry terminator) no longer
  burns unit attempts. Only residual: a hard `POST /batches` 4xx raises a
  `RuntimeError` that errors the whole round (loud failure, not a per-item
  masquerade); a graceful pause+retry+alert there is a nice-to-have, not a
  correctness bug.

**P2 — operational hardening** (unchanged; runbook/ops, not in this PR): provider
preflight (balance + consumption budget); full-size smoke batch;
`--stale-after-seconds` plumbing through `start_coordinator`; watchdog discipline
(verify via `/run/systemd/shutdown/scheduled`); `cs_stop_vms` a2-ultragpu
`--discard-local-ssd`; unit→pending reset after worker kill; LPT-order the rerun.
`KIMI_TIMEOUT_OVERRIDE` default — **DONE** (§7.1). Sync transport — **decided:
keep** (§7.2).

**P3 — analysis follow-ups**
- Qwen "explores 2–4× longer, solves no more" — token/latency data in §5;
  behavioural characterisation deferred (use `env_step_count`, not `query_count`).
- Progress-aware stall gaming — deferred (feeds per-tile step-cap redesign).
- Claude thinking depth vs xhigh — **code/config CONFIRMED correct** (§4); the
  *why-shallow* behavioural study is the remaining open question.
- Outage timeline + full cost reconciliation — partial in §5 (Kimi wall-clock,
  ~165 A100-h dominates); a precise UTC/CEST incident timeline is still owed.

---

## 10. Files changed on this branch

Fixes (product code):
- `gridworld/custom_env.py` — `Switch.render`/`encode` state fix.
- `interface/episode_step.py`, `interface/observation.py`,
  `interface/episode_checkpoint.py`, `scorer/runtime.py` — `position_*` →
  `position_*_row_col` rename.
- `interface/agents/kimi_k26.py` — removed `KIMI_TIMEOUT_OVERRIDE` env read.
- `gridworld/fixtures/run_config.r1.json` — Kimi `timeout: 2400`.

Tests (regression):
- `tests/test_render_mechanism_state.py` (new, 4) — switch/door/gate render state.
- `tests/test_transcript_coordinates.py` (new, 3) — coord naming + rotation guard.
- `tests/test_kimi_k26_agent.py` (+1) — env override ignored.
- `tests/test_r1_run_configs.py` (+1) — deep-thinking timeout > 600 s.
- Updated key names in `test_scoring_system.py`, `test_episode_checkpoint.py`,
  `test_prompt_observation_text.py`, `test_text_summary.py`.

Suite: **1029 passing** (+9 new regression tests), 0 regressions.
