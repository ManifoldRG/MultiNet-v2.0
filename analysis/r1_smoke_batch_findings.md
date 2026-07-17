# R1 batch-API validation smoke — findings

**Date:** 2026-07-17. **Cell:** minimal / image_only / egocentric / zero_shot /
text_summary_and_last3 / stateless / step_by_step, `progress_stall_k=30`,
thinking ON, uniform 64k cap, Claude effort=xhigh, Kimi temp=1.0.
**Runs:** original (`.runs/r1_smoke_batch`) hit a config bug and ran the batch
leg thinking-OFF; the corrected re-run (`.runs/r1_smoke_batch_v2`, Claude-only)
is the source of the numbers below. Kimi thinking-on confirmed via a single
round. See §Provenance.

---

## 1. Headline outcomes (v2, Claude thinking-on, batch)

All 5 mazes **STALLED, 0 solved** — even with full adaptive thinking at xhigh.
Same terminal outcome as thinking-off, but the *behavior* differs (§3).

## 2. Per-episode breakdown + stall diagnosis

| maze | end | queries | distinct pos | MOVED | BLOCKED | TURNED | stall_gap | out med / max | thinking |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| D1_10x10_dense_wrong_ky_kr_sg_1 | stalled | 37 | 3 | 3 | 33 | 1 | 30 | 24 / 2178 | 6/37 |
| D1_14x14_dense_wrong_ky_kr_0 | stalled | 32 | 2 | 2 | 30 | 0 | 30 | 24 / 295 | 2/32 |
| M6_10x10_dense_kr_sg_kb_0 | stalled | 35 | 3 | 2 | 32 | 1 | 30 | 214 / 780 | 10/35 |
| M5_10x10_corridor_kr_kb_1 | stalled | 83 | 12 | 21 | 21 | 7 | 30 | 371 / 5256 | 64/83 |
| S4_10x10_dense_0 | stalled | 62 | 17 | 16 | 39 | 7 | 30 | 370 / 21426 | 26/62 |

- `distinct pos` = distinct agent positions reached; `stall_gap` = steps between
  the last NEW position and episode end.
- **Every `stall_gap` = 30** → all hit `progress_stall_k=30` exactly (stall-K is
  the terminator, never max_steps).

## 3. Increase K, or straight visual stall? → mostly visual

Two clusters:

- **Straight visual stall (3/5: D1_10, D1_14, M6):** only 2–3 distinct positions,
  ~90% BLOCKED, almost no thinking. Never found a path — stuck against walls from
  the start. **Raising K does nothing** (no new position in 30 steps because the
  maze isn't being parsed at all).
- **Progress-then-stuck (2/5: M5, S4):** genuine navigation — 12 and 17 distinct
  positions, real MOVED counts, much more thinking (M5 thought on 64/83 steps).
  Each hit an impasse (a wall/door it couldn't pass) and stalled 30 steps later.
  **Raising K mostly buys more BLOCKED steps at the same impasse.**

**Verdict:** the binding constraint is **image_only maze comprehension**, not K.
Thinking helps navigation (more thinking ⇄ more distinct positions) but not enough
to solve. If the goal is solves, the lever is observation richness / prompt, not K.

## 4. Per-step token usage (n=249 queries, Claude thinking-on)

| | output tok | input tok |
|---|--:|--:|
| mean | 574 | 886 |
| median | 252 | — |
| p90 | 1,249 | — |
| max | 21,426 | — |

**cost/step ≈ $0.00939** (batch pricing: output $12.50 / input $2.50 per MTok).

## 5. 50-maze R1 cost/token estimate (Claude, batch, using this run's mean tokens/step)

Step counts from the balanced_03 panel BFS sum = **2,891** across 50 mazes
(per-maze optimal 23–106, mean 57.8).

| Scenario | Steps | Output tok | Input tok | **Batch cost** |
|---|--:|--:|--:|--:|
| Solve @ BFS | 2,891 | 1.66M | 2.56M | **~$27** |
| Solve @ BFS×1.5 | 4,337 | 2.49M | 3.84M | **~$41** |
| Solve @ BFS×3 (ceiling) | 8,673 | 4.98M | 7.68M | **~$81** |

- With stall-K capping episodes at ~30–83 steps (as observed), the *actual* Claude
  batch leg lands near the **BFS row (~$25–27)**; the ×3 ceiling should never occur
  given early termination. Sync/non-batch ≈ 2× these.
- **Kimi:** one datapoint = 16,551 reasoning tokens for a single step → ~$0.04/step
  at Moonshot batch rates (~4× Claude/step) and far slower (~17 min/round). **Qwen
  $0/token** (local). Batch economics stay Claude-primary.

## 6. Latency (cap-sizing)

| Model | Batch per-round latency | Notes |
|---|---|---|
| Claude (Anthropic batch) | **median 7 min** (5.6–11) | tight |
| Kimi (Moonshot batch) | **median 17 min** (5.6–67) | wild, 60+ min outliers |

Claude batch leg wall-clock: ~2,891-step / ~80-round longest-maze → the v2 5-maze
leg took ~8 h (longest maze ran 83 rounds). Full R1 (longest maze 106 steps) via
batch is a multi-hour-to-overnight Claude job; Kimi-via-Moonshot much longer.

## 7. Batch-vs-sync control + official projection (from `smoke_report.json`)

Run complete. **Total spend $4.52** (batch $2.34 + sync control $2.18).

- **Batch discount confirmed:** batch $2.34 vs the same tokens at sync price $4.68
  → **exactly 50% savings** (Anthropic batch). Check (d) ✓.
- **No truncation:** `token_truncated_count = 0` across all 249 queries at the 64k
  cap. Check (c) ✓ (thinking present, no cap hits).
- **Round latency (83 rounds):** median **365.8 s (6.1 min)**, p90 487 s (8.1 min),
  max 1885 s (31 min), min 92 s.
- **Sync control (S4):** 65 queries, out 76,303 tok, **$2.18** — also stalled;
  ~1,174 out-tok/query vs the batch mean 574 (sync path elicited more thinking on
  this maze, though it's a separate exploration so not a clean per-step A/B).

**Official R1 projection** (report basis = per-episode averages × 50 mazes, i.e.
the *stall-reality* not a hypothetical solve):

| | Claude batch | Claude sync |
|---|--:|--:|
| Projected R1 (50 mazes, 1 seed) | **~$23.4** | ~$46.8 |

This ~$23.4 stall-based projection sits just under the §5 "solve @ BFS" row (~$27)
— consistent, because stall-K caps episodes near the BFS step range for this panel.

**Recommended caps (from measured latency):** worst-case round ≈ `batch_deadline_s`
2 h + `batch_cancel_grace_s` 5 min; observed max round 31 min. Coordinator
`--stale-after-seconds ≥ 9000` (already specified). `MAX_RUN_DURATION` / `BATCH_CAP`
should exceed the multi-round tail: the 5-maze leg took ~8 h; full R1 (longest maze
106 steps) implies a Claude batch leg of ~10–13 h — size the VM caps accordingly,
and note Kimi-via-Moonshot is the true long pole.

## 8. Why the fast stalls? — image_only gives no action-outcome feedback

Investigated the 3 fast-stall mazes (D1_10, D1_14, M6: only 2–3 distinct
positions). The prompt was inspected verbatim (`agent_messages`, image redacted).

**The text summary is present and correct.** D1_14 query #11 carried
`Activity summary: first you navigated to (1, 2), finally you passed (1, 3)` —
which matches the env ground truth (`MOVED to (1,2)`, `MOVED to (1,3)`, then
`BLOCKED`). It is sparse only because the agent genuinely reached 2 cells.

**But the model never sees action outcomes.** The env produces rich feedback
(`BLOCKED — MOVE_FORWARD blocked by wall`, `MOVED — Moved to (1,2)`,
`TURNED — Now facing SOUTH`), yet across all queries the prompt contains **0**
mentions of block/wall/feedback/position. Root cause in `interface/observation.py`:

- For `text_summary_and_last3` + **image_only**, `history_text` returns `""`
  (line 56–61) and `current_observation_text` returns `""`, so no feedback text
  is emitted.
- The last3 is rendered as image blocks by `history_content_blocks` using
  `LAST3_USER_PROMPT["image_only_step"]` = `"Your inventory: {inventory}.\n
  FINAL_OUTPUT: {action}"` — **inventory + action only; no `position_after`, no
  `feedback`.**
- By contrast text_only/image_text use `RECENT_HISTORY_STEP` which **does** include
  `Position after: (r,c) … Feedback: {feedback}`.

This is **by design** (docstring: "image_only — PNGs + inventory/action labels,
no text history") — but it means the model is blind to whether its moves
succeeded. Claude's own thinking confirms it: *"the path ahead is clear, I'll
continue"* / *"repeated MOVE_FORWARD … makes sense … logical next action"* — it
rationalizes blocked moves as progress and repeats them until stall-K.

**So the fast stalls are feedback-starvation, not just hard vision.** The static
scene is parsed adequately (agent/keys/doors/goal identified); what fails is
self-localization + outcome tracking across near-identical frames with zero
textual feedback. **Decision for R1:** either accept image_only as a
deliberately feedback-free pure-vision condition (stalls are the finding), or add
minimal outcome feedback to the image_only last3/current step (e.g. surface
`BLOCKED`/position) — which would likely convert many fast-stalls into progress
but moves image_only toward image_text. Not a scorer/runtime bug; a
condition-design choice.

## 9. Start-pose grounding — behavioral test (negative)

Added a persistent `"You started at (r, c) facing DIR."` anchor to the text
summary (commit `fb2df4d`) and re-ran the worst fast-stall maze (M6, which
reached only 3 positions in v2) through the sync path with the anchor live:

| | end | queries | distinct pos | MOVED | BLOCKED |
|---|---|--:|--:|--:|--:|
| v2 (no anchor, batch) | stalled | ~35 | 3 | 2 | 32 |
| start-pose (sync) | stalled | 35 | **3** | 2 | 32 |

**Identical outcome.** Distinct positions stuck at 3 from q10 on while BLOCKED
climbed — 2 moves, then a wall hammered to stall-K. Knowing the start cell does
not help when the model still gets no signal a move was BLOCKED. (One earlier
crashed trajectory navigated 60+ queries, so there is real run-to-run variance;
the clean run reproduced the v2 stall exactly.) **The anchor is a low-risk
grounding aid, kept, but it is not a fix for the image_only fast-stall** — that
is feedback-blindness, which is by-design for image_only (§8). A single maze ×
few trajectories is noisy; do not over-read either direction.

## Provenance / caveats

- **Config-source bug (fixed):** the smoke driver built its batch agent from the
  stripped `plan["models"]` (no `enable_thinking`/`effort`) → the original run's
  batch leg ran thinking-OFF (24-tok stalls). The real fleet lockstep worker uses
  `unit["model_config"]` and is unaffected; the smoke fix sources the same. All
  §1–6 numbers are from the corrected v2 run.
- **Batch API verified:** Anthropic batch honors adaptive+xhigh thinking (text &
  image A/Bs); Moonshot batch honors Kimi thinking (single-round confirmation,
  no truncation at 64k). Batch mechanics work for both models.
- Token/cost estimates use this run's *stalled-episode* per-step averages; a
  genuinely-solving run could shift the per-step profile.
