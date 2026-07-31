# MultiNet-v2.0 Conditional-Eval — Consolidated Findings

> Master findings doc. Each analysis phase runs as its own sub-project (branch) and
> emits its own report; those are **merged here**. Sources:
> `analysis-validity` (VALIDITY.md), `analysis-difficulty` (DIFFICULTY.md, weights.md),
> `analysis-exploration` (EXPLORATION.md). Live findings appended by the driver.
>
> **Scope of clean data:** Claude (`claude-opus-4-8`) + Kimi (`kimi-k2.6`) across 10 clean
> cells of `cond-sweep-20260704-api` (360 episodes, 1 seed, 15 mazes). Qwen
> (`Qwen3.6-27B`) complete across `cond_massive` (thinking OFF except `baseline_thinking`)
> + `cond_prompt`. **Everything is n=15 mazes / 1 seed — treat as directional.**
>
> **`cond_prompt` is reported disaggregated** into its 3 prompt-detail variants
> (`·minimal` / `·standard` / `·verbose`), each 15 mazes/model (45 pooled), so they slot into the
> rankings alongside the other 15-maze configs; the former single `cond_prompt` row = these three
> pooled (135 episodes). Optimality figures use the stored `optimality_ratio` (a few stale per-config
> optimality numbers have been refreshed).

## 0. Overall solve rates
Claude **58.9%** > Kimi **37.2%** > Qwen **30.6%** (pooled across conditions).

## 0.5 Re-analysis 2026-07-08 — 3-model re-fit, query mining, visuals
_New pass folding Qwen fully in, mining the 41,318 per-step queries, and building the visual layer.
Code: `analysis/queries_enrich.py`, `analysis/plots.py`, `analysis/build_artifact.py`. Figures:
`analysis/figures/fig01–10`. Interactive report: `analysis/figures/report.html`._

- **Canonical set reconciles to exactly 540** (source = `artifacts-pulled/`, not the reorg backup
  `Multinet-v2-results/` which lost the `issues_/` quarantine). Buggy runs are no longer discarded —
  the loader relabels them to a **`bugged_*`** family (kept for mining, excluded from stats via
  `is_clean()`): `bugged_1_hist_multiturn_icldup` (ICL-dup: Claude 53%→**7%**), `bugged_2/3` = stale
  ctx partials (query-only).
- **SP3 re-fit, 3 models:** solve × `optimal_steps` r = **−0.837**, × `mechanism_count` r = +0.19 (≈0).
  Logistic `P(solve|model,config,difficulty)` **Brier 0.148** (was 0.168 on 2 models). Per-model
  difficulty slope/intercept: Claude −1.16/+0.34 (shallowest slope, highest floor → most robust);
  Kimi −1.59/−0.85 (collapses fastest); Qwen −1.36/−1.13 (worst floor). Ridge: `optimal_steps` −0.194
  dominates. → **fig03 (difficulty curve), fig04 (mechanism confound), fig01 (solve heatmap).**
- **Stats:** model 95% bootstrap CI — Claude 58.9 [51.7,66.1] ≫ Kimi 37.2 [30.6,44.4] ≈ Qwen 30.6
  [23.9,37.2]. Claude>Kimi (+21.6pp) and Claude>Qwen (+28.3pp) SIG; **Kimi>Qwen +6.7pp [−16,+3] NOT
  significant** — that ordering is within noise. → **fig02.**
- **`qry_subgoal` delegation RESOLVED:** the model emits `SUB_GOAL:` then `FINAL_OUTPUT: <full primitive
  action sequence>` (Claude ≤90 actions/query); the **harness does NOT pathfind**. Claude's 93% is fair
  (its own route planning + re-planning). Kimi/Qwen dump giant sequences that wall-bang. Settles A6.
  → **fig10.**
- **Token efficiency:** tok/success frontier spans 2 orders of magnitude — `qry_full_trajectory` 25k
  (champion, brittle) → `obs_image_only` **1.98M** (money pit, 78×). Per model Claude 300k < Kimi 478k <
  Qwen 631k. → **fig05.**
- **Failure taxonomy (KMeans on query-mined behavior, 230 fails):** Wanderer 116 (revisit .85, Kimi),
  Wall-banger 68 (blocked .82, Qwen), Early-parse-fail 38 (act_cardinal, Qwen), Toggle-jammer 8. →
  **fig06 (clusters), fig08 (blocked-rate).** Toggle-perseveration is systematic (13 eps ≥10 futile
  toggles, all 3 models; §10 was one anecdote). `parse_ok` does NOT decay over an episode — lowest at
  steps 1–10, stable after (format-fumbles at the start, not fatigue) → **fig09.**
- **Spatial failures (fig07):** per-step coordinates recovered for 9/10 configs. Long corridors pile the
  whole episode-mass on one cell: `s5` 16/33 die at start (12,1) never escaping; `winding` 19/33 die at
  the (18,3) dead-end 3 cells short of goal (18,6). Coordinate note: prompt tuples are (row,col).
- **World-model fidelity (Kimi+Qwen; Claude reasoning redacted, only 49 narrated queries):** mined 12,419
  reasoning traces (`analysis/beliefs.py`) comparing stated belief to ground truth. **Perception is fine,
  transformation is not:** self-state (facing/position) echo 99.96–100%, wall-map recall 95–98%, but
  **mental rotation of a planned turn chain is wrong ~46%** (Kimi 46.7% / Qwen 45.3%, ~7,800 turn-claims)
  — barely above chance. Mechanistically explains the wall-banger cluster (plan a turn → believe you face
  the corridor → actually face a wall → ram it). Rotation convention validated 100% vs 11,260 real env
  transitions. → **fig11.** Reasoning is inline prose (no `<think>` tags / reasoning field); split on the
  `FINAL_OUTPUT:`/`SUB_GOAL:` sentinel. Richest seams: Qwen `baseline_thinking` (9.6k chars), Kimi
  `qry_full_trajectory` (10k; Kimi's `baseline_thinking` is token-cap-truncated to blanks).
  - **Four-faculty staircase (fig11):** self-state echo ~100% > wall-map recall 95–98% > single-step
    translation 92–94% (into-wall on adjacent path steps: Kimi 5.9% / Qwen 8.1%, confirmed (row,col)
    convention — models do NOT mix conventions, 1025:0) > **mental rotation 53–55%**. Only rotation is
    broken; perception, recall, and translation are intact.
- **`act_cardinal` is a Qwen-only instruction-following failure:** absolute directions
  (MOVE_NORTH/S/E/W) replace ego-centric verbs. Claude/Kimi adapt (parse-OK 0.95/0.85, 40% solve); Qwen
  keeps emitting `MOVE_FORWARD` (default vocab) → parse-OK 0.46, **0% solve**. Not a navigation gap.
- **ICL-example dose-response for Claude (attractor/distractor):** example copies in context vs Claude
  solve — 0 (icl_zero_shot ablated) **80%** > 1 (hist_multiturn clean) 53% > 3 (bugged_1 ICL-dup) **7%**.
  Monotonic; the dup run grinds Claude to the cap (45→93 steps, 192k→537k tok, 14/15 truncated). Kimi
  hurt less (33→20) — example-dependent, not example-distracted.
- **Harness parser bug (scoring artifact):** the `qry_full_trajectory` parser's `_ACTIONS_RE`
  (`ACTIONS\s*:\s*(.+)`) matches the first `Actions:` ANYWHERE, so a reasoning line "Actions: MOVE_FORWARD
  x 5" preempts the real `FINAL_OUTPUT:` fallback → `parsed_actions=[]`. **28% (19/67) of full_trajectory
  queries fail this way** (Qwen worst). Qwen's `empty_room` "failure" is one — a correct plan rejected.
  full_trajectory's low solve is partly artifact. FIX: anchor the parser to the designated output line.
- **Rotation error is structured & config-dependent — but does NOT directly cause wall-banging (correction).**
  → **fig12, fig13.** Shape of the 3,575 wrong rotations: **34% inverted-turn (literal LEFT/RIGHT swap;
  Qwen 37% / Kimi 19%)**, 54% "claimed no rotation" (partly annotation-style, caveat), 12% perpendicular.
  Error rate scales with in-head planning: `qry_full_trajectory` 18% < `ctx_current` 24% < `ctx_text_summary`
  26% < `cond_prompt·verbose` 47% < `cond_prompt·standard` 50% < `icl` 52% < `cond_prompt·minimal` 53% <
  **`baseline_thinking` 61%** (more reasoning = more compounding). The three `cond_prompt` variants cluster
  at ~47–53% (same query mode).
  BUT episode-level rotation-error rate barely predicts blocked-rate (r≈+0.10) or solve (36% vs 34%) —
  **step-by-step querying re-grounds the true facing each step (~100% perception), so mis-planned rotations
  are mostly corrected before execution.** Tempers the earlier "rotation explains the wall-banger cluster":
  it's a latent reasoning-fidelity gap that bites when a route is committed, not a per-step ram cause.
- **Cost / recoverable waste (fig13):** in FAILED episodes **~46% of output tokens are spent thrashing after
  the last new-cell visit** (Qwen 54% / Claude 43% / Kimi 31%; ~7.7M tokens); successes waste <1%. A
  frozen-in-place watchdog (abort after K steps stuck in one cell, safer than no-progress which kills
  key-backtrackers) at **K=20 recovers ~23% of output tokens for a 2% success loss** (5/223); recovery from a
  deep freeze is rare. Biggest lever remains condition choice (obs_image_only 1.98M tok/solve vs full_traj 25k).
- **Distractor capture is a myth (corrects §11):** on v09/v10 models almost never reach/dwell at decoy keys
  (median 0 near-decoy steps; rare tail — one v09 ep 52 steps). The penalty is that FAILED episodes never reach
  the CORRECT key (38–43% vs ~100% for solves) — extra branches hide the right key, they don't tempt the wrong
  one. (v10 green key = decoy despite a matching door → dead-end spur.)
- **Can't reason your way out of rotation:** per-query Qwen rotation-error RISES with reasoning length (32%
  shortest quintile → 56% longest, r=+0.29) — longer plan = longer turn-chain = more compounding. "Add more
  CoT" is not the fix. (Confound: harder mazes → longer chains; per-turn rate still rises.)
- **Data-integrity anomaly:** BFS `compute_difficulty` **over-counts `d3_dense_deadend` by 8** (reports
  31; two models solved in 23 legal steps → true optimum ≤23; inflated its cap to 93 vs ~69). Contained
  to 1/15 mazes; a benign +1/2 solver drift affects 8 mazes (use eval-time stored `optimal_steps`).

## 1. Data validity (SP2)
- **Scoring is clean:** 0/330 mismatches (recorded success == position-recompute == env `goal_reached`). No mis-scoring, no same-cell-pickup / wrong-done bugs.
- **Prompts faithful to maze truth:** world size, goal, every wall match spec; the row/col coordinate fix is confirmed in place.
- **All mazes solvable** (BFS-beatable); hardest is empirically hard, not broken.
- **Kimi's low scores are REAL** (a navigation ceiling, not a harness artifact): failures are `truncated`/`exhausted` (not `parse_failed`); 589 wall_confusion / 1271 repeated_plan / 84 give_up in its reasoning vs Claude ~0.
- Quarantined to `issues_/`: `cond_hist_multiturn_BUGGY` (ICL-dup bug), and the stale `cond_ctx_*_partial` dirs (superseded by the complete `ctxfix_*` re-run).

## 2. Difficulty model (SP3)
- **Difficulty is dominated by path length.** `optimal_steps` weight +0.184 (univariate r **+0.83**); maze solve rate correlates **r = −0.837** with optimal_steps but **~0 (+0.075)** with `mechanism_count`.
- **Mechanisms (keys/switches/distractors) show NO detectable independent difficulty effect** — BUT this is a **selection confound**: the 2 hardest mazes are long bare corridors with zero mechanisms, while keyed/switched mazes have shorter paths. Cannot be separated at n=15. **Needs the 200-maze run to disentangle.**
- `P(solve | model, config, difficulty)`: logistic, difficulty log-odds slope −4.09, well-calibrated (Brier 0.168). Kimi degrades faster and hits 0% on hard mazes; Claude retains ~8–10%.
- Caveats: score ≈ path-length ranking, so `winding_corridor` (~0%) is under-ranked; `blind_probe` (structurally trivial) is empirically 0.25-hard (features don't encode "blindness"); 36 mazes flagged `out_of_range`.

## 3. Exploratory analysis (the six questions)
1. **Hardest mazes:** `validation_10_v02_winding_corridor` (opt 61) and `14x14_corridor_1` (opt 89), each **2.9%** solved — both **zero-mechanism long corridors**. Easiest: `empty_room` (opt 11, 88.9%). Length, not mechanisms, drives it.
2. **Tokens:** scale ~linearly with path length (r +0.987); 33× spread hardest-vs-easiest maze. **`obs_image_only` is the money pit: ~2.0M tokens/success at 11% solve.** Token efficiency: Claude 299k/success < Kimi 478k < Qwen 576k.
3. **Distractors:** weak & heterogeneous once path length is controlled. Visual multi-key distractors (`v09`/`v10`) cost ~−0.15 at matched length (Kimi worst); the "wrong-key" decoy corridor shows **0** penalty (floor effect). Real but small; n=15.
4. **Feast-or-famine: YES, emphatically.** Solved → near-BFS (Claude median optimality 0.94, 66% ≥0.9). Failed → grind to the **exactly 3× optimal** step cap (81–86% of failures truncate/exhaust). No partial-progress middle ground.
5. **One-shot (reframed):** `icl_zero_shot` actually **ablates** the default worked example. So it measures *dependence* on it: Claude 0.60→**0.80** (example is a distractor for Claude), Kimi 0.80→**0.20** (uniquely dependent — without it 12/15 truncate), Qwen ~flat. The striking spread = "how much each model leans on the format demonstration."
6. **Failure taxonomy:** wander-to-cap (`obs_image_only`, parse_ok≈1 but circles), action-vocabulary interference (Qwen `act_cardinal` parse_ok→0.46, keeps emitting `MOVE_FORWARD`, 0% solve), open-loop plan brittleness (`qry_full_trajectory` dumps ~20–42 actions, no course-correction, ends `exhausted`). Verbose prompting doesn't help (minimal≈standard>verbose). Hard wall past ~45 optimal steps (all models →~0%).

## 4. ⚠️ `baseline_thinking` ran with UNEQUAL per-model token caps (CRITICAL — confounds the flagship condition)
- **Root cause:** the caps were **not** equal. `run_config.conditional_baseline_thinking_claude_kimi_qwen.json` sets `max_tokens` **Qwen 4,096 / Claude 8,192 / Kimi 16,384** — a 4× spread (verified against realized `output_tokens` ceilings). In thinking mode a model can spend the **entire** budget before emitting an action → truncated → `parse_ok: False`; `interface/runner.py` ends the episode `parse_failed` after **3 consecutive** unparseable replies.
- **The failures differ by model** (they are NOT a single shared bug):
  - **Claude (8,192): 6/6 blank truncations** (score 9/15 = 60%). `assistant_reply` is literally empty (redacted thinking ate the budget); ≥4 died at step 2–3 on solvable mazes (`d1` opt44 @2, `v06` opt28 @3, `s5` @7, `v02` @23).
  - **Kimi (16,384): 3/3 blank truncations** (score 12/15 = 80%). Same empty-reply signature — but at **double Claude's budget**; it still blanked 3×.
  - **Qwen (4,096): 0 blank** (score 9/15 = 60%). Qwen emits inline reasoning, so its replies are full (~10k chars); 3 failures are **genuine navigation grind-to-cap** (`parse_ok` True, ran to the step limit) and 3 are **reasoning overflowing the tiny 4,096 cap** with no parseable action.
- **Implication:** the cross-model ranking (Kimi 80 > Claude ≈ Qwen 60) is **confounded by the unequal cap, not reasoning.** Kimi did not out-reason Claude — it had 2×/4× the budget and still blanked. Claude blanks rather than death-loops and solves near-BFS (median 22 steps), so with Kimi's 16,384 budget it would very plausibly recover several step-2/3 deaths and climb past Qwen toward Kimi. Qwen's 60% is the most **navigation-real** of the three (actual grind-outs + a too-small cap, not a thinking blackout).
- **Fix:** re-run `baseline_thinking` only, with an **equal, large cap (≥16,384) for all three models**, for a clean thinking comparison.

## 5. Optimality, feast-or-famine by model, thinking efficiency, ablation ranking
- **2× step cap would cost 27/200 successes (13.5%)** (steps between 2× and 3× optimal): Claude 6/106 (5.7%), **Kimi 16/67 (24%)**, Qwen 5/27. Tightening the cap disproportionately punishes Kimi's marginal wins. Relevant to the per-tile-cap redesign.
- **Feast-or-famine is model-specific.** Failures grind to the cap for everyone (Claude 84% / Kimi 86% / Qwen 100%). But only **Claude solves near-BFS** (median optimality 0.94, 66% ≥0.9); Kimi & Qwen "solve messily" (median 0.78, 15–39% ≥0.9) — they reach the goal at ~2× the BFS path.
- **Thinking efficiency (parse-fails removed) is model-dependent — the "fewer tokens" was partly the bug.** Claude solved-episode: think-ON 154k tok/31 steps/0.94 vs think-OFF **116k tok**/36/0.97 → thinking is NOT more token-efficient for Claude (already navigates well). Kimi solved: think-ON 234k/**30 steps/1.00** vs think-OFF 217k/**93 steps/0.53** → thinking is a huge *path-quality* win for Kimi (stops it wandering).
- **Ablation ranking (Claude/Kimi, easiest→hardest):** baseline_thinking 68% > cond_prompt·minimal 63% > cond_prompt·standard 60% > qry_subgoal 57% (Claude 93% @ **29k tok** — efficiency sweet spot) > ctx_text_summary / icl_zero_shot / cond_prompt·verbose 50% > ctx_current 47% > hist_multiturn 43% (211k tok, priciest) > act_cardinal 40% > qry_full_trajectory 33% (**5.3k tok**, cheap/brittle) > **obs_image_only 13% (worst AND ~205k tok)**. (The prompt-detail variants straddle the pack: minimal/standard mid-high, verbose sinks with the middling configs — verbose helps least.)

## 6. Prompt-vs-code-rules audit + per-maze looping breakdown
_From `analysis-loopaudit` (`analysis/LOOP_AND_RULES.md`, commit 19d7558)._

### 6a. Prompt fidelity — VERDICT: prompts do NOT misdescribe the environment
Every interaction rule matches `gridworld/custom_env.py`: key pickup = same-cell; switch = same-cell TOGGLE; door = adjacent/facing TOGGLE w/ matching key; gate = switch-controlled (NOT directly toggleable); `act_cardinal` vocab faithful (`interface/action_space.py`). The **verbose** `RULES` block is accurate line-by-line; the default prompt under-specifies (lists actions/mechanism positions, not the "how") but never states a wrong rule.

**The toggle-spam episode = genuine model failure, not a prompt bug.** `v05_single_switch`, Claude, `obs_image_only`, seed 0: parked at `[8,2]` facing the closed GATE, emitted TOGGLE in **30 of 96 queries**, each returning the identical corrective feedback *"Gates cannot be toggled directly. Activate switch s1 instead"* — ignored 30×. Code and live feedback both state the rule correctly; Claude perseverated. (On the same maze under `ctx_current` it self-corrected after 2 no-op toggles and solved it → condition-specific perseveration, worst under image-only.)

### 6b. Per-corridor failure trajectories (coords `[x,y]`)
- **v02_winding_corridor** (goal `[18,6]`): **maze-driven** — all models oscillate at the NE dead-end **`(18,3)`**, ~3 rows short (Claude ping-pongs `(17,3)↔(18,3)`, Qwen `(18,1)↔(18,3)`). Only Claude/`qry_subgoal` escaped.
- **s5_14x14_corridor** (opt 89): **Claude wall-bangs at the start `(1,12)`** — **254/267 MOVE_FORWARD BLOCKED** ramming the south wall (directional lock, only 4 cells explored). Kimi wanders to the cap; Qwen stalls early.
- **m1 / d1_corridor** (goal `[1,8]`): shared wrong-branch dead-end **`(6,1)`**; Claude wall-bangs (m1: 104/132 BLOCKED). The wrong-key decoy adds NO new failure (floor effect).
- **v10_distractor_chain** (goal `[14,2]`): **distractor capture is model-specific** — Kimi/Qwen get trapped in the distractor arm ~`(7,3)/(8,2)/(8,6)`; Claude resists.

**Signature failure modes:** Claude = **wall-banging directional lock** (freezes at one cell, ≥50% BLOCKED actions); Kimi = wander-to-cap; Qwen = early stalls. Long bare corridors fail maze-wide.
- Ablation Table B (Qwen-integrated) reproduces A1/A2; integrating Qwen lifts `icl_zero_shot` (Qwen 67%) and sinks `qry_subgoal` (a **Claude-only** win). `act_cardinal`/`ctx_current` are Qwen 0%.

## Open issues / next steps
- **`baseline_thinking` unequal token caps (§4):** each cap is a fair *within-model* hard budget, but the **cross-model comparison is invalid** (Qwen 4k / Claude 8k / Kimi 16k — a 4× spread). To publish a thinking comparison, re-run `baseline_thinking` only with an **equal, large cap (≥16,384) for all three**. Also raises full-run costing: thinking needs a large cap to avoid budget-exhaustion blanks.
- The mechanism-vs-path-length difficulty question needs the **200-maze run** (§2).
- **Verify what `qry_subgoal` delegates** (does the harness pathfind to each subgoal, or does the model emit the sub-paths?) — decides whether Claude's 93% is a fair comparison (see A6).
- Qwen is fully folded in; re-fit SP2/SP3 on the complete 3-model set if those analyses need publication-grade refreshes.
- Consolidate the SP branches; merge the pending `LOOP_AND_RULES.md` (§6).

---

## Appendix A — Full data tables (compaction-proof record)
_Claude+Kimi from `cond-sweep-20260704-api`; Qwen complete across `cond_massive` (thinking OFF except baseline) + `cond_prompt`. n=15 mazes / 1 seed throughout._

### A1. Pass rate — model × config (%)
| config | Claude | Kimi | Qwen |
|---|---|---|---|
| baseline_thinking | 60 | 80 | 64 |
| cond_prompt·minimal | 67 | 60 | 53 |
| cond_prompt·standard | 73 | 47 | 60 |
| cond_prompt·verbose | 67 | 33 | 67 |
| qry_subgoal | 93 | 20 | 20 |
| icl_zero_shot | 80 | 20 | 67 |
| ctx_text_summary | 60 | 40 | 13 |
| ctx_current | 53 | 40 | 0 |
| hist_multiturn | 53 | 33 | 13 |
| act_cardinal | 40 | 40 | 0 |
| qry_full_trajectory | 47 | 20 | 7 |
| obs_image_only | 13 | 13 | 7 |

Overall pooled: **Claude 58.9% > Kimi 37.2% > Qwen 31.1%**.

### A2. Token spend + wall-clock per config (pooled 3 models; wall-clock = summed LLM latency)
| config | n | solve% | median tok | total tok (M) | LLM-hours |
|---|---|---|---|---|---|
| baseline_thinking | 45 | 67 | 186k | 11.6 | 54.3 |
| cond_prompt·standard | 45 | 60 | 174k | 10.8 | 22.9 |
| cond_prompt·minimal | 45 | 60 | 131k | 10.5 | 20.0 |
| cond_prompt·verbose | 45 | 56 | 173k | 11.3 | 20.3 |
| icl_zero_shot | 45 | 56 | 116k | 7.6 | 17.9 |
| qry_subgoal | 45 | 44 | 58k | 4.1 | 4.5 |
| ctx_text_summary | 45 | 38 | 146k | 8.2 | 17.4 |
| hist_multiturn | 45 | 33 | 232k | 12.1 | 8.2 |
| ctx_current | 45 | 31 | 159k | 8.5 | 20.3 |
| act_cardinal | 45 | 27 | 39k | 3.6 | 3.6 |
| qry_full_trajectory | 45 | 24 | 5.1k | 0.3 | 1.6 |
| obs_image_only | 45 | 11 | 195k | 9.9 | 24.0 |

**Totals: 98.5M tokens, 214.9 LLM-hours, 540 episodes.**

### A3. Optimality & a 2× step cap (ratio = optimal/steps; 3× cap floors at ~0.33)
| model | successes | would fail @2× cap (steps>2×opt) | min ratio |
|---|---|---|---|
| Claude | 106 | 6 (5.7%) | 0.34 |
| Kimi | 67 | 16 (24%) | 0.34 |
| Qwen | 27 | 5 (19%) | 0.36 |

27/200 successes (13.5%) would flip to failure under a 2× cap.

### A4. Feast-or-famine by model
| model | solved: median optimality | % solved ≥0.9 | % failures grind-to-cap |
|---|---|---|---|
| Claude | 0.94 | 66% | 84% |
| Kimi | 0.78 | 39% | 86% |
| Qwen | 0.78 | 15% | 100% |

### A5. Thinking vs no-thinking (solved episodes, parse-fails removed)
| model | mode | median tok | median steps | solved optimality |
|---|---|---|---|---|
| Claude | think-ON | 154k | 31 | 0.94 |
| Claude | think-OFF | 116k | 36 | 0.97 |
| Kimi | think-ON | 234k | 30 | 1.00 |
| Kimi | think-OFF | 217k | 93 | 0.53 |

Thinking is not more token-efficient for Claude; it's a big path-quality win for Kimi (30 vs 93 steps).

### A6. Claude query-mode comparison (why subgoal ≫ step-by-step)
| config | solve% | median tok | LLM calls | tok/call | median steps | solved opt |
|---|---|---|---|---|---|---|
_Solved-opt refreshed from stored `optimality_ratio` (prior per-config values were stale). The three `cond_prompt` variants are the same **step-by-step** query mode (differ only in prompt detail), so they cluster — query mode, not prompt detail, moves this table._
| qry_subgoal | 93 | 20k | 6 | 3,494 | 33 | 0.95 |
| icl_zero_shot | 80 | 82k | 35 | 2,188 | 35 | 0.94 |
| cond_prompt·standard (step-by-step) | 73 | 116k | 36 | 3,223 | 36 | 0.97 |
| cond_prompt·minimal (step-by-step) | 67 | 120k | 37 | 3,253 | 37 | 0.98 |
| cond_prompt·verbose (step-by-step) | 67 | 127k | 36 | 3,519 | 36 | 0.92 |
| baseline_thinking | 60 | 148k | 22 | 6,310 | 22 | 0.94 |
| qry_full_trajectory | 47 | 5k | 1 | 4,932 | 20 | 1.00 |

Driver = **decision count**, not per-step overthinking (tok/call ~equal at ~3.2–3.5k). Step-by-step forces **~36 independent observation→action decisions**; subgoal collapses to **6 route decisions**. Subgoal **rescues the long/hard mazes** step-by-step fails: `winding_corridor`, `multi_path`, `s5_corridor`, `d3_deadend` (SOLVE under subgoal, FAIL step-by-step). Only `d1_wrong_ky` flips the other way (decoy key likely picked as a subgoal). ⚠️ Confirm subgoal doesn't offload pathfinding to the harness (solved optimality 0.95, not 1.0, suggests the model still navigates).

### A7. baseline_thinking unequal per-model token caps (§4)
| model | max_tokens | score | failures | blank truncations | other failures |
|---|---|---|---|---|---|
| Kimi | 16,384 | 12/15 (80%) | 3 | 3 / 3 | 0 |
| Claude | 8,192 | 9/15 (60%) | 6 | 6 / 6 | 0 |
| Qwen | 4,096 | 9/15 (60%) | 6 | 0 | 3 navigation grind-to-cap + 3 reasoning-overflow parse-fails |

Cross-model ranking is confounded by the **4× budget spread**, not reasoning (§4).

### A8. Per-model tokens + wall-clock, and the calls-per-step token driver
_⚠️ Wall-clock = single-stream summed LLM latency. Claude/Kimi = API latency; **Qwen = local A100 inference (far slower per token) — Qwen minutes are NOT comparable to the API models.**_

Selected per-episode (median tokens / mean wall-min):
| config | Claude | Kimi | Qwen |
|---|---|---|---|
| cond_prompt·minimal | 120k / 15.1m | 158k / 15.5m | 123k / 49.2m |
| cond_prompt·standard | 116k / 12.8m | 218k / 14.3m | 167k / 64.5m |
| cond_prompt·verbose | 127k / 13.5m | 218k / 12.4m | 190k / 55.2m |
| baseline_thinking | 148k / 16.5m | 240k / 40.2m | 255k / 160.3m |
| qry_subgoal | 20k / 1.9m | 64k / 4.0m | 162k / 12.3m |
| obs_image_only | 208k / 6.0m | 203k / 23.2m | 174k / 66.7m |

**Token cost = number of LLM calls = env steps taken** (step-by-step querying makes one call per step). Calls/episode by config: obs_image_only 96, ctx_current 92, hist_multiturn 86, ctx_text_summary 84, **cond_prompt·verbose 65 / ·standard 64 / ·minimal 60**, icl 55, baseline_thinking 30, qry_subgoal 21, act_cardinal 16, qry_full 1.

**Why the `cond_prompt` prompt-detail arm is the biggest *combined* spend (32.6M tok / 63h) at a healthy 56–60% solve — NOT death loops:**
1. **It is three conditions, not one** — the arm runs minimal/standard/verbose (15 mazes × 3 models = 45 pooled each), so the *combined* 135-episode total is ~3× a single config; each variant on its own (~10.5–11.3M tok) is in line with the other high-spend configs.
2. **Step-by-step querying** — ~60–65 LLM calls/episode (one per env step), each re-processing the observation; even a clean success costs 36–94 calls.
3. **Step-count varies by model** — Claude ~36 steps/36 calls (efficient); **Kimi wanders to 65–94 steps → 158–218k tok** even when it solves (messy wins); Qwen 44–65. Kimi is the token hog here.
- Variant effect (pooled): minimal 60 calls / standard 64 / verbose 65, and **verbose solves *worst*** (56% vs 60% for minimal/standard) — verbose prompting costs more and helps less. The verbose hit is Kimi-driven (60→33%); Qwen actually improves (53→67%), Claude is flat.
