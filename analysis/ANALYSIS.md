# MultiNet-v2.0 Conditional-Eval — Verbose Analysis (full tables + detail)

Companion to the summary `FINDINGS.md`. This document keeps the **complete** tables and
per-model detail. Data: Claude (`claude-opus-4-8`) + Kimi (`kimi-k2.6`) from
`cond-sweep-20260704-api` (10 clean cells, 1 seed); Qwen (`Qwen3.6-27B`) complete
across `cond_massive` (thinking OFF except `baseline_thinking`) + `cond_prompt`. **n=15
distinct mazes / 1 seed everywhere — all numbers are directional.** `issues_/` cells
(`hist_multiturn_BUGGY`, stale ctx partials) are excluded by the loader.

**`cond_prompt` is shown disaggregated** into its 3 prompt-detail variants —
`cond_prompt·minimal` / `cond_prompt·standard` / `cond_prompt·verbose` — each 15 mazes/model
(45 pooled), so the three slot into the rankings alongside the other 15-maze configs. The former
single `cond_prompt` row = these three pooled (135 episodes). Optimality figures are computed from
the stored `optimality_ratio` (some earlier per-config optimality numbers were stale and have been
refreshed).

Overall pooled solve: **Claude 58.9% · Kimi 37.2% · Qwen 30.6%**.

---

## 1. Pass rate — model × config (%)
| config | Claude | Kimi | Qwen | pooled |
|---|---|---|---|---|
| baseline_thinking | 60 | 80 | 64 | 68 |
| cond_prompt·minimal | 67 | 60 | 53 | 60 |
| cond_prompt·standard | 73 | 47 | 60 | 60 |
| cond_prompt·verbose | 67 | 33 | 67 | 56 |
| qry_subgoal | 93 | 20 | 20 | 44 |
| icl_zero_shot | 80 | 20 | 67 | 56 |
| ctx_text_summary | 60 | 40 | 13 | 38 |
| ctx_current | 53 | 40 | 0 | 31 |
| hist_multiturn | 53 | 33 | 13 | 33 |
| act_cardinal | 40 | 40 | 0 | 27 |
| qry_full_trajectory | 47 | 20 | 7 | 24 |
| obs_image_only | 13 | 13 | 7 | 11 |

Notes: `qry_subgoal` is a **Claude-only** win (93 vs 20/20). `icl_zero_shot` (=one-shot example *ablated*) helps Claude & Qwen, sinks Kimi. `act_cardinal`/`ctx_current` are Qwen 0%. `baseline_thinking`'s top rank is **confounded** by unequal per-model token caps (§9).

## 2. Full per-model × config table (median tokens / mean wall-min per episode)
_⚠️ Wall-clock = single-stream summed LLM latency. Claude/Kimi = API latency; **Qwen = local A100 inference (far slower per token) — not comparable across hardware.**_

### Claude
| config | n | solve% | median tok | wall min/ep |
|---|---|---|---|---|
| qry_subgoal | 15 | 93 | 20,367 | 1.9 |
| icl_zero_shot | 15 | 80 | 82,264 | 10.6 |
| cond_prompt·standard | 15 | 73 | 116,038 | 12.8 |
| cond_prompt·minimal | 15 | 67 | 120,363 | 15.1 |
| cond_prompt·verbose | 15 | 67 | 126,695 | 13.5 |
| baseline_thinking | 15 | 60 | 148,076 | 16.5 |
| ctx_text_summary | 15 | 60 | 82,646 | 5.8 |
| hist_multiturn | 15 | 53 | 191,581 | 17.9 |
| ctx_current | 15 | 53 | 170,897 | 5.8 |
| qry_full_trajectory | 15 | 47 | 4,932 | 0.8 |
| act_cardinal | 15 | 40 | 87,127 | 5.3 |
| obs_image_only | 15 | 13 | 207,523 | 6.0 |

### Kimi
| config | n | solve% | median tok | wall min/ep |
|---|---|---|---|---|
| baseline_thinking | 15 | 80 | 239,822 | 40.2 |
| cond_prompt·minimal | 15 | 60 | 157,819 | 15.5 |
| cond_prompt·standard | 15 | 47 | 217,735 | 14.3 |
| act_cardinal | 15 | 40 | 49,198 | 4.6 |
| ctx_current | 15 | 40 | 156,380 | 16.9 |
| ctx_text_summary | 15 | 40 | 143,405 | 22.9 |
| cond_prompt·verbose | 15 | 33 | 217,695 | 12.4 |
| hist_multiturn | 15 | 33 | 230,236 | 5.9 |
| icl_zero_shot | 15 | 20 | 173,307 | 9.8 |
| qry_full_trajectory | 15 | 20 | 5,326 | 1.4 |
| qry_subgoal | 15 | 20 | 63,547 | 4.0 |
| obs_image_only | 15 | 13 | 203,317 | 23.2 |

### Qwen (local A100; wall-clock not comparable to API)
| config | n | solve% | median tok | wall min/ep |
|---|---|---|---|---|
| cond_prompt·verbose | 15 | 67 | 190,210 | 55.2 |
| icl_zero_shot | 15 | 67 | 80,484 | 51.1 |
| baseline_thinking | 15 | 60 | 255,488 | 160.3 |
| cond_prompt·standard | 15 | 60 | 167,084 | 64.5 |
| cond_prompt·minimal | 15 | 53 | 123,487 | 49.2 |
| qry_subgoal | 15 | 20 | 161,717 | 12.3 |
| hist_multiturn | 15 | 13 | 232,144 | 9.1 |
| ctx_text_summary | 15 | 13 | 160,073 | 41.0 |
| qry_full_trajectory | 15 | 7 | 4,550 | 4.1 |
| obs_image_only | 15 | 7 | 173,855 | 66.7 |
| act_cardinal | 15 | 0 | 13,359 | 4.4 |
| ctx_current | 15 | 0 | 186,424 | 58.6 |

## 3. Token spend + wall-clock per config (pooled 3 models)
| config | n | solve% | median tok | total tok (M) | LLM-hours |
|---|---|---|---|---|---|
| baseline_thinking | 45 | 67 | 185,616 | 11.6 | 54.3 |
| obs_image_only | 45 | 11 | 195,113 | 9.9 | 24.0 |
| cond_prompt·standard | 45 | 60 | 174,482 | 10.8 | 22.9 |
| cond_prompt·verbose | 45 | 56 | 172,846 | 11.3 | 20.3 |
| ctx_current | 45 | 31 | 159,251 | 8.5 | 20.3 |
| cond_prompt·minimal | 45 | 60 | 130,988 | 10.5 | 20.0 |
| icl_zero_shot | 45 | 56 | 116,088 | 7.6 | 17.9 |
| ctx_text_summary | 45 | 38 | 145,977 | 8.2 | 17.4 |
| hist_multiturn | 45 | 33 | 231,770 | 12.1 | 8.2 |
| qry_subgoal | 45 | 44 | 58,292 | 4.1 | 4.5 |
| act_cardinal | 45 | 27 | 39,388 | 3.6 | 3.6 |
| qry_full_trajectory | 45 | 24 | 5,103 | 0.3 | 1.6 |
| **TOTAL** | **540** | | | **98.5M** | **214.9** |

## 4. The token driver: LLM calls = env steps (step-by-step querying)
Median LLM calls per episode by config: obs_image_only **96**, ctx_current 92, hist_multiturn 86, ctx_text_summary 84, cond_prompt·verbose 65, cond_prompt·standard 64, cond_prompt·minimal 60, icl_zero_shot 55, baseline_thinking 30, qry_subgoal 21, act_cardinal 16, qry_full_trajectory **1**. Token cost scales directly with calls, and calls = env steps taken (one LLM call per step under the default query mode).

### cond_prompt·{minimal,standard,verbose} deep-dive (why the prompt-detail arm dominates total spend — not death loops)
Pooled, the three prompt variants together were the biggest total spend (32.6M tok / 63.1 LLM-h) at a healthy 56–60% solve. Two structural reasons, plus the variant effect:
1. **It is three conditions, not one** — the prompt-detail arm runs minimal/standard/verbose (15 mazes × 3 models each = 45 pooled per variant, 135 combined), so the *combined* total is ~3× a single config.
2. **Step-by-step querying** — ~60–65 calls/episode, each re-processing the observation.
3. **Step-count varies by model** (per model × variant, n=15 each):

| model | variant | n | solve% | median tok | median calls | tok/call | median steps | wall min/ep |
|---|---|---|---|---|---|---|---|---|
| Claude | minimal | 15 | 67 | 120,363 | 37 | 3,253 | 37 | 15.1 |
| Claude | standard | 15 | 73 | 116,038 | 36 | 3,223 | 36 | 12.8 |
| Claude | verbose | 15 | 67 | 126,695 | 36 | 3,519 | 36 | 13.5 |
| Kimi | minimal | 15 | 60 | 157,819 | 65 | 2,428 | 65 | 15.5 |
| Kimi | standard | 15 | 47 | 217,735 | 94 | 2,316 | 93 | 14.3 |
| Kimi | verbose | 15 | 33 | 217,695 | 93 | 2,341 | 93 | 12.4 |
| Qwen | minimal | 15 | 53 | 123,487 | 44 | 2,807 | 44 | 49.2 |
| Qwen | standard | 15 | 60 | 167,084 | 59 | 2,832 | 59 | 64.5 |
| Qwen | verbose | 15 | 67 | 190,210 | 65 | 2,926 | 65 | 55.2 |

Kimi takes **65–93 steps vs Claude's ~36** for the same mazes → ~2× the calls → 205–218k tok even when it solves ("messy wins"). Variant effect (pooled): minimal 60 calls / standard 64 / verbose 65 — **verbose solves worst (56% vs 60% for minimal/standard) and costs the most**. The verbose penalty is Kimi-driven (60→33%) and reverses for Qwen (53→67%); Claude is flat.

## 5. Optimality & a 2× step cap
`ratio = optimal/steps` (1.0 = perfect BFS). Runtime cap = 3× optimal (floor ~0.33).
| model | successes | ratio<0.5 (steps>2×opt) → fail@2× | min ratio |
|---|---|---|---|
| Claude | 106 | 6 (5.7%) | 0.34 |
| Kimi | 67 | 16 (24%) | 0.34 |
| Qwen | 27 | 5 (19%) | 0.36 |

**27/200 successes (13.5%) would flip to failure under a 2× cap** — disproportionately Kimi's marginal wins. Example lost successes: Claude/blind_probe 50/opt20; Kimi/v04 96/opt33, v07 58/opt28, d3 66/opt31, v05 82/opt32.

## 6. Feast-or-famine by model
| model | solved median optimality | % solved ≥0.9 | % failures grind-to-cap |
|---|---|---|---|
| Claude | 0.94 | 66% | 84% |
| Kimi | 0.78 | 39% | 86% |
| Qwen | 0.78 | 15% | 100% |

Universal: failures grind to the cap. But only **Claude solves near-BFS**; Kimi/Qwen "solve messily" at ~2× the optimal path.

## 7. Thinking vs no-thinking (solved episodes, parse-fails removed)
| model | mode | median tok | median steps | solved optimality |
|---|---|---|---|---|
| Claude | think-ON | 154k | 31 | 0.94 |
| Claude | think-OFF | 116k | 36 | 0.97 |
| Kimi | think-ON | 234k | 30 | 1.00 |
| Kimi | think-OFF | 217k | 93 | 0.53 |

Thinking is **not** more token-efficient for Claude (already navigates well). For Kimi it's a huge **path-quality** win: 30 vs 93 steps, optimality 1.00 vs 0.53.

## 8. Claude query-mode comparison (why subgoal ≫ step-by-step)
_Solved-opt refreshed from stored `optimality_ratio` (the prior per-config values here were stale)._ The three `cond_prompt` variants are all the same **step-by-step** query mode (they differ only in prompt detail), so they cluster together — the query mode, not the prompt detail, is what moves this table.
| config | solve% | median tok | LLM calls | tok/call | median steps | solved opt |
|---|---|---|---|---|---|---|
| qry_subgoal | 93 | 20,367 | 6 | 3,494 | 33 | 0.95 |
| icl_zero_shot | 80 | 82,264 | 35 | 2,188 | 35 | 0.94 |
| cond_prompt·standard (step-by-step) | 73 | 116,038 | 36 | 3,223 | 36 | 0.97 |
| cond_prompt·minimal (step-by-step) | 67 | 120,363 | 37 | 3,253 | 37 | 0.98 |
| cond_prompt·verbose (step-by-step) | 67 | 126,695 | 36 | 3,519 | 36 | 0.92 |
| baseline_thinking | 60 | 148,076 | 22 | 6,310 | 22 | 0.94 |
| qry_full_trajectory | 47 | 4,932 | 1 | 4,932 | 20 | 1.00 |

**Driver = decision count, not overthinking** (tok/call ≈ equal). Step-by-step = ~36 independent observation→action commitments (each a fresh chance to misread and turn wrong; errors compound over long paths). Subgoal = 6 route decisions → rescues the long/hard mazes step-by-step fails: `winding_corridor`, `s5_corridor`, `multi_path`, `d3_deadend` (SOLVE under subgoal, FAIL step-by-step). Only `d1_wrong_ky` flips the other way (decoy key chosen as a subgoal). ⚠️ Confirm subgoal doesn't offload pathfinding to the harness (solved optimality 0.95, not 1.0, suggests the model still navigates).

## 9. ⚠️ baseline_thinking ran with UNEQUAL per-model token caps (confounds the flagship condition)
The three models did **not** share a budget — `run_config.conditional_baseline_thinking_*` sets
`max_tokens` **Qwen 4,096 / Claude 8,192 / Kimi 16,384** (a 4× spread). In thinking mode a model can
spend the whole budget before emitting an action → truncated → `parse_ok=False`; `interface/runner.py`
ends the episode `parse_failed` after **3 consecutive** unparseable replies. The failures are NOT the
same across models:
| model | max_tokens | score | failures | failure nature |
|---|---|---|---|---|
| Kimi | 16,384 | 12/15 (80%) | 3 | **3/3 blank truncation** — replies empty (redacted thinking ate the budget), pinned at 16,384 |
| Claude | 8,192 | 9/15 (60%) | 6 | **6/6 blank truncation** at 8,192; ≥4 died at step 2–3 on solvable mazes (`d1` opt44 @2, `v06` @3, `s5`, `v02`) |
| Qwen | 4,096 | 9/15 (60%) | 6 | **0 blank** — Qwen shows inline reasoning, so replies are full (~10k chars). 3 = genuine navigation grind-to-cap (`parse_ok` still True); 3 = reasoning overran the 4,096 cap with no parseable action |

**Consequence: the cross-model ranking (Kimi 80 > Claude ≈ Qwen 60) is confounded by the unequal cap, not reasoning.** Kimi didn't out-reason Claude — it had **2× Claude's / 4× Qwen's** budget and still blanked 3×. Claude blanks (rather than death-loops) and its solves are near-BFS (median 22 steps); given Kimi's 16,384 budget it would very plausibly recover several step-2/3 deaths and climb past Qwen toward Kimi. Qwen's 60% is the most *navigation-real* of the three (its failures are actual grind-outs + a too-small 4,096 cap, not a thinking blackout).

**Decision:** as a *within-model* metric each cap is a fair hard budget, but the **cross-model comparison is not fair** (unequal budgets). For a clean thinking comparison, re-run `baseline_thinking` only, with an **equal, large cap (≥16,384) for all three**.

## 10. Prompt-vs-code-rules audit
**Verdict: prompts do NOT misdescribe the environment.** Every rule matches `gridworld/custom_env.py`: key pickup = same-cell; switch = same-cell TOGGLE; door = adjacent/facing TOGGLE w/ matching key; **gate = switch-controlled (not directly toggleable)**; `act_cardinal` vocab faithful. The **verbose** RULES block is accurate line-by-line; the default under-specifies but never states a wrong rule.

**Toggle-spam episode = genuine model failure:** `v05_single_switch`, Claude, `obs_image_only`, parked at `[8,2]` facing the closed gate, fired TOGGLE in **30 of 96 queries**, each returning *"Gates cannot be toggled directly — activate switch s1 instead"* — ignored 30×. Same maze under `ctx_current` it self-corrected after 2 toggles and solved. Condition-specific perseveration, worst under image-only.

## 11. Per-maze looping / corridor failure trajectories (coords `[x,y]`)
- **v02_winding_corridor** (goal `[18,6]`): **maze-driven** — all models oscillate at the NE dead-end `(18,3)`, ~3 rows short (Claude `(17,3)↔(18,3)`, Qwen `(18,1)↔(18,3)`). Only Claude/`qry_subgoal` escaped.
- **s5_14x14_corridor** (opt 89): **Claude wall-bangs at start `(1,12)`** — 254/267 MOVE_FORWARD BLOCKED ramming the south wall (directional lock, 4 cells explored). Kimi wanders to cap; Qwen stalls early.
- **m1 / d1_corridor** (goal `[1,8]`): shared wrong-branch dead-end `(6,1)`; Claude wall-bangs (m1: 104/132 BLOCKED). Wrong-key decoy adds no new failure (floor effect).
- **v10_distractor_chain** (goal `[14,2]`): **model-specific distractor capture** — Kimi/Qwen trapped in the distractor arm ~`(7,3)/(8,2)/(8,6)`; Claude resists.

**Signature failure modes:** Claude = wall-banging directional lock (≥50% BLOCKED); Kimi = wander-to-cap; Qwen = early stalls.

## 12. Difficulty model (SP3)
Path length dominates: solve rate r=−0.837 with `optimal_steps`, ~0 (+0.075) with `mechanism_count`. Ridge weight `optimal_steps` +0.184 (top). Mechanisms show no detectable independent difficulty **but that's a selection confound** (hardest mazes are long bare corridors with 0 mechanisms). LOMO-CV MAE 0.198 vs 0.218 naive (n=15). `P(solve|model,config,difficulty)` logistic slope −4.09, Brier 0.168. **Needs the 200-maze run to separate mechanism difficulty from path length.**

## 13. Cost (Qwen A100 fleet)
**Full program (reconstructed from every start/stop operation; watchdog stops injected):**
| fleet | A100-hrs | what |
|---|---|---|
| 2026-07-03 dev fleet (3 VMs) | 7.6 | early smoke/dev |
| 2026-07-04 main fleet (3 VMs) | 45.4 | smoke + batch attempts + massive run + completed backfill |
| **TOTAL** | **53.1** | **≈ $265 @ $5/A100-hr** |

- The *main massive run* alone was ~37.5 A100-hrs (~$187); the ~15 extra A100-hrs are the 07-03 dev fleet + 07-04 smoke/batch restarts + backfill.
- **Inference workload:** 128 GPU-hours summed single-stream latency (177 episodes, 12,830 queries) — concurrency-inflated (≤48 batched), **not billable**; the ~3.4× vs billed uptime is the batching win.
- Caveats: assumes $5/A100-hr (a2-ultragpu-1g on-demand; adjust for actual rate/CUD); A100-only (small n2 coordinator adds a few $, negligible); includes some idle time (fleet up but unsaturated between batches / during hard-maze tails).

## Data caveats (apply throughout)
- **n=15 mazes / 1 seed** — directional, not powered for significance.
- Path-length ⟂ mechanism confound unresolved (long mazes happen to be mechanism-free).
- Qwen wall-clock is local-A100 inference, not comparable to API latency.
- `baseline_thinking` confounded by **unequal** per-model token caps (Qwen 4k / Claude 8k / Kimi 16k, §9); its cross-model rank is an artifact of budget, not reasoning.
- Qwen backfill is complete; all 540 conditional episodes are present.
