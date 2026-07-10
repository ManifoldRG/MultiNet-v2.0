# Design proposals — post-sweep iteration (2026-07)

Three designs requested after the conditional sweep, **not yet implemented**.
Each section says what the feature is, the design decision points, the
recommendation, and exactly which code changes it needs. Data citations are
from `analysis/FINDINGS.md` (540 episodes, 3 models, 15 mazes, 1 seed).

Related work already landed on `fix/run-bugfixes` (not proposals — done):
resolved-goal consistency + executable-action difficulty counts, the
FINAL_OUTPUT-anchored parser, lean multiturn history turns, the equal-token-cap
guard (`allow_unequal_max_tokens` opt-out), and BATCH_CAP required-no-default.

---

## 1. New context mode: `summary_last3` (text_summary + rolling 3 images)

**What:** one condition combining the two context mechanisms that each helped
different models: the cumulative `text_summary` sentence ("You picked up the
yellow key, then opened the yellow door, then…") plus the last-3
decision-frame images (`last3`'s visual half). Today `context_window` is a
single enum — the two are mutually exclusive.

**Why:** `ctx_text_summary` beat `ctx_current` for Claude/Kimi (50 vs 47%) but
starved image-reliant behavior; `last3` carries recency but forgets mechanism
history past 3 steps. The combination tests whether "long-horizon events as
text + short-horizon dynamics as pixels" is additive.

**Design:** add a `ContextWindow` value `"summary_last3"` that composes the two
existing builders rather than introducing a new pathway:

- `history_text(...)` returns `text_summary_history(...)` for both
  `"text_summary"` and `"summary_last3"`.
- `history_content_blocks(...)` / `recent_history_steps(...)` return the last-3
  image blocks for both `"last3"` and `"summary_last3"`.

That is deliberately it — both builders are already pure functions of the
transcript, so composition is a membership test on the enum, no new state.

**Code changes:**

| file | change |
|---|---|
| `interface/observation.py` | `ContextWindow` literal + the two membership tests (`recent_history_steps`: `context_window in ("last3", "summary_last3")`; `history_text`: summary branch matches both) |
| `interface/config.py` | accept the new value in `ExperimentConfig.context_window` validation (if enumerated) |
| `gridworld/fixtures/run_config.conditional_context_window_*.json` | add the cell to the Context-window condition set |
| `tests/` | unit: a transcript with a pickup + 5 moves yields BOTH the summary block and exactly 3 image blocks; e2e (pattern of `tests/test_history_dedup.py`): stateless episode carries both sections, rolling episode carries neither (lean-turn rule) |
| `analysis/data.py` | config-name mapping for the new cell |

**Interaction rule (already enforced):** with `chat_history != stateless` the
embedded history sections are disabled (`with_context_history=False` in
`interface/runner.py`) — `summary_last3` composes with multiturn chat the same
way `last3` does: the chat carries it.

**Cost note:** ~3 extra images per query on top of `text_summary`'s flat text;
expect `ctx_last3`-like token profiles (~146-159k median/episode), not
`obs_image_only`'s 195k.

---

## 2. New condition: full context at every step

**What:** every query carries the complete episode so far (all observations /
actions / feedback), not a 3-step window.

**Two possible constructions — recommend (a):**

- **(a) `chat_history: "full"` (multiturn, no trim) — already implemented.**
  The runner supports `rolling` (trimmed to `chat_turns_max`) and `full`
  (untrimmed); `hist_multiturn` ran `rolling`+3. A "full context" cell is a
  **run-config change, zero code**: `chat_history: "full"`,
  `context_window: "current"`. Provider prompt caching makes append-only
  multiturn the cheap construction (each query re-reads the prefix from
  cache), and the lean-turn rule keeps turns non-duplicative.
- **(b) `context_window: "all"` (stateless, in-prompt replay).** New enum value
  where `recent_history_steps` returns `history_steps(transcript)` unsliced.
  Costs O(n²) tokens over an episode with NO cache reuse (the prompt is
  rebuilt each step), and re-renders every historical frame for image modes.
  Only worth building if the experiment explicitly wants "full context without
  chat structure" as its own axis (i.e. to separate "information" from
  "conversation shape").

**Caps interaction:** a 90-step episode × step-by-step querying means the last
queries carry ~90 turns. With images this can exceed `max_model_len` for local
Qwen (16384) long before the step cap — (a) needs either `text_only`, image
stripping from old turns (keep the text label, drop the pixels past N turns —
small change in `interface/runner.py` before append), or a documented
Qwen exclusion. Flag this in the run config description rather than silently
truncating.

**Code changes (a):** new fixture run_config cell + analysis config-name only.
**Code changes (b):** `interface/observation.py` (`"all"` in `ContextWindow` +
unsliced branch), config validation, fixture, tests mirroring the last3 ones.

---

## 3. Step-cap redesign: from global 3× to a progress watchdog

**Today:** `_runtime_capped_spec` (`scripts/run_pipeline.py:341`) clamps
`max_steps` to `3 × canonical optimal`. Episodes that fail overwhelmingly
grind to exactly this cap (81-100% of failures by model), and in failed
episodes ~46% of output tokens are spent thrashing after the last new-cell
visit (~7.7M tokens across the sweep).

**The user-floated options and their problems:**

1. **"20 steps without stepping on a new tile."** If "new" means
   never-visited-before, mazes that require backtracking (key → return through
   the same corridor → door) impose a design constraint we don't want: a long
   legal backtrack reads as zero progress. FINDINGS supports the concern —
   a *no-new-cell* watchdog "kills key-backtrackers".
2. **"BFS solution as a list; 20 steps that neither touch a new tile nor
   advance the BFS state."** Problems: (i) the BFS path is *one* optimal
   route; `multi_path` and any alternate-route solve legitimately leaves the
   list and would never "advance" it; (ii) after the resolved-goal fix there
   is still no guarantee the model's strategy tracks BFS order (subgoal mode
   demonstrably re-plans); (iii) it couples the runtime to a solver artifact,
   which we just finished decoupling for correctness reasons.

**Recommendation: progress = novel *runtime state*, not novel tile.**
Define a step as progress iff it produces a never-seen episode signature:

```
signature = (agent_pos, frozenset(inventory), frozenset(open_doors),
             frozenset(active_switches), frozenset(open_gates))
```

- Loop-backs are handled naturally: re-walking a corridor after picking up
  the key is a **new** signature for every cell (inventory changed) — one
  free traversal per mechanism state change, which is exactly the legal
  backtrack budget a keyed maze needs.
- Facing is deliberately excluded so spin-in-place doesn't count as progress.
- Wall-bangs, toggle-jams, and oscillation (the three failure clusters:
  wall-banger 68, wanderer 116, toggle-jammer 8) all stop producing novel
  signatures within a handful of steps.

**Watchdog rule:** end the episode with `end_reason="stalled"` after
`K = 20` consecutive steps with no novel signature. Keep the global cap as a
backstop at **3×** — do NOT tighten it to 2×: 27/200 sweep successes (13.5%)
land between 2× and 3×, and the pain is concentrated on Kimi (24% of its
wins). The empirical trade at K=20 (measured on the frozen-in-place variant,
the closest proxy): ~23% of failed-episode output tokens recovered for a ~2%
success loss.

**Comparability warning (why this is NOT a drop-in change):** any watchdog
changes the success/steps distribution vs the completed sweep. Ship it as a
new config knob defaulting OFF; enable per-run-config. Per
`per-tile-step-cap-next-iteration` this belongs to the next iteration, not a
patch of the current one.

**Code changes:**

| file | change |
|---|---|
| `interface/config.py` | `progress_stall_k: int \| None = None` (None = off) |
| `interface/runner.py` | track `seen_signatures: set` + `stall_count` in the step loop (state is already snapshotted per step); `break` with `end_reason="stalled"` |
| `gridworld/backends/base.py` | nothing — `GridState` already exposes position/inventory/doors/switches/gates |
| `scorer` / `episode_metrics` | count `stalled` as a failure end_reason (alongside `truncated`/`exhausted`) |
| `analysis/data.py` | `end_reason` vocabulary + `is_clean` unaffected (stalled is a legitimate outcome, not a bug) |
| `tests/` | scripted-agent e2e (pattern of `tests/test_cardinal_runner.py`): oscillating agent stalls at exactly K; key-backtrack agent does NOT stall while re-walking a corridor post-pickup; K=None keeps today's behavior byte-identical |

**Rejected variant worth recording:** per-tile budget (`3-5× BFS` *per tile*,
from the earlier brainstorm). It punishes revisit-heavy but progressing
strategies (wanderers that eventually solve — Kimi's signature win mode) and
needs per-tile bookkeeping the signature set gives us for free.

---

## Follow-ups that fall out of the bug fixes (not designs, just do next)

1. **ogbench submodule data fix:** `D3/10x10_dense_deadend_ky_dy_1.json` and
   `S4/10x10_dense_1.json` have `maze.goal ≠ goal.target`; this repo now
   fail-closes on them at validation. Fix upstream (set `maze.goal :=
   goal.target`, the as-run value) and bump the submodule.
2. **baseline_thinking re-run** with one equal cap ≥16384 for all three models
   (the guard now forces the choice to be explicit).
3. **d3 difficulty artifacts:** stored `optimal_steps=31` / `max_steps=93` for
   d3 were computed against the wrong goal (true optimum 23 → cap would be
   69). Analysis already uses stored eval-time values consistently; regenerate
   canonical paths before any re-run of that maze.
