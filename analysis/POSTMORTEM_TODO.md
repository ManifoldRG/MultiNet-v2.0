# R1 (r1-20260717) — Post-mortem action checklist

Standalone checklist extracted from `analysis/r1_full_fixes.md` (the full
incident log). Each item cites its source section there. Ordered by severity /
blocking-ness for the next run.

## 🔴 P0 — data-validity blockers (decide before trusting or re-running)

- [ ] **Fix the image_only dynamic-mechanism render bug** (§MAJOR BUG). Doors,
      switches, and gates render their INITIAL glyph only — runtime state
      (door open/closed, switch on/off, gate open/closed) is never shown.
      Confirmed: red-door cell pixel-identical open vs closed (diff 0.0);
      `Switch.render()` ignores `is_active`; `Gate` extends `Door`. Fix the
      renderer to draw open doors/gates distinctly (MiniGrid convention:
      hollow/outline) and switch on/off distinctly.
  - [ ] Root-cause: is `is_open`/`is_active` not updated on the grid object, or
        does `render()` not read it? (Confirmed the *symptom*, not the cause.)
  - [ ] Repro path: bare `load_task_from_file(spec).reset()` shows only
        Wall+Goal — use the runtime's full build path or drive a real episode.
        Method: crop mechanism cell (`cell_px = frame_w // grid_w`) pre/post
        interaction and diff.
- [ ] **Caveat ALL image_only mechanism-task results** in any writeup until the
      render fix + re-run. 4/6 solves still cleared key-doors, so results are
      depressed-but-real, not noise — but solve rates are a lower bound.
- [ ] **Decide: image_only re-run** after the render fix (+ rotation fix below).
      Current image_only mechanism data measures models fighting a broken
      observation.
- [ ] **Make `text_summary_and_last3` the primary mechanism comparison** — it
      carries mechanism-state via feedback text, so it's the clean read on
      whether models can do mechanisms at all (image_only can't show state).

## 🟠 P1 — correctness bugs surfaced this run

- [ ] **Rotation off-by-one** (§RUN-LEVEL ASTERISK). User read frames as the
      agent's facing/rotation offset by one; suspected regression of a
      previously-fixed bug. Check facing-vs-frame AND action→direction mapping.
      Interacts with the render bug — both hit navigation.
- [ ] **Transcript `position_before`/`position_after` are (row,col), not (x,y)**
      (§coordinate bug). They equal `state.position_row_col`; the authoritative
      x,y is `state.agent_position`. Fix logging to (x,y) or rename fields; and
      **audit any analysis that read `position_*` as x,y** (it's transposed).
- [ ] **Infra failures must not consume the parse-retry budget** — FIXED this
      run (commit `8b8e7e5`: `_INFRA_STOP_REASONS` excluded from the terminator;
      token-cap truncation still counts). Verify the fix ships/merges; it was
      the root cause of the launch dead-letter, both Moonshot outages, and the
      final-maze false termination (§3/§12/§16/§20/§21).
- [ ] **Batch-level provider failure must not masquerade as per-item parse
      failures** (§3/§12). Detect batch `status=failed`/deadline-cancel and
      pause+retry+alert instead of burning unit attempts. (§21 fix covers the
      per-item path; the batch-submission path still needs this.)

## 🟡 P2 — operational hardening

- [ ] **Provider preflight before paid runs**: Moonshot balance
      (`GET /v1/users/me/balance`) AND project consumption budget — the balance
      precheck killed the launch (§3) and the consumption cap tripped mid-run
      (§16). Add both to the launch checklist.
- [ ] **Smoke must include ≥1 full-size batch** (not just 1-request) to exercise
      the balance reservation (§3).
- [ ] **Default Kimi timeout > 600s** for deep-thinking runs, and **remove the
      temporary `KIMI_TIMEOUT_OVERRIDE` env** (§Operator flags / §20) — fold the
      proper default into run_config/KimiK26Config.
- [ ] **Kimi sync-transport bridge** (`KIMI_BATCH_TRANSPORT=sync`, commit
      `e0542b5`) — decide if it stays as a supported fallback or is removed.
- [ ] **Plumb `--stale-after-seconds` through `start_coordinator`** (§2) — it is
      hard-omitted (defaults to 300, wrong for batch); had to be re-added by
      hand on every serve restart.
- [ ] **Watchdog discipline** (§17/§22): extend with WIDE margin the moment a
      long/blocking wait begins, never at the deadline; ALWAYS verify fire-time
      via `/run/systemd/shutdown/scheduled`, never mental math. Two stops this
      run were self-inflicted timing misses.
- [ ] **`cs_stop_vms` silently fails on a2-ultragpu** (needs
      `--discard-local-ssd`; the 400 is swallowed by `|| true`) (§8). Fix.
- [ ] **After a mid-turn worker kill, reset the unit to `pending`** — the
      coordinator holds it assigned to the dead worker until the stale timer;
      the "kick" only works with the explicit reset (learned live, §kick).
- [ ] **LPT-order the phase-2 rerun** (§15): plain `coordinator-prepare`
      preserves manifest order (ascending), so the biggest mazes queued LAST —
      anti-LPT worst-case tail. LPT-order units or emit the rerun manifest
      descending. (Order is hash-free — safe to change.)
- [ ] **Finalize/publish must include the `runs/` raw subtree** — the R1
      finalize committed only the aggregate jsonls + docs; the `runs/` per-episode
      tree that every `raw_output_ref` points at never landed (all 150 refs
      dangling, caught post-hoc). Publish must copy `runs/` too AND verify each
      `raw_output_ref` resolves before declaring a run published. The aggregate
      looking complete masked the gap. (Fixed for R1 in results commit
      `6d5334b09`, assembled from the outcome-matched canonical sources —
      `claude_pulled` / `kimi_final` / `qwen_merged`; note NOT the stale
      `kimi_pulled_final`.)

## 🟢 P3 — analysis follow-ups

- [ ] **Qwen "explores 2–4× longer, solves no more"** (§eval). Median episode
      ~2× Claude's; 3 env-truncations to 132 steps; still 1 solve. Confirm from
      `env_step_count` (NOT `query_count`, which overcounts by parse-failures).
- [ ] **Can the progress-aware stall rule be gamed by wandering?** (§analysis
      questions) — the env-truncated episodes reset stall-K on new tiles without
      converging. Compare coverage-over-time vs distance-to-goal for
      env-truncated vs stalled vs solved. Feeds the per-tile step-cap redesign.
- [ ] **Claude thinking depth vs `effort: xhigh`** (§analysis questions) —
      median only ~442 out-tok/turn (27% of turns ~24 tok, no thinking block)
      vs Kimi/Qwen ~15–20k. Is adaptive thinking skipping trivial moves, or is
      image_only failing to trigger deliberation?
- [ ] **Compile the outage timeline** (§action item) — exact UTC+CEST
      timestamps/durations of every incident (2 Moonshot batch outages, quota
      trip, engine_overloaded window, hung streams, watchdog stops) to quantify
      the wall-clock cost of Moonshot's launch-week instability.
- [ ] **Full cost reconciliation** — Claude ~$31; Kimi ~$150 (batch + sync, 2
      resumes); Qwen $0 tokens; A100 ~165 VM-hours ≈ $775–840. Produce a clean
      final tally from token + VM-hour data.

## Data-integrity notes for whoever loads the results

- Final dataset: `Multinet-v2-results/r1-20260717/R1_FINAL_episode_runs.jsonl`
  (150 rows; Qwen rows are `pass=2`, two-tier merged).
- **Asterisk exactly 1 row:** kimi `r1_M6_14x14_dense_kr_sg_kb_1`
  (`end_reason=parse_failed` = Moonshot-outage infra termination after real
  progress — opened the red door — NOT a model/stall outcome).
- **1 `truncated_at_ceiling`:** qwen `r1_D2_14x14_dense_wrong_ky_kr_inactive_sb_sg_kb_1`
  (still hit 64k in phase 2 — genuine finding, not a bug).
- Kimi episodes span batch AND sync transport (switch mid-campaign); Qwen
  phase-1 vs old corpora are prompt-non-comparable (temp 0.6→1.0). Don't pool
  silently.
