# R1 Kimi make-up rerun — ready-to-launch package (2026-07-28, rev 2026-07-30)

**Status: PREPARED, NOT LAUNCHED.** Launch is Sean's call after the Moonshot
balance gate. Everything below is built and validated on branch
`fix/drop-key-bookkeeping` (DROP feature + review fixes, tests green).

## Design: three parallel arms (rev 2026-07-30)

| Arm | Maze | Harness | Purpose |
|---|---|---|---|
| 1 | M6 14×14 — **resume from step 143** | **as-run code (no DROP)**, via worktree @ `e0542b5` + `scripts/resume_from_archive.py` | Finish the infra-killed R1 episode under its original condition — corpus-grade completion (with the Kimi snapshot-drift caveat) |
| 2 | M6 14×14 — fresh from step 1 | DROP-enabled branch | Make-up episode under the new harness; vs arm 1, a read on whether the prompt change (DROP in the vocabulary) shifts behavior on a maze where DROP is functionally irrelevant |
| 3 | D2 8×8 (wrong-key) — fresh | DROP-enabled branch | The actual DROP test: does Kimi avoid or recover from the decoy-key trap |

Interpretation caveats, stated up front: n = 1 per arm at temperature 1.0,
so treat outcomes as sanity signals, not tests. Arm 1 is *strictly easier*
than arm 2 (door already open, red key already collected, 175 of 318 steps
of budget remaining), so "arm 1 succeeds, arm 2 fails" is over-determined —
it cannot cleanly be attributed to DROP. M6 contains **no decoy keys**
(red and blue are both required), so DROP should never rationally fire in
arms 1–2; if arm 2's transcript shows DROP usage on M6, that itself is a
finding (prompt-induced action noise). Arm 3 is where DROP can genuinely
matter.

## What gets rerun, and why (evidence)

Exactly two Kimi episodes from R1 are invalid for model-capability reasons:

1. **`r1_M6_14x14_dense_kr_sg_kb_1`** — ended `parse_failed`: a
   Moonshot-outage infra termination, not a model outcome (already excluded
   from R1 failure stats; the × in every figure).
2. **`r1_D2_8x8_corridor_wrong_ky_inactive_sb_sg_kr_1`** — the DROP-unfair
   episode. Transcript evidence (from
   `Multinet-v2-results/r1-20260717/runs/.../kimi-k2.6/.../episode.json` and
   `metrics/long_table.csv`): at env step 23 Kimi picked up the **yellow
   decoy key**; the maze's door requires **red**; carry capacity is 1 and
   the R1 harness had no DROP action, so from step 23 the episode was
   mechanically unwinnable (`doomed=True, doomed_at_env_step=23` — the only
   doomed episode in the entire 150-episode corpus). Kimi then logged 13
   BLOCKED/NOTHING events while stuck holding the wrong key and died
   `stalled` at step 58.

No other Kimi episode (and no Claude/Qwen episode) is doomed, so the rerun
set is complete at these two.

## Fixtures (committed, validated)

- `gridworld/fixtures/manifest.r1_kimi_rerun.json` — the 2 tasks, copied
  verbatim from `manifest.r1_balanced_03.json`, provenance in `selection`.
  Validated: `python -m scripts.validate_fixtures --manifest
  gridworld/fixtures/manifest.r1_kimi_rerun.json` → OK.
- `gridworld/fixtures/run_config.r1.kimi_rerun.json` — Kimi only, identical
  fixed cell to R1 (`minimal` / `image_only` / `egocentric` / `zero_shot` /
  `text_summary_and_last3` + `chat_history=stateless`, `progress_stall_k=30`,
  thinking ON, `max_tokens=64000`, temperature 1.0 — Moonshot mode-forces
  it), `max_in_flight=2`.

## Comparability caveats (state these wherever results are merged)

- **DROP changes the cell.** The rerun runs with DROP in the action
  vocabulary and prompt (that is the point for the D2 maze), so these two
  episodes are *not* prompt-identical to the R1 corpus — same deliberate
  incomparability class as the last3 prompt change flagged in the
  early-terminate review. Report them as make-up episodes with a footnote,
  not as silent replacements.
- **Stall-watchdog fix rides along.** The review fix adds `key_positions` to
  the progress signature (commit `5617ca3`) so a drop-and-retrace recovery
  is not counted as stalling. Replay-neutral for all DROP-free episodes.
- **Kimi snapshot drift.** R1's Kimi leg was the 2026-07-21 snapshot; a
  rerun now samples whatever Moonshot serves today. Record the date in the
  run notes (the k3-launch token-estimate banner may also still apply).

## Launch file

One command runs all three arms in parallel (after the balance precheck):

```bash
MOONSHOT_API_KEY=... scripts/launch_kimi_rerun.sh <artifacts-root>
```

It gates on: key present → clean committed tree on this branch → a FREE
deterministic replay-verify of the archived episode (143/143 steps must
reproduce byte-identically through the `e0542b5` worktree at
`/tmp/multinet-r1-asrun`; the script creates the worktree if missing). Then
arm 1 (`scripts/resume_from_archive.py --mode continue`) and arms 2+3
(`run_pipeline` on the 2-maze fixtures) run as parallel background jobs,
logs under `<artifacts-root>/logs/`, and a final summary prints
end_reason/success/steps and any DROP actions per arm.

Resume-arm facts, verified 2026-07-30: replay of all 143 archived steps
reproduces the archive exactly (final position (row,col)=(8,6), red key
collected, first door open, stall counter only 2/30, ~175 steps of cap
left). The outage shows up in-archive as empty replies (3 earlier blips at
queries 13/15/68 plus the killing streak); the resume clears the trailing
parse-failure state and re-issues the round. Note: the archive itself lost
52 late-episode frame PNGs (queries 134–146, never pulled before the
outage) — the resume records explicit placeholders for those.

## Manual launch procedure (equivalent, if you prefer step-by-step)

```bash
# 0) Gate: Moonshot balance precheck (the batch precheck killed the leg
#    once before at $0 spent — check balance BEFORE launching):
#    https://platform.moonshot.ai console → balance covers ~$20 headroom.
export MOONSHOT_API_KEY=...   # or .env per deploy/key resolution

# 1) From repo root on fix/drop-key-bookkeeping (code-sync invariant:
#    committed SHA, clean tree):
git status --short   # must be clean
python -m pytest -q  # expect 1026 passed

# 2) The run (local, API-only — no fleet needed for 2 mazes):
python -m scripts.run_pipeline \
  --run-config gridworld/fixtures/run_config.r1.kimi_rerun.json \
  --manifest  gridworld/fixtures/manifest.r1_kimi_rerun.json \
  --seeds 0

# 3) Artifacts: <artifacts-root>/runs/<task>/minigrid/kimi-k2.6/seed_0/...
#    Verify both episodes end with a real end_reason (not parse_failed),
#    then hand episode.json paths to the analysis merge.
```

## Cost estimate

R1 actuals: ~$150 for 49 completed Kimi episodes ≈ **$3/episode** (64k
thinking, ~20k output tokens/query median). The two mazes bracket the size
range (8×8 corridor ≈ cheap, 14×14 dense ≈ expensive):
**expected ≈ $6, budget $20 ceiling.** No GPU cost (API only).

## After the run

- Merge the two episodes into the analysis with the caveats above;
  regenerate figures (`python -m pubfigs.figures && python -m pubfigs.axes`
  in `Multinet-v2-results/r1-20260717/analysis/`) — the × marker and
  doomed-episode footnotes drop out automatically once the rows are
  replaced and `INFRA`/doomed handling is updated in `pubfigs/data.py`.
- Update the Kimi 49/50 completeness banner in the Notion bundle
  (`Parity & Data Integrity`) and rebuild the zip
  (`python scripts/build_notion_r1.py`).
