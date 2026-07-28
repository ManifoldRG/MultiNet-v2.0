# R1 Kimi make-up rerun — ready-to-launch package (2026-07-28)

**Status: PREPARED, NOT LAUNCHED.** Launch is Sean's call after the Moonshot
balance gate. Everything below is built and validated on branch
`fix/drop-key-bookkeeping` (DROP feature + review fixes, 1026 tests green).

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

## Launch procedure (when approved)

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
