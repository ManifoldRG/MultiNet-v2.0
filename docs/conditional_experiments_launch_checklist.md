# Conditional Experiments — Launch Checklist

Single source of truth for the conditional-experiments launch. Check items off as
they land. ⚠️ marks a decision needed before the dependent work is correct.

## 0. Naming (agreed terms)

| Term | Maze set | Manifest |
|---|---|---|
| **Phase 1 full run** | OGBench 50 (S/M/B/D selection) | `manifest.ogbench_50_smbd.json` |
| **Conditional evaluation** | validation_10 + 5 (1 S, 1 M, 1 B, 2 D) = 15, incl. the blind probe maze | `manifest.conditional_eval.json` (NEW) |
| **Smoke eval** | 3 mazes, orchestration only | `manifest.smoke_eval.json` (NEW) |

## 1. Experiment matrix — 6 condition sets → 14 variants

Per the new spec. "Baseline" = the no-op variant (== default `ExperimentConfig`);
the baseline is shared across all sets, so the 14 slots are **9 unique configs**.

| # | Set (`--conditions`) | Variants (target) | Baseline | Implemented today | Gap |
|---|---|---|---|---|---|
| 1 | Prompt | minimal, **standard**, verbose (3) | standard | standard, verbose | add `minimal` (PR #23) |
| 2 | Observation format | image_text, **image_only** (2) | image_only | image_only, text_only, image_text | **D1**: run image_only + image_text; keep `text_only` implemented but omit from the launch rollout (no code change) |
| 3 | Context window | **current**, last3, text_summary (3) | current | current, last3 (text_summary = not impl) | wire `text_summary` (PR #23 history summary) |
| 4 | **Action space** (NEW) | **egocentric**, cardinal (2) | egocentric | none — no config knob | add `action_space` knob + cardinal translation + new condition set; re-introduce cardinal (removed in 410f5f7) |
| 5 | Querying strategy | **step_by_step**, subgoal, full_trajectory (3) | step_by_step | step_by_step, subgoal, full_trajectory | **D2**: run all 3 — no code change |
| 6 | In-context learning | **zero_shot**, 1-shot (2) | zero_shot | set marked not-impl; one_shot not-impl | enable set; wire `one_shot` (PR #23); add ICL example trajectories |

**Counts** (3 models = qwen[local] + kimi + claude; 15 mazes):
- Variant-slots: 3+2+3+2+3+2 = **15**; unique configs after baseline dedup: **10**.
- Episode-cells (dedup): 10 × 3 × 15 = **450** (paid = kimi+claude only: 10 × 2 × 15 = **300**).
- Without dedup: 15 × 3 × 15 = 675 (paid 450). Use `--prompt-variant` dedup (see
  `docs/validation10_condition_sweep_rollout.md`).

## 2. Dependencies to merge (into `Distributed-run-pipeline`, never main)

- [x] Merge **PR #23** (`history_summary`): minimal prompt (Set 1), history/text
      summary (Set 3 `text_summary`), one-shot example (Set 6 `one_shot`).
      Merged into `Distributed-run-pipeline`; resolved a same-cell-pickup
      runtime↔solver conflict (see [[same-cell-key-pickup-invariant]]).
- [x] Confirm S/M/B/D fixture mazes exist for the conditional set. S/M/D pulled
      from ogbench (all sources verified on disk); the **B** maze is the
      hand-authored held-out blind probe (see §4).

## 3. Code implementation

- [x] **Set 1**: `minimal` variant registered in `condition_set_1_prompt.py`
      (delivered by PR #23).
- [x] **Set 2** (D1 = run 2): no registry change; `text_only` stays implemented,
      omitted from the launch rollout (handled via `--prompt-variant` in the
      rollout doc, not the run-config).
- [x] **Set 3**: `text_summary` implemented in `condition_set_3_context_window.py`
      and wired to `ExperimentConfig.context_window="text_summary"` (PR #23).
- [x] **Set 4 (NEW Action space)**: added `action_space` knob; cardinal vocabulary
      (MOVE_NORTH/SOUTH/EAST/WEST, PICKUP, INTERACT, DONE) in
      `interface/action_space.py` with facing-relative expansion in the runner
      (each primitive = 1 step; `cardinal_action` provenance recorded);
      `condition_set_4_action_space.py` registered in `CONDITION_SETS`.
- [x] **Set 5** (D2 = run all 3): no code change; all three querying variants run.
- [x] **Set 6**: ICL set enabled with `one_shot` (PR #23); disjointness test added
      (`tests/test_one_shot_solution.py`) asserting the example uses no eval maze.
- [x] Update `CONDITION_SETS` registry + `condition_variant_names` so all 6 sets /
      15 variant-slots resolve; `variant.name` globally unique (locked by tests).
- [x] Update `docs/validation10_condition_sweep_rollout.md` for 6 sets / new counts
      and the conditional_eval manifest, and the dedup coverage tests.

## 4. Mazes & manifests

- [x] **Generate the blind box probe maze**: hand-authored held-out 11th B
      (blind) maze `mazes/conditional/blind_probe_B_holdout.json` — keys/doors
      uniform grey, switches uniform white (no white key/door in the MiniGrid
      palette), 4 doors / 2 keys / 2 switches with positional+consumption decoys.
      Genuine wrong-keys/doors deferred (see [[blind-mazes-and-wrong-key-refactor]]).
- [x] **Render the blind maze in 2D** → `mazes/maze_image/conditional/blind_probe_B_holdout.png`.
- [x] **D3 = the blind probe maze is one of the 5 (S/M/B/D/D); conditional set stays 15.**
      It is the held-out **B** maze.
- [x] Create `manifest.conditional_eval.json` = validation_10 + S/M/B/D/D (B = the
      blind probe) = 15 total; all 5 added mazes disjoint from the 50-maze run.
- [x] Create `manifest.smoke_eval.json` = 3 mazes.
- [ ] Confirm/alias `manifest.ogbench_50_smbd.json` as the "Phase 1 full run".

## 5. Run configs (3 models, with the H1/H2 manifest+conditions guard)

- [x] One run-config per condition set against `manifest.conditional_eval.json`,
      each declaring `manifest` + `conditions` (new
      `run_config.conditional_*_claude_kimi_qwen.json` files; H1/H2 guard tested).
- [x] Action space and In-context learning run-configs added.
- [x] Variant trimming (Set 2 omits `text_only`) is handled in the rollout via
      `--prompt-variant`, not the run-config.
- [x] Smoke run-config `run_config.smoke_eval_qwen_kimi.json`: qwen
      `worker_count: 2`, kimi `worker_count: 1`.

## 6. Smoke / orchestration test

- [ ] 3-maze `manifest.smoke_eval.json` + run-config validating: **2 parallel Qwen
      runners + 1 Kimi runner**, and the coordinator handing the next maze to a
      runner as soon as one finishes (work-stealing).
- [ ] Decide if other coordinator capabilities need exercising (e.g. stale
      reassignment, GCS mirror, finalize) and extend the smoke if so.

## 7. Cost estimate

- [ ] Local weekend Qwen run over the 15-maze conditional set (10 unique configs ×
      15 = 150 local episodes) to get rough tokens/episode.
- [ ] Extrapolate to paid models (kimi+claude) for conditional eval (300 paid
      episodes) and to Phase 1 full run (50 mazes).

## 8. Outstanding Medium items (from the launch review)

- [ ] **M5**: Kimi `enable_thinking` is silently dropped by the agent factory
      (`run_pipeline._build_agent_from_spec` kimi branch) — wire it or document.
- [ ] **M6**: validate the 4096 `max_tokens` budget on a 1-task API smoke before the
      full run (qwen notes saw truncation/parse-failure at 4096).
- [ ] **M4**: make `_jsonable` fallback `str(value)` raise instead of stringifying,
      so a non-primitive can't poison the cross-machine cache hash.

## 9. Decisions (resolved)

- **D1** ✅ Set 2: run `image_only` + `image_text`; keep `text_only` implemented but
  omit from the launch rollout.
- **D2** ✅ Set 5: run all 3 (`step_by_step`, `subgoal`, `full_trajectory`) → Set 5 is
  3 variants, total 15 variant-slots / 10 unique configs.
- **D3** ✅ Blind probe maze is one of the S/M/B/D/D 5; conditional set stays 15.
- **D4** ✅ The per-set decision rules (deltas: <5% / >5% / >15%) live in the **final
  results doc**, which flags the chosen number for easy reference. This is **external
  to the run** — not encoded in the pipeline or monitoring.
