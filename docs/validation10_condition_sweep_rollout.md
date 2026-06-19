# validation_10 / tests 1-3 condition-sweep rollout

How to launch the prompt-condition sweep across **Qwen3.5-27B (local), Kimi, and
Claude** so that every prompt variable is covered **once**, without re-paying for
the shared baseline in every condition set.

- Manifest: `gridworld/fixtures/manifest.json` (the 16 validation_10 + tests 1-3 rows).
- Run configs: `gridworld/fixtures/run_config.validation10_*_claude_kimi_qwen.json`
  (the four files share an identical model list; the only thing that differs is the
  `description`).
- The prompt axis is selected by the **`--conditions` CLI flag**, *not* by the
  run-config filename. Passing the wrong `--conditions` silently runs the wrong
  axis — always set it explicitly, and always pass `--manifest` explicitly
  (it otherwise defaults to `manifest.json`, which happens to be right here, but
  do not rely on the default).

Coverage and the dedup invariant below are locked by tests in
`tests/test_run_pipeline.py`:
`test_launch_condition_sets_expose_expected_variants`,
`test_baseline_variant_of_every_launch_set_is_the_default_config`,
`test_dedup_rollout_covers_every_unique_variant_config_once`, and
`test_distributed_prepare_honors_prompt_variant`.

## Variant inventory

Each condition set contains one **baseline** variant whose `config_overrides` are
empty, so it builds the identical default `ExperimentConfig`. These four baselines
are byte-identical prompts and (at `temperature 0.0`) deterministic — they are the
*same run*:

```
standard  ≡  image_only  ≡  current  ≡  step_by_step   (= default ExperimentConfig)
```

| Condition set (`--conditions`) | Variant (run-dir name) | Kind | Implemented |
|---|---|---|---|
| `Prompt` | `standard` | **baseline** | yes |
| `Prompt` | `verbose` | distinct | yes |
| `Observation format` | `image_only` | baseline (≡ standard) | yes |
| `Observation format` | `text_only` | distinct | yes |
| `Observation format` | `image_text` | distinct | yes |
| `Context window` | `current` | baseline (≡ standard) | yes |
| `Context window` | `last3` | distinct | yes |
| `Context window` | `text_summary` | distinct | **no** (not implemented) |
| `Querying strategy` | `step_by_step` | baseline (≡ standard) | yes |
| `Querying strategy` | `subgoal` | distinct | yes |
| `Querying strategy` | `full_trajectory` | distinct | yes |
| `In-context learning` | `zero_shot` / `one_shot` | — | **no** (whole set not implemented) |

Distinct configs to cover = **7**: the shared baseline plus `verbose`, `text_only`,
`image_text`, `last3`, `subgoal`, `full_trajectory`.

## Deduplicated rollout (covers all 7 once)

Run the **shared baseline once** (it comes for free with the `Prompt` set, which
also gives `verbose`), then run only the **non-baseline** variants of the other
three sets via `--prompt-variant`. Each batch is a separate distributed job, so
give each its own `--artifacts-root`.

| Batch | `--conditions` | `--prompt-variant` | Variants produced | Artifacts root |
|---|---|---|---|---|
| 1 | `Prompt` | _(none — all)_ | `standard` (baseline), `verbose` | `artifacts/val10/prompt` |
| 2 | `Observation format` | `text_only` | `text_only` | `artifacts/val10/obs_text_only` |
| 3 | `Observation format` | `image_text` | `image_text` | `artifacts/val10/obs_image_text` |
| 4 | `Context window` | `last3` | `last3` | `artifacts/val10/ctx_last3` |
| 5 | `Querying strategy` | `subgoal` | `subgoal` | `artifacts/val10/qry_subgoal` |
| 6 | `Querying strategy` | `full_trajectory` | `full_trajectory` | `artifacts/val10/qry_full_trajectory` |

Total paid (Kimi + Claude) episodes ≈ 7 variants × 16 tasks × 1 seed × 2 API
models = **224**, vs **320** for the naive four-set rollout (~96 redundant
baseline episodes avoided).

### Analysis note

The baseline rollout lives in **batch 1** under the `standard/` run dir
(`artifacts/val10/prompt/runs/<task>/minigrid/<model>/seed_0/standard/`). When you
compare *within* the `Observation format`, `Context window`, or `Querying strategy`
axes, use that `standard/` rollout as the baseline cell — there is no
`image_only/`, `current/`, or `step_by_step/` directory in batches 2-6 by design.

## Per-batch distributed flow

For each batch, run the standard distributed roles against that batch's
`--artifacts-root` (prepare → serve → workers/api-client → finalize). Prepare
overwrites `plan.json` per artifacts-root, which is why each batch needs its own.

```bash
# Batch 1 — Prompt (produces the shared baseline + verbose)
multinet-run-pipeline --distributed-role coordinator-prepare \
  --run-config gridworld/fixtures/run_config.validation10_prompt_claude_kimi_qwen.json \
  --manifest   gridworld/fixtures/manifest.json \
  --conditions "Prompt" \
  --seeds 0 \
  --artifacts-root artifacts/val10/prompt --run-set-id val10_prompt \
  --difficulty-max-static-score <MAX>

# Batch 2 — Observation format, non-baseline only
multinet-run-pipeline --distributed-role coordinator-prepare \
  --run-config gridworld/fixtures/run_config.validation10_observation_format_claude_kimi_qwen.json \
  --manifest   gridworld/fixtures/manifest.json \
  --conditions "Observation format" --prompt-variant text_only \
  --seeds 0 \
  --artifacts-root artifacts/val10/obs_text_only --run-set-id val10_obs_text_only \
  --difficulty-max-static-score <MAX>

# ...batches 3-6 follow the same pattern with the rows in the table above.
# Then per batch: coordinator-serve, worker / coordinator-run-api-client, coordinator-finalize.
```

### Durable storage (GCS)

The coordinator stores artifacts on its own (possibly ephemeral) disk. Pass
`--storage-config gridworld/fixtures/storage_config.example.json` (with a real
`bucket`) to **coordinator-serve** and **coordinator-finalize** so the coordinator
mirrors each verified run dir to `<bucket>/<run_set_id>/<run_dir>` as it is
produced (via `gsutil`), and the aggregate (`episode_runs.jsonl` + `reports/`) at
finalize. A failed mirror never drops a paid run — the unit is flagged
`gcs_pending` and re-pushed at finalize. Without a configured bucket, mirroring is
a no-op (local/dev runs are unaffected).

## Simpler alternative (no dedup)

If the operational overhead of six jobs is not worth it, run the four condition
sets as four full jobs (one per `--conditions`, no `--prompt-variant`) and accept
the ~96 redundant baseline episodes. Because the baselines are deterministic at
`temperature 0.0`, the repeats carry no additional signal — treat the four
per-set baselines as one cell in downstream analysis.
