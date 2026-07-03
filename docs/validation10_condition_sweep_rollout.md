# Conditional condition-sweep rollout

How to launch the full **6-set** condition sweep across **Qwen3.6-27B (local,
FP16/vLLM), Kimi, and Claude** over the 15-maze conditional evaluation, covering
every prompt variable **once** without re-paying for the shared baseline in every
condition set.

**Models & thinking (as configured in the run-configs):**

- **Claude = Opus 4.8** (`claude-opus-4-8`, key `claude_opus`). The agent omits
  `temperature` for the Opus-4.7+ family (it 400s otherwise). The 10 experimental
  configs run **adaptive thinking at `effort: low`** so the prompt/observation
  manipulations remain the dominant signal; paid `max_tokens` stays 4096.
- **Kimi = `enable_thinking: true`, `temperature: 0.6`** on every config (Moonshot
  rejects temperatures below 0.6 with a 400; the agent also floors it as a net).
- **Qwen = Qwen3.6-27B, FP16 via vLLM** on the `qwen-fp16-80` A100-80GB image
  (FP8/3.5-HF retired), `enable_thinking: true` (local; no token cost). The vLLM
  memory knobs (`max_model_len: 16384`, `gpu_memory_utilization: 0.9`,
  `enforce_eager: false`) are best-effort for FP16-27B + thinking and must be
  confirmed on the image via the smoke (OOM / context-overflow risk).
- **Config #11 — baseline with full thinking**
  (`run_config.conditional_baseline_thinking_claude_kimi_qwen.json`): the shared
  baseline (`--conditions "Prompt" --prompt-variant standard`) re-run with Opus at
  **`effort: xhigh`** and all three models thinking-on, paid `max_tokens` raised to
  8192. This isolates the value of reasoning depth versus the `effort: low` sweep.

**Launching a batch via `launch_distributed.sh`:** the launcher runs
`coordinator-prepare`, which enforces the H1/H2 guard — so you **must** pass the
condition axis. Set it via the `CONDITIONS` env var (and `PROMPT_VARIANT` for the
dedup batches); the launcher threads them into `--conditions`/`--prompt-variant`:

```bash
# Batch 5 — Action space, non-baseline only
RUN_CONFIG=gridworld/fixtures/run_config.conditional_action_space_claude_kimi_qwen.json \
MANIFEST=gridworld/fixtures/manifest.conditional_eval.json \
CONDITIONS="Action space" PROMPT_VARIANT=cardinal \
MAX_RUN_DURATION=12h DIFFICULTY_MAX=1000 RUN_ID=cond_act_cardinal \
  ./launch_distributed.sh

# Batch 11 — baseline with full thinking (its own artifacts-root/run-id)
RUN_CONFIG=gridworld/fixtures/run_config.conditional_baseline_thinking_claude_kimi_qwen.json \
MANIFEST=gridworld/fixtures/manifest.conditional_eval.json \
CONDITIONS="Prompt" PROMPT_VARIANT=standard \
MAX_RUN_DURATION=12h DIFFICULTY_MAX=1000 RUN_ID=cond_baseline_thinking \
  ./launch_distributed.sh
```

Use `DIFFICULTY_MAX=1000` across all batches (this is also the launcher default) —
these runs exist to *find* the real difficulty ceiling, so a fixed high value keeps
runtime normalization consistent across every batch.

- Manifest: `gridworld/fixtures/manifest.conditional_eval.json` (10 validation_10
  mazes + 5 held-out S/M/B/D/D mazes, incl. the blind probe B maze; 15 total).
  All 5 added mazes are disjoint from `manifest.ogbench_50_smbd.json` so the
  verification sweep cannot affect the 50-maze run.
- Run configs: `gridworld/fixtures/run_config.conditional_<set>_claude_kimi_qwen.json`
  (one per condition set; the six share an identical model list and differ only
  in `description` + `conditions`).
- The prompt axis is selected by the **`--conditions` CLI flag**, *not* by the
  run-config filename. Passing the wrong `--conditions` silently runs the wrong
  axis — always set it explicitly, and always pass `--manifest` explicitly. The
  run-configs declare both `manifest` and `conditions`, so the H1/H2 guard
  (`check_run_config_expectations`) rejects a mispaired launch before any paid
  work.

Coverage and the dedup invariant below are locked by tests in
`tests/test_run_pipeline.py`:
`test_launch_condition_sets_expose_expected_variants`,
`test_baseline_variant_of_every_launch_set_is_the_default_config`,
`test_dedup_rollout_covers_every_unique_variant_config_once`,
`test_conditional_run_configs_pair_conditional_eval_with_all_six_sets`, and
`test_distributed_prepare_honors_prompt_variant`.

## Variant inventory

Each condition set contains one **baseline** variant whose `config_overrides` are
empty, so it builds the identical default `ExperimentConfig`. These six baselines
are byte-identical prompts and (at `temperature 0.0`) deterministic — they are the
*same run*:

```
standard ≡ image_only ≡ current ≡ egocentric ≡ step_by_step ≡ zero_shot   (= default ExperimentConfig)
```

| Condition set (`--conditions`) | Variant (run-dir name) | Kind | Implemented |
|---|---|---|---|
| `Prompt` | `standard` | **baseline** | yes |
| `Prompt` | `minimal` | distinct | yes |
| `Prompt` | `verbose` | distinct | yes |
| `Observation format` | `image_only` | baseline (≡ standard) | yes |
| `Observation format` | `text_only` | distinct (omitted from launch, D1) | yes |
| `Observation format` | `image_text` | distinct | yes |
| `Context window` | `current` | baseline (≡ standard) | yes |
| `Context window` | `last3` | distinct | yes |
| `Context window` | `text_summary` | distinct | yes |
| `Action space` | `egocentric` | baseline (≡ standard) | yes |
| `Action space` | `cardinal` | distinct | yes |
| `Querying strategy` | `step_by_step` | baseline (≡ standard) | yes |
| `Querying strategy` | `subgoal` | distinct | yes |
| `Querying strategy` | `full_trajectory` | distinct | yes |
| `In-context learning` | `zero_shot` | baseline (≡ standard) | yes |
| `In-context learning` | `one_shot` | distinct | yes |

Launch variant-slots = 3 + 2 + 3 + 2 + 3 + 2 = **15** (Observation format runs
`image_only` + `image_text` only; `text_only` stays implemented but is omitted
per D1). Unique configs after baseline dedup = **10**: the shared baseline plus
`minimal`, `verbose`, `image_text`, `last3`, `text_summary`, `cardinal`,
`subgoal`, `full_trajectory`, `one_shot`.

## Deduplicated rollout (covers all 10 once)

Run the **shared baseline once** (it comes for free with the `Prompt` set, which
also gives `minimal` + `verbose`), then run only the **non-baseline** variants of
the other five sets via `--prompt-variant`. Each batch is a separate distributed
job, so give each its own `--artifacts-root`.

| Batch | `--conditions` | `--prompt-variant` | Variants produced | Artifacts root |
|---|---|---|---|---|
| 1 | `Prompt` | _(none — all)_ | `standard` (baseline), `minimal`, `verbose` | `artifacts/cond/prompt` |
| 2 | `Observation format` | `image_text` | `image_text` | `artifacts/cond/obs_image_text` |
| 3 | `Context window` | `last3` | `last3` | `artifacts/cond/ctx_last3` |
| 4 | `Context window` | `text_summary` | `text_summary` | `artifacts/cond/ctx_text_summary` |
| 5 | `Action space` | `cardinal` | `cardinal` | `artifacts/cond/act_cardinal` |
| 6 | `Querying strategy` | `subgoal` | `subgoal` | `artifacts/cond/qry_subgoal` |
| 7 | `Querying strategy` | `full_trajectory` | `full_trajectory` | `artifacts/cond/qry_full_trajectory` |
| 8 | `In-context learning` | `one_shot` | `one_shot` | `artifacts/cond/icl_one_shot` |
| 9 | `Prompt` | `standard` (thinking config #11) | `standard` @ full thinking | `artifacts/cond/baseline_thinking` |

Batch 9 uses `run_config.conditional_baseline_thinking_claude_kimi_qwen.json`
(not a six-set file): the baseline re-run with Opus `effort: xhigh` and all models
thinking-on, to measure reasoning-depth value against the `effort: low` sweep.

Total paid (Kimi + Claude) episodes = 11 unique configs × 15 mazes × 1 seed × 2
API models = **330** (300 for the 10 `effort: low` configs + 30 for config #11).
Including local Qwen: 11 × 15 × 3 = **495** episode-cells. Config #11's paid
`max_tokens` is 8192 (vs 4096 for the 10), and Opus xhigh + Kimi/Qwen thinking
make it the most expensive single batch — budget for it separately.

## The 11 configs at a glance

Every unique config that runs, numbered. Configs 1–10 are the deduplicated
`effort: low` sweep; config 11 is the thinking probe. Thinking columns are
**Opus effort / Kimi / Qwen** (`max_tokens`: paid = 4096 for 1–10, 8192/16384 for
11; Qwen = 8192 throughout).

| # | Config (run-dir) | Axis it varies | Batch | Opus | Kimi | Qwen |
|---|---|---|---|---|---|---|
| 1 | `standard` | baseline (default `ExperimentConfig`) | 1 | low | off | on |
| 2 | `minimal` | Prompt | 1 | low | off | on |
| 3 | `verbose` | Prompt | 1 | low | off | on |
| 4 | `image_text` | Observation format | 2 | low | off | on |
| 5 | `last3` | Context window | 3 | low | off | on |
| 6 | `text_summary` | Context window | 4 | low | off | on |
| 7 | `cardinal` | Action space | 5 | low | off | on |
| 8 | `subgoal` | Querying strategy | 6 | low | off | on |
| 9 | `full_trajectory` | Querying strategy | 7 | low | off | on |
| 10 | `one_shot` | In-context learning | 8 | low | off | on |
| 11 | `standard` **+ thinking** | baseline re-run — reasoning-depth probe | 9 | **xhigh** | **on** | on |

Configs 1 and 11 share the run-dir name `standard` but live in different
artifacts roots (`artifacts/cond/prompt/…standard/` vs
`artifacts/cond/baseline_thinking/…standard/`), so they never collide. Config 11
is the only row where the paid models think — everything else isolates the prompt
manipulation at `effort: low` / Kimi thinking-off (thinking-on truncates; see the
**Models & thinking** note above).

> `text_only` is intentionally not in the rollout (D1). If you later want it, add
> one batch `Observation format --prompt-variant text_only`.

### Analysis note

The baseline rollout lives in **batch 1** under the `standard/` run dir
(`artifacts/cond/prompt/runs/<task>/minigrid/<model>/seed_0/standard/`). When you
compare *within* the `Observation format`, `Context window`, `Action space`,
`Querying strategy`, or `In-context learning` axes, use that `standard/` rollout
as the baseline cell — there is no `image_only/`, `current/`, `egocentric/`,
`step_by_step/`, or `zero_shot/` directory in batches 2-8 by design.

## Per-batch distributed flow

For each batch, run the standard distributed roles against that batch's
`--artifacts-root` (prepare → serve → workers/api-client → finalize). Prepare
overwrites `plan.json` per artifacts-root, which is why each batch needs its own.
Pass `--difficulty-max-static-score 1000` where `1000` is the stable maximum for
the conditional set (derive once from the conditional_eval static scores and reuse
it across all batches so runtime normalization is comparable).

```bash
# Batch 1 — Prompt (produces the shared baseline + minimal + verbose)
multinet-run-pipeline --distributed-role coordinator-prepare \
  --run-config gridworld/fixtures/run_config.conditional_prompt_claude_kimi_qwen.json \
  --manifest   gridworld/fixtures/manifest.conditional_eval.json \
  --conditions "Prompt" \
  --seeds 0 \
  --artifacts-root artifacts/cond/prompt --run-set-id cond_prompt \
  --difficulty-max-static-score 1000

# Batch 5 — Action space, non-baseline only
multinet-run-pipeline --distributed-role coordinator-prepare \
  --run-config gridworld/fixtures/run_config.conditional_action_space_claude_kimi_qwen.json \
  --manifest   gridworld/fixtures/manifest.conditional_eval.json \
  --conditions "Action space" --prompt-variant cardinal \
  --seeds 0 \
  --artifacts-root artifacts/cond/act_cardinal --run-set-id cond_act_cardinal \
  --difficulty-max-static-score 1000

# ...batches 2-4, 6-8 follow the same pattern with the rows in the table above.
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

## Smoke first (orchestration only)

Before the paid sweep, validate coordinator work-stealing with the smoke job:
`run_config.smoke_eval_qwen_kimi.json` over `manifest.smoke_eval.json` (3 mazes,
2 Qwen runners + 1 Kimi runner, no Claude). It is for orchestration only, not
measurement.

## Simpler alternative (no dedup)

If the operational overhead of eight jobs is not worth it, run the six condition
sets as six full jobs (one per `--conditions`, no `--prompt-variant`) and accept
the redundant baseline episodes. Because the baselines are deterministic at
`temperature 0.0`, the repeats carry no additional signal — treat the six per-set
baselines as one cell in downstream analysis.
