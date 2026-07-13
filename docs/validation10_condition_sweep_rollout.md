# Conditional condition-sweep rollout

How to launch the **7-set** condition sweep across **Qwen3.6-27B (local,
FP16/vLLM), Kimi, and Claude** over the 15-maze conditional evaluation, covering
every prompt variable **once** without re-paying for the shared baseline in every
condition set.

## Fair default baseline (2026-07-04 rebase)

The sweep ablates **down** from a baseline the models can actually perform in, so
that changing one dependent variable measures signal, not noise off a floored
baseline (an earlier stateless/image-only default made every model loop to the
step cap — see the 2026-07-03 smoke). The default `ExperimentConfig` is:

```
prompting=standard · observation=image_text (+ current-observation description) ·
context_window=last3 (single-message history, chat_history=stateless) ·
action_space=egocentric · querying=step_by_step · in_context_learning=one_shot
```

Every set's **baseline variant is the one that equals this default**, so it is
run once (under `Prompt/standard`) and reused as the comparison cell for all
axes:

```
standard ≡ image_text ≡ last3 ≡ egocentric ≡ step_by_step ≡ one_shot ≡ single_message   (= default ExperimentConfig)
```

Each condition set then runs only the variants that **differ** from this default.

**History is two coupled knobs.** *Amount* is `context_window`
(`current`/`last3`/`text_summary`, all rendered in one stateless message) — set 3.
*Mechanism* is the new **History mechanism** set (set 7): the single-message
default vs a multi-turn rolling chat (`chat_history=rolling`, one image per prior
turn). The two are coupled in the variant overrides so history is never
double-counted (in-prompt **and** as turns).

**Models & thinking (as configured in the run-configs):**

- **Claude = Opus 4.8** (`claude-opus-4-8`, key `claude_opus`). The agent omits
  `temperature` for the Opus-4.7+ family (it 400s otherwise). The experimental
  configs run **adaptive thinking at `effort: low`**; paid `max_tokens` stays 4096.
- **Kimi = `enable_thinking: false`, `temperature: 0.6`** on the experimental
  configs (thinking-on truncates before FINAL_OUTPUT at 4096; Moonshot rejects
  temp < 0.6).
- **Qwen = Qwen3.6-27B, FP16 via vLLM** on the `qwen-fp16-80` A100-80GB image,
  `enable_thinking: true` (local). **Deferred for now** — single-stream decode of
  a 27B FP16 model on A100 is ~20–35 tok/s, so a thinking maze takes hours; the
  Qwen leg waits on a batched-rollout throughput fix (tracked separately). The
  API leg (Kimi + Claude) runs first on API-only infra (`SWEEP_TOPO=api`, no
  A100s) via `run_config.conditional_<set>_claude_kimi.json`.
- **The thinking probe** (`run_config.conditional_baseline_thinking_*`): the
  shared baseline re-run with Opus `effort: xhigh` and all models thinking-on,
  paid `max_tokens` raised to 8192 — isolates reasoning-depth value vs the
  `effort: low` sweep.

## Variant inventory

Each set's **baseline** variant builds the default `ExperimentConfig` above (its
`config_overrides` resolve to the default values); the baselines are
byte-identical prompts and (at `temperature 0.0`) the *same run*.

| Condition set (`--conditions`) | Variant (run-dir name) | Kind | Run in sweep |
|---|---|---|---|
| `Prompt` | `standard` | **baseline** | yes (batch 1) |
| `Prompt` | `minimal` | distinct | yes |
| `Prompt` | `verbose` | distinct | yes |
| `Observation format` | `image_text` | baseline (≡ standard) | as baseline |
| `Observation format` | `image_only` | distinct | yes |
| `Observation format` | `text_only` | distinct | **deferred (future point)** |
| `Context window` | `last3` | baseline (≡ standard) | as baseline |
| `Context window` | `current` | distinct (0-history) | yes |
| `Context window` | `text_summary` | distinct | yes |
| `Action space` | `egocentric` | baseline (≡ standard) | as baseline |
| `Action space` | `cardinal` | distinct | yes |
| `Querying strategy` | `step_by_step` | baseline (≡ standard) | as baseline |
| `Querying strategy` | `subgoal` | distinct | yes |
| `Querying strategy` | `full_trajectory` | distinct | yes |
| `In-context learning` | `one_shot` | baseline (≡ standard) | as baseline |
| `In-context learning` | `zero_shot` | distinct | yes |
| `History mechanism` | `single_message` | baseline (≡ standard) | as baseline |
| `History mechanism` | `multiturn` | distinct | yes |

Unique configs after baseline dedup = **12** (the shared baseline plus `minimal`,
`verbose`, `image_only`, `text_only`, `current`, `text_summary`, `cardinal`,
`subgoal`, `full_trajectory`, `zero_shot`, `multiturn`). `text_only` is deferred,
so **11 unique effort-low configs run**, plus the thinking probe.

Coverage + the dedup invariant are locked by tests in
`tests/test_run_pipeline.py`
(`test_launch_condition_sets_expose_expected_variants`,
`test_baseline_variant_of_every_launch_set_is_the_default_config`,
`test_dedup_rollout_covers_every_unique_variant_config_once`,
`test_conditional_run_configs_pair_conditional_eval_with_all_six_sets`) and the
batch list in `tests/test_sweep_state.py`.

## Deduplicated rollout (the sweep BATCHES)

Run the **shared baseline once** (free with the `Prompt` set, which also gives
`minimal` + `verbose`), then run only the **non-baseline** variants of the other
sets via `--prompt-variant`. Encoded in `scripts/sweep_state.py::BATCHES`
(batch 0 = smoke):

| Batch | `--conditions` | `--prompt-variant` | run_id |
|---|---|---|---|
| 1 | `Prompt` | _(none — all)_ → standard(baseline), minimal, verbose | `cond_prompt` |
| 2 | `Observation format` | `image_only` | `cond_obs_image_only` |
| 3 | `Context window` | `current` | `cond_ctx_current` |
| 4 | `Context window` | `text_summary` | `cond_ctx_text_summary` |
| 5 | `Action space` | `cardinal` | `cond_act_cardinal` |
| 6 | `Querying strategy` | `subgoal` | `cond_qry_subgoal` |
| 7 | `Querying strategy` | `full_trajectory` | `cond_qry_full_trajectory` |
| 8 | `In-context learning` | `zero_shot` | `cond_icl_zero_shot` |
| 9 | `History mechanism` | `multiturn` | `cond_hist_multiturn` |
| 10 | `Prompt` | `standard` (thinking probe) | `cond_baseline_thinking` |

Batch 10 uses `run_config.conditional_baseline_thinking_*`: the baseline re-run
with Opus `effort: xhigh` + all models thinking-on.

## The configs at a glance

Configs 1–11 are the deduplicated `effort: low` sweep; config 12 is the thinking
probe. Thinking columns are **Opus effort / Kimi / Qwen**.

| # | Config (run-dir) | Axis it varies | Batch | Opus | Kimi | Qwen |
|---|---|---|---|---|---|---|
| 1 | `standard` | baseline (default `ExperimentConfig`) | 1 | low | off | on |
| 2 | `minimal` | Prompt | 1 | low | off | on |
| 3 | `verbose` | Prompt | 1 | low | off | on |
| 4 | `image_only` | Observation format | 2 | low | off | on |
| 5 | `current` | Context window (amount) | 3 | low | off | on |
| 6 | `text_summary` | Context window (amount) | 4 | low | off | on |
| 7 | `cardinal` | Action space | 5 | low | off | on |
| 8 | `subgoal` | Querying strategy | 6 | low | off | on |
| 9 | `full_trajectory` | Querying strategy | 7 | low | off | on |
| 10 | `zero_shot` | In-context learning | 8 | low | off | on |
| 11 | `multiturn` | History mechanism | 9 | low | off | on |
| 12 | `standard` **+ thinking** | baseline re-run — reasoning-depth probe | 10 | **xhigh** | **on** | on |

> `text_only` (Observation format) is intentionally not in the rollout right now
> (future point). The variant stays implemented and registry-covered; to run it
> later, add one batch `Observation format --prompt-variant text_only`.

### Analysis note

The baseline rollout lives in **batch 1** under the `standard/` run dir. When you
compare *within* the `Observation format`, `Context window`, `Action space`,
`Querying strategy`, `In-context learning`, or `History mechanism` axes, use that
`standard/` rollout as the baseline cell — there is no `image_text/`, `last3/`,
`egocentric/`, `step_by_step/`, `one_shot/`, or `single_message/` directory in
batches 2–9 by design.

## Split: API-only first, Qwen later

`SWEEP_TOPO=api` selects the `run_config.conditional_<set>_claude_kimi.json`
fixtures (Kimi + Claude only, no qwen model). The topology derives 0 GPU workers,
so `provision` skips the A100 hunt and brings up only the coordinator + 2 e2 API
VMs. Run the API leg today (`QWEN_WORKER_COUNT=0`); add Qwen back via the full
`_claude_kimi_qwen` configs once the throughput fix lands.

## Manifest / difficulty

- Manifest: `gridworld/fixtures/manifest.conditional_eval.json` (15 mazes).
- Use `DIFFICULTY_MAX=1000` across all batches (launcher default) so runtime
  normalization is comparable.

## Smoke first (orchestration only)

Before the paid sweep, validate coordinator work-stealing with the smoke job
(`run_config.smoke_qwen36_kimi_claude.json` over `manifest.smoke_eval.json`,
3 mazes). It is for orchestration only, not measurement.
