# Qwen two-tier rerun — operator runbook

Reproduce the Qwen phase-1/phase-2 truncation-requeue flow end to end from the
files. **Qwen-only:** Claude and Kimi are API models with no KV-cache constraint
and run the full 64k cap from the start (via the batch lockstep runner) — they
are never two-tiered. Design and rationale:
[`docs/qwen-two-tier-rerun-design.md`](qwen-two-tier-rerun-design.md);
launch context and footguns:
[`docs/r1-run-preparation.md`](r1-run-preparation.md); served-vLLM reload
mechanics and the ~14-min reload:
[`docs/qwen-served-vllm-concurrency.md`](qwen-served-vllm-concurrency.md).

## What & why (KV-vs-parallelism)

Qwen3.6-27B runs locally on A100 via **served** vLLM. A large output budget is
expensive in KV cache: `max_model_len=96000` (needed to fit a 64k output cap)
collapses concurrency from ~13-16 episodes/server to ~2-3, wrecking the
"~4-5 hour, wide-parallel" plan. But Qwen's *observed* output demand is tiny
(median 142, p90 ~2815, max ~4921 tokens; only ~4.8% of image_only queries hit
the old buggy 4k cap). So: **run wide at a small cap, then re-run only the few
mazes that actually truncate, at 64k.**

| Phase | `max_model_len` (serve env) | `max_tokens` | timeout / attempts | concurrency | mazes | run config |
|---|---|---|---|---|---|---|
| 1 (fast/wide) | 16384 | 8000 | 1800 s / 3 | full (~13-16/server) | all | `gridworld/fixtures/run_config.r1.json` |
| 2 (deep/narrow) | 96000 | 64000 | 2400 s / 2 | reduced (~2-3/server, `--max-num-seqs 3`) | flagged only | `gridworld/fixtures/run_config.r1.qwen_phase2.json` |

Two facts that force the shape of this flow (both verified against the code):

- **Serve args are hard-coded, not config-driven.** The served-vLLM path
  ignores the run-config's `max_model_len` / `gpu_memory_utilization` /
  `enforce_eager`. The KV tradeoff is set at the serve line in
  `lib/vllm_serve_args.sh` from env knobs: `QWEN_MAX_MODEL_LEN` (default 16384),
  `QWEN_MAX_NUM_SEQS` (default 64), `QWEN_GPU_MEMORY_UTILIZATION` (default 0.9).
  **Unset env reproduces the exact phase-1 string** — phase 1 needs no env.
  Switching to phase-2 args is a real server reload (~14 min), not a runtime
  toggle.
- **Two run configs, not one toggle.** `max_tokens` is baked into the run config
  and into the episode/unit hash, so phase-2 units are legitimately distinct
  from phase-1 units (the intended re-pay). Phase 1 mixes caps (Qwen 8k vs
  Claude/Kimi 64k), so it declares `"allow_unequal_max_tokens": true` (the
  documented exception to the equal-caps invariant); phase 2 is Qwen-only, so
  the caps guard auto-passes. Phase 2's `manifest` key points at the generated
  rerun manifest — which does not exist until phase 1 has run and been scanned.

Placeholder conventions (as in the other runbooks): `$SWEEP_ID` is the fleet /
artifacts namespace; `<run>` / `.runs/<run>/…` are the phase-labeled artifacts
roots; angle-bracketed tokens are yours to fill.

---

## The freeze rule (read first — money-critical)

**`SCORER_VERSION`, `PIPELINE_VERSION`, and the scorer weights MUST NOT change
between the smoke, phase 1, and phase 2 of one campaign.** Unit identity folds
these in; bumping any of them on a re-prepare churns `unit_id`s, orphans already
verified (paid) units, and makes fresh workers re-pay episodes. Verify before
each phase that these are untouched:

- `scorer/config.py::SCORER_VERSION` (currently `0.3.0`) and the scorer weights.
- `scripts/run_pipeline.py::PIPELINE_VERSION` (currently `0.1.2`).

The phase change carries its own hash difference (the `max_tokens` cap); nothing
else about scoring may move. The `phase`/`pass` provenance fields are additive
and deliberately *excluded* from the hash (`_phase_episode_provenance` in
`scripts/run_pipeline.py`), so labeling a phase never churns a cached episode.

---

## End-to-end command sequence

Set the namespace once, from a single value, so the artifacts path and the
fleet id can never diverge:

```bash
export RUN=r1-YYYYMMDD          # artifacts namespace (.runs/$RUN/…)
export SWEEP_ID="$RUN"          # fleet id (reload-qwen-phase2 RUN_ID default); same value
```

### 1. Phase 1 — run all mazes wide at 8k

Serve env **unset** = phase-1 defaults (16384 / 64 / 0.9). Qwen episodes land
under a phase-labeled artifacts root `.runs/$RUN/qwen_phase1/`. `run_pipeline`
is a *client* — it does not start a model server, and the Qwen block points at
`http://127.0.0.1:8000/v1`, so a `vllm serve` must already be healthy before it
runs. Pick the path that matches your setup; each is complete on its own.

**(a) Fleet path (normal).** Phase 1 is the standard R1 distributed launch: the
coordinator/worker driver (`sweep_run.sh provision`, then `next-batch` for the
following batches — see
[`docs/r1-run-preparation.md`](r1-run-preparation.md) §Launch sequence step 5 and
[`docs/sequential_sweep_runbook.md`](sequential_sweep_runbook.md)) SSHes each
A100 VM and `start_worker` (`lib/distributed_start.sh`) launches `vllm serve`
with the phase-1 defaults from `lib/vllm_serve_args.sh` (env unset), blocking on
`/v1/models` readiness before any episode is dispatched. This same launch carries
Claude/Kimi at 64k. **Do not export any `QWEN_*` serve env for phase 1** — the
worker reproduces the phase-1 serve string verbatim when the knobs are unset.

**(b) Local / single-VM path.** Start the phase-1 server yourself first, wait for
it to answer `/v1/models`, then run the client. The serve line is the exact
phase-1 default string rendered by `lib/vllm_serve_args.sh` (env unset):

```bash
# 1. Launch the phase-1 vLLM server (background). Serve args == vllm_serve_args() defaults.
source lib/vllm_serve_args.sh
vllm serve Qwen/Qwen3.6-27B --served-model-name Qwen/Qwen3.6-27B $(vllm_serve_args) &
#   expands to: --port 8000 --gpu-memory-utilization 0.9 --max-model-len 16384 \
#               --max-num-seqs 64 --dtype bfloat16 --trust-remote-code

# 2. Wait until the server is healthy (must return 200 before step 3).
until curl -fsS http://127.0.0.1:8000/v1/models >/dev/null; do sleep 5; done

# 3. Run the phase-1 client against the healthy server.
python -m scripts.run_pipeline \
  --run-config gridworld/fixtures/run_config.r1.json \
  --manifest   gridworld/fixtures/manifest.r1_balanced_03.json \
  --artifacts-root .runs/$RUN/qwen_phase1
```

The config's `phase` block (`{"pass": 1, "label": "phase1"}`) stamps `pass` and
the caps onto every `episode.json`.

### 2. Scan — emit the phase-2 rerun manifest

Flags episodes where any query record hit the cap (`usage.output_tokens >= cap`
or `finish_reason == "length"`; the cap is read per-run from
`run_inputs.json → model_config.max_tokens`, or override with `--cap`). Writes a
verbatim subset of the source manifest. **`--out` must be exactly the path the
phase-2 config declares as its `manifest`** — otherwise the phase-2
manifest-match guard fails:

```bash
python -m scripts.scan_truncations \
  --artifacts-root  .runs/$RUN/qwen_phase1 \
  --source-manifest gridworld/fixtures/manifest.r1_balanced_03.json \
  --out             gridworld/fixtures/manifest.r1_qwen_phase2.json \
  --model qwen36_27b_vllm
```

Fail-closed: the scanner exits non-zero if any run's cap is unresolvable, unless
you pass `--allow-unresolved` (it is a money-deciding tool). Runnable live as
phase-1 episodes complete, or post-hoc on the finished root — functionally
equivalent, because the reload forces a phase boundary regardless. If nothing
truncated, the manifest has zero tasks and phase 2 is a no-op — you are done.

### 3. Reload — switch the fleet to phase-2 serve args

On the already-running fleet, per GPU VM: `stop_gpu_worker` (fail-closed until
the GPU is free), kill any lingering `vllm serve` with a bounded wait, relaunch
with `QWEN_MAX_MODEL_LEN=96000 QWEN_MAX_NUM_SEQS=3`, drop worker concurrency, and
health-check `/v1/models`. `BATCH_CAP` is required (cost-safety parity). This is
the ~14-min reload:

```bash
BATCH_CAP=9h \
RUN_CONFIG=gridworld/fixtures/run_config.r1.qwen_phase2.json \
RUN_ID=$SWEEP_ID \
./sweep_run.sh reload-qwen-phase2
```

`reload-qwen-phase2` exports the phase-2 `QWEN_*` env itself — you do not set the
serve knobs by hand. It re-arms the on-VM watchdog at `BATCH_CAP` and aborts
fail-closed if any GPU does not free or a server does not answer `/v1/models`.

### 4. Phase 2 — rerun the flagged mazes deep at 64k

Qwen-only, against the generated rerun manifest, into a **separate**
phase-labeled root `.runs/$RUN/qwen_phase2/`:

```bash
python -m scripts.run_pipeline \
  --run-config gridworld/fixtures/run_config.r1.qwen_phase2.json \
  --manifest   gridworld/fixtures/manifest.r1_qwen_phase2.json \
  --artifacts-root .runs/$RUN/qwen_phase2
```

The config's `phase` block (`{"pass": 2, "label": "qwen_phase2"}`) stamps
`pass=2` and the 64k cap. Separate roots are mandatory: a shared run-dir would
silently clobber phase-1 episodes, and merging must be explicit (next step)
rather than relying on an accidental same-path overwrite.

### 5. Merge — later-pass-wins

Phase-2 episodes overwrite phase-1 per unit (keyed by
`task_id / agent_or_model / seed / condition / prompt_variant`); unflagged mazes
keep their phase-1 episodes. Provenance (`pass`, caps, `truncated_at_ceiling`) is
stamped on the **copies** written into the out root — the two source roots are
never mutated:

```bash
python -m scripts.merge_two_tier \
  --phase1 .runs/$RUN/qwen_phase1 \
  --phase2 .runs/$RUN/qwen_phase2 \
  --out    .runs/$RUN/qwen_merged \
  --expected-rerun-manifest gridworld/fixtures/manifest.r1_qwen_phase2.json
```

`--expected-rerun-manifest` fail-closes if any `task_id` the scan flagged is
missing from phase 2 (a dropped rerun). The merge writes a
`.runs/$RUN/qwen_merged/two_tier_merge.json` summary:
`{"total", "from_phase2", "truncated_at_ceiling": […], "cap_unresolved": […]}`.
If a phase-2 winner's ceiling status is unresolvable it fails closed by default;
`--allow-unresolved-cap` downgrades that to a stderr WARN + a `cap_unresolved`
entry (mirrors the scanner's policy).

---

## Where every artifact lands

| Artifact | Path |
|---|---|
| Phase-1 Qwen episodes | `.runs/$RUN/qwen_phase1/runs/<task>/<backend>/<model>/seed_<n>/<variant>/episode.json` (each stamped `pass=1`, `max_tokens=8000`) |
| Phase-2 rerun manifest | `gridworld/fixtures/manifest.r1_qwen_phase2.json` (subset of `manifest.r1_balanced_03.json`; the phase-2 config's `manifest` key) |
| Phase-2 Qwen episodes | `.runs/$RUN/qwen_phase2/runs/…/episode.json` (each stamped `pass=2`, `max_tokens=64000`) |
| Merged results | `.runs/$RUN/qwen_merged/` (episode copies, later-pass-wins) |
| Merge summary | `.runs/$RUN/qwen_merged/two_tier_merge.json` |

The phase-labeled roots are self-documenting: `qwen_phase1` / `qwen_phase2` in
the path is the phase, and each `episode.json` records its `pass` + caps.

## Terminal case: `truncated_at_ceiling`

**64k is the ceiling — there is no third tier.** If a maze *still* hits
`output_tokens >= 64000` (or `finish_reason == "length"`) in phase 2, the merge
keeps the episode and marks it `truncated_at_ceiling: true` rather than looping.
These `task_id`s are listed in `two_tier_merge.json` — **report them, do not
retry**: a genuinely unbounded Qwen thinking loop is a finding, not a bug to
retry forever. The name is distinct on purpose — `truncated` /
`end_reason == "truncated"` already mean *environment* truncation.

---

**Verified against** commit `0367b0d` (`feature/early_terminate`): both
`gridworld/fixtures/run_config.r1.json` and
`gridworld/fixtures/run_config.r1.qwen_phase2.json` read as quoted (phase blocks,
caps, timeouts, `allow_unequal_max_tokens`, `max_in_flight`); `--help` exits 0
for `scripts.scan_truncations`, `scripts.merge_two_tier`, and
`scripts.run_pipeline`, and every flag shown above is present in their argparse;
`sweep_run.sh reload-qwen-phase2` exists (dispatch + `cmd_reload_qwen_phase2`,
requires `BATCH_CAP` + `RUN_CONFIG`, health-checks `/v1/models`);
`lib/vllm_serve_args.sh` renders the serve line from `QWEN_MAX_MODEL_LEN` /
`QWEN_MAX_NUM_SEQS` / `QWEN_GPU_MEMORY_UTILIZATION` (defaults 16384 / 64 / 0.9);
the `--out` value this runbook passes to the scan and the phase-2 config's
`manifest` key both equal `gridworld/fixtures/manifest.r1_qwen_phase2.json`
(`--out` has no argparse default — omitting it makes the scan a summary-only dry
run that writes no manifest); `SCORER_VERSION` (`scorer/config.py`)
and `PIPELINE_VERSION` (`scripts/run_pipeline.py`) exist as named.
