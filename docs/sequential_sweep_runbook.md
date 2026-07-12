# Sequential Supervised Sweep — Operator Runbook

Runs the **11-batch** conditional sweep (smoke + 10 conditional configs) as one
supervised sequence on a **single reused fleet** — 3 Qwen (A100-80GB) + 1 Kimi +
1 Claude + 1 coordinator (n2). The fleet is provisioned once; each batch is
(re)started on the same VMs. VMs are **STOPped, never deleted**; artifacts live on
the coordinator's 500 GB disk and are egressed to the operator before each advance.

The sequencer (`sweep_run.sh`) only orchestrates — all cost-safety logic lives in
the tested `lib/cost_safety.sh`, `lib/distributed_start.sh`, and `launch_distributed.sh`.

---

## Two duration caps (do NOT conflate)

| knob | meaning | value | enforced by |
|---|---|---|---|
| `MAX_RUN_DURATION` | GCP provision ceiling (backstop; every VM STOPs at this age) | `120h` | `--max-run-duration` at create |
| `BATCH_CAP` | per-batch on-VM watchdog, re-armed each `next-batch` | `6h` | `arm_watchdog … 0` (fail-closed) |

`provision` passes `MAX_RUN_DURATION` to the launcher. `next-batch` re-arms the
watchdog with `MAX_RUN_DURATION="$BATCH_CAP"` scoped to that call only. Never set
the GCP ceiling to `BATCH_CAP`.

## The 11 batches

`n=0` smoke → then 10 conditional batches. Each ablates one variable DOWN from the
fair default baseline (image_text · last3 single-message · egocentric · one_shot;
see `docs/validation10_condition_sweep_rollout.md`). Order and weights (relative
runtime, for the live-calibrated ETA) are in `scripts/sweep_state.py::BATCHES`:

| n | run_id | conditions / variant | weight |
|---|---|---|---:|
| 0 | smoke | 3-model fleet smoke (non-conditional) | 0.1 |
| 1 | cond_prompt | Prompt (standard baseline + minimal + verbose) | 3.0 |
| 2 | cond_obs_image_only | Observation format / image_only | 1.0 |
| 3 | cond_ctx_current | Context window / current (0-history) | 1.0 |
| 4 | cond_ctx_text_summary | Context window / text_summary | 1.0 |
| 5 | cond_act_cardinal | Action space / cardinal | 1.0 |
| 6 | cond_qry_subgoal | Querying strategy / subgoal | 1.0 |
| 7 | cond_qry_full_trajectory | Querying strategy / full_trajectory | 1.0 |
| 8 | cond_icl_zero_shot | In-context learning / zero_shot | 1.0 |
| 9 | cond_hist_multiturn | History mechanism / multiturn | 1.3 |
| 10 | cond_baseline_thinking | Prompt / standard (thinking ON) | 3.0 |

(`text_only` is deferred to a future point — variant stays implemented, not run.)

**Batch 0 (smoke) is started by `provision` itself** (the launcher always starts a
run; we make that run the cheap 3-model smoke). Batches 1–10 are each started by
`next-batch N`. Batch 0 validates all three workers on the real fleet before any
paid conditional batch.

**Batch 10 (`cond_baseline_thinking`) runs Kimi in thinking-ON mode** → the Kimi
agent pins its temperature to **1.0** (thinking-off batches use 0.6); this is forced
in `interface/agents/kimi_k26.py`, not the config. Thinking-ON Kimi can spend its
whole token budget on reasoning and truncate on hard mazes even at `max_tokens
16384` — expect some Kimi `parse_failed` on batch 10 and do not mistake it for a
fleet fault.

---

## 1. Preflight

```bash
set -a; source .env; set +a                              # .env is NOT auto-sourced
: "${ANTHROPIC_API_KEY:?}"; : "${MOONSHOT_API_KEY:?}"    # fail fast if unset
python -m pytest tests/test_sweep_run.py tests/test_sweep_state.py \
                 tests/test_summarize_run.py tests/test_distributed_topology.py -q
git diff --quiet && git diff --cached --quiet || echo "COMMIT FIRST: a paid run needs a clean sha"
```

- All four test files green.
- Working tree clean (the launcher's `require_clean_tree` gate refuses a dirty run;
  the on-VM code is verified to match the committed sha).
- Coordinator image disk has headroom for 10 batches of artifacts (500 GB is ample;
  each batch is egressed + can be pruned).
- The results repo (`Multinet-v2-results/`) is a **separate** initialized git repo
  with an `origin` remote you can push to (it is gitignored from the code repo).

## 2. Provision (brings the fleet UP; starts batch 0 only)

```bash
export SWEEP_ID=cond-sweep-YYYYMMDD
export ZONE=asia-northeast1-a
export ZONES="asia-northeast1-a asia-northeast1-c us-central1-a us-central1-b us-central1-c us-central1-f us-east1-b us-east4-c europe-west4-a europe-west4-b asia-southeast1-b asia-southeast1-c me-west1-b me-west1-c"
export DEST="artifacts-pulled/$SWEEP_ID"
export MAX_RUN_DURATION=120h        # GCP ceiling (required)
export BATCH_CAP=6h                 # per-batch watchdog
export QWEN_WORKER_COUNT=3
export DIFFICULTY_MAX=1000

./sweep_run.sh provision
```

`provision` hunts the `$ZONES` for A100-80GB capacity (NE-Asia first), code-syncs
the committed sha, applies the cost-safety net, starts the fleet running the smoke,
and writes `.runs/$SWEEP_ID/{manifest.json,sweep_state.json}`. **It does not start
any conditional batch** — stop here and wrap the loop.

Check state any time: `./sweep_run.sh status`.

## 3. The `/loop` kickoff prompt (paste into `/loop`, self-paced)

> You are supervising the conditional sweep `$SWEEP_ID` on the already-provisioned
> reused fleet (`.runs/$SWEEP_ID/manifest.json`). Work one batch at a time, disk-first,
> egress-before-advance. NEVER delete a VM. NEVER push to the code repo's `main`.
>
> Per-batch cycle (batch 0 is already running from provision; batches 1–9 you start):
> 1. **Poll** the current batch until terminal. Read live counts with:
>    `Z=$(python3 -c 'import json;d=json.load(open(".runs/'"$SWEEP_ID"'/manifest.json"));print(d["zone"])'); C=$(python3 -c 'import json;d=json.load(open(".runs/'"$SWEEP_ID"'/manifest.json"));print(d["coordinator"]["name"])'); gcloud compute ssh "$C" --zone "$Z" --command 'curl -fsS http://127.0.0.1:8765/status'`
>    Terminal = `verified==unit_count` (complete) OR `running==0 && pending==0`
>    (partial). Watch `progress_total` to tell a slow batch from a stalled one; if it
>    is frozen past ~`BATCH_CAP` the on-VM watchdog will STOP the fleet (fail-closed).
> 2. **Egress + record:** only once the batch is genuinely terminal (all
>    `verified==unit_count`, or an accepted partial per §6 — never a still-running
>    batch), run `./sweep_run.sh finalize-batch N`. A non-empty egress means "data
>    landed", NOT "batch complete" — you own the completeness check (step 1). If
>    finalize exits **40** (nothing landed) the fleet is left UP on purpose —
>    investigate; do NOT advance.
> 3. **Summaries:** dispatch three summary subagents (template in §4), one per model
>    (Qwen / Kimi / Claude), against `$DEST/<run_id>/episode_runs.jsonl`.
> 4. **Publish:** `./sweep_run.sh publish <run_id>` (mirrors to the results repo
>    minus PNGs, commits, pushes — results repo only).
> 5. **Cycle the API VMs (optional cost save):** once a batch's Kimi + Claude units
>    are all verified but Qwen is still finishing, `./sweep_run.sh stop-apis`
>    (STOPs only the e2 API runners). `next-batch` STARTs them back.
> 6. **Advance:** `./sweep_run.sh next-batch $((N+1))` (restarts coordinator + workers
>    for the next batch, re-arms the watchdog @ `BATCH_CAP`).
> 7. Update your running notes; `./sweep_run.sh status` shows the progress table + ETAs.
>
> Failure policy: see §6. **Confirm-gate:** before starting batch 10
> (`cond_baseline_thinking`), STOP and get explicit human go (§7).

## 4. Summary-subagent prompt template (one per model)

> Run: `python -m scripts.summarize_run $DEST/<run_id>/episode_runs.jsonl --model <M>
> --batch <name> --out artifacts/summaries/<run_id>__<M>.md` (the `<run_id>__` key MUST
> match `publish`'s copy glob `artifacts/summaries/<run_id>__*.md`; `<name>` is only the
> markdown heading). M ∈ {`Qwen`, `Kimi` (or
> `kimi`), `claude`} — match the `agent_or_model` substring). Then read the produced
> `.md` and add 2–3 sentences of qualitative read: loops/truncations? which mazes
> failed? token/cost surprises? Stage the 3rd backup: `mkdir -p
> Multinet-v2-results/$SWEEP_ID/<run_id>/summaries && cp your `.md` there. Do **NOT**
> run git and do **NOT** modify any source artifact — the loop publishes.

## 5. API cycling

The e2 API runners (Kimi + Claude) finish well before Qwen on most batches. Once
their units all verify they can `stop-apis` (STOP only, disks preserved) to save cost
while Qwen's tail runs; `next-batch` STARTs them again at the next batch. The
coordinator (n2) and the three Qwen (A100) VMs are **never** in the stop-apis set.

## 6. Failure runbook

Classify from the terminal counts (`verified V / total T / failed F`):

- **Light-partial** (`V ≥ ~0.8·T`, a few `failed`/truncated): accept the batch —
  `finalize-batch` (egress the partial), publish, note the failures, advance. Model
  truncation (esp. Kimi thinking-ON on batch 10) is expected, not a fleet fault.
- **Heavy-fail** (`V < ~0.5·T`, or `finalize-batch` exits 40, or a stall STOPped the
  fleet): do **not** advance. Retry the batch **once** — if the fleet is up, re-run
  `next-batch N`; if a watchdog/hardcap STOPped it, `./sweep_run.sh provision` reuses
  the STOPped disks (data intact) then resume at N. On the **second** failure:
  `./sweep_run.sh teardown` (STOP all, preserve disks) and escalate to the human with
  the coordinator + worker logs.
- **Egress fail-closed (exit 40):** the batch's artifacts did not land at
  `$DEST/<run_id>`. The fleet is intentionally left UP (data on the coordinator disk).
  Fix the egress (disk/quota/ssh) and re-run `finalize-batch N` before advancing.

`teardown` never deletes; `provision` after a teardown reattaches the same disks, so
no data is lost across a stop/restart.

## 7. Confirm-gate before batch 10

Batch 10 (`cond_baseline_thinking`, weight 3.0, Kimi thinking-ON) is the most
expensive batch. Before `next-batch 10`, STOP the loop and get an explicit human
go-ahead, presenting: batches 0–9 all egressed + published, cumulative token/cost so
far, and the calibrated ETA for batch 10 from `./sweep_run.sh status`.

---

## Known live-path risk (read before the first advance)

The **reused-fleet per-batch flow (`next-batch` / `finalize-batch`) has never run
live** — only the single-run provision→supervise→pull path was validated in the
smoke. Two things to watch on the **first** `next-batch` (0→1 transition):

1. `next-batch` `pkill`s the prior coordinator-serve and worker processes to free
   port 8765, then restarts everything. On the Qwen workers this drops the in-process
   vLLM engine → a **model reload (~14 min per GPU VM)** at every batch boundary. This
   is safe but adds ~15 min of idle GPU time per transition (×9). If a future
   run_pipeline worker tolerates a coordinator restart without exiting, the Qwen
   workers could be left warm — an optimization, not needed for correctness.
2. Confirm the new coordinator binds `:8765` cleanly (the port-free check aborts
   `next-batch` if the old serve did not die) and that the workers re-attach and pull
   the new batch's units. Watch the 0→1 transition end-to-end before trusting the loop.
