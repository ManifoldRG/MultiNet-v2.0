# Qwen served-vLLM concurrency — change log & pain points

How the Qwen distributed eval went from "can't complete a single multi-batch run" to
running at ~13× via a served-vLLM continuous-batching architecture. Written 2026-07-06.

## Outcome

Qwen (`Qwen/Qwen3.6-27B`, 3× A100-80GB) now runs the conditional sweep with each GPU
worker driving a **persistent `vllm serve`** (OpenAI server, continuous batching) and
running **16 episodes in parallel**, so the server batches their (expensive multimodal)
prefills. Validated live: 42 units in flight, servers batching 16/10/15 concurrent
requests, **1.33 gen-steps/s fleet-wide (13× the serial 0.10)**.

**Throughput caveat (important):** aggregate throughput is 13×, but a *batch's* wall
time is bounded by its **longest single episode**, because an episode's steps are
sequential and each step is ~31 s (batched per-request latency for a multimodal prompt).
Hard mazes run to 3× optimal — `s5_corridor` ≈ 267 steps ≈ 2.3 h as one episode — so each
batch is ~2.5 h and the 10-batch sweep is ~20–25 h (~$375), NOT the ~5–6 h I first
(wrongly) estimated. Concurrency cannot shorten a single long episode. (Next-iteration
fix: per-tile step budget — see the `per-tile-step-cap-next-iteration` memory.)

## The chain of fixes (in order, each caught the next)

| # | commit | bug | fix |
|---|---|---|---|
| 1 | `795d0b1` | next-batch teardown left vLLM's separate `EngineCore` process orphaned holding ~70 GiB → new `LLM()` OOMed ("Engine core initialization failed") | `lib/gpu_teardown.sh`: SIGTERM the worker + every GPU compute-app, poll `nvidia-smi` until free, fail-closed; never SIGKILL (that wedges CUDA) |
| 2 | `fa788fd` | teardown's gpu-branch called `worker_field` before `TOPO_JSON` was exported → silent fall-through to the old broken pkill | move topo derivation above the teardown loop |
| 3 | `f133658` | `prepare_job` kept a same-job's stale `job_state`, so a prior run's units failed-at-attempt-cap were never retried → coordinator dispatched nothing, workers idled | `_reset_state_for_rerun`: keep only verified/uploaded units, reset the rest to pending on re-prepare |
| 4 | `c6a8598` | configs ran thinking-ON with `max_tokens=8192` → ~2125 output tok/step (~5 min/step) AND a confound vs the thinking-OFF API baseline | thinking OFF + `max_tokens=4096` for all Qwen batches except `baseline_thinking` |
| 5 | `a632ef8` | in-process offline `LLM.chat()` is prefill-bound at batch=1 (~30 s/step) and NOT thread-safe → concurrency impossible | served architecture: persistent `vllm serve` per worker + `qwen_vllm_api` agent + `run_worker_loop(concurrency=N)` thread pool; teardown keeps the server (no reload/orphan) |
| 6 | `c41de78` | served vLLM showed "Running: 16 reqs" but only ONE distinct maze — coordinator `assign` re-returned the worker's single active unit, so 16 threads ran the same episode | `assign` is concurrency-aware (worker sends `worker_concurrency`; may hold that many DISTINCT active units); also parallel `start_worker` (3 servers load at once) + `pgrep vllm serve` guard |

Config: providers `qwen_vllm` → `qwen_vllm_api` (base_url `127.0.0.1:8000/v1`), `max_in_flight`
32 → 64 (coordinator cap must exceed 3 workers × 16). Knob `WORKER_CONCURRENCY` (default 16).

## Pain points / gotchas (what actually cost time)

1. **vLLM v1 runs `EngineCore` as a separate process.** Killing the worker by cmdline
   (`pkill -f 'distributed-role worker'`) never touched it. `vllm==0.19/0.24` v1 engine.
2. **The 02:22 "Engine core init failed" WAS the OOM** (fix #1), not a config bug — vLLM
   wraps an init-time OOM in that generic message; "root cause above" was in lost stderr.
3. **Deterministic `job_id`** (hash of run_config+manifest): a reused fleet re-prepares the
   *same* job_id, which is why stale-state preservation (#3) silently stranded every unit.
4. **Empty `worker.log` was buffering, not silence** — block-buffered stdout on an idle
   process. The real failure reason was on the *coordinator* (`job_state.json.failure_reason`),
   readable by starting only the cheap n2 coord VM.
5. **`max_in_flight` is a COORDINATOR throttle, not per-worker concurrency.** Bumping 3→32
   did nothing for throughput; the worker loop was serial. This misled the first fix attempt.
6. **Throughput measured from a 93 s `progress_total` delta was garbage** (long thinking-on
   generations finish in bursts → undersampled → false 15 tok/s). Measure over minutes and
   cross-check against verified-unit rate + the server's own throughput log lines.
7. **vLLM v1 (0.19/0.24) does NOT emit the old "Avg prompt/generation throughput" log lines
   for the offline engine** — but the *served* engine's `APIServer` logger DOES (that's how
   the 2280 tok/s prompt / 200 tok/s gen batching was confirmed).
8. **Sequential server loads:** `start_worker` blocks until its server is up, so a serial
   loop loaded 3 servers back-to-back (~42 min). Fixed to parallel `start_worker` (#6).
9. **Long-blocking `next-batch` background tasks get killed.** The first served-vLLM run
   (14-min server wait) was killed mid-`start_worker`; nohup'd VM processes survived, leaving
   a confusing half-started fleet. Run `next-batch` only once servers are up so it returns fast.
10. **Live validation caught two bugs that unit tests + code review missed** (#5 offline
    thread-safety, #6 coordinator 1-unit-per-worker). Cheap live smokes earn their keep.

## Ops runbook

```
# relight the stopped fleet, sync the committed sha, run a batch
gcloud compute instances start <coord> <3 workers> --zone us-central1-a
# sync HEAD to all VMs (git archive | ssh tar -x + verify .deployed_sha)
# first next-batch loads servers (~14 min, now parallel); run it AFTER servers are up so
# the call returns fast, or expect the long-blocking process to be killable
SWEEP_TOPO=qwen QWEN_WORKER_COUNT=3 WORKER_CONCURRENCY=16 ZONE=us-central1-a \
  SWEEP_ID=cond-sweep-20260704-qwen ... bash sweep_run.sh next-batch <N>
# per batch: finalize-batch N -> summarize_run --model Qwen -> publish <run_id> -> next-batch N+1
# servers persist across batches (no reload). STOP (not delete) to preserve disks.
```
