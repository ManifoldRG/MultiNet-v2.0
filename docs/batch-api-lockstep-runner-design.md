# Batch-API step-lockstep runner — design spec

**Status:** design, approved 2026-07-16; **amended 2026-07-16 after full codebase +
provider fact-check** (every claim below verified against `feature/early_terminate`
working tree and provider docs — corrections are integrated in place, with a
"Verified reality" note where the original draft was wrong).
**Motivation:** R1 runs Claude Opus 4.8 (xhigh, 64k cap) and Kimi k2.6 over 50
mazes. Output cost dominates. Anthropic's Batch API (50% off: Opus 4.8 at
$2.50/$12.50 per MTok vs $5/$25) and Moonshot's Batch API (40% off: kimi-k2.6 at
$0.57 in / $2.40 out per MTok vs $0.95/$4.00) cut that roughly in half. The 50
mazes are mutually independent, so at any tick every maze that is *waiting on a
model query* contributes one independent request that can be batched together.

## Scope

- **In:** a new lockstep runner that advances N mazes together and batches each
  tick's queries; a batched/unbatched method pair on the agent interface; Claude
  and Kimi batch-API implementations; coordinator integration; a validation
  smoke.
- **Out:** Qwen (local vLLM, batches server-side already — uses its existing
  path); the as-ready/bubble-popping optimization (future work, below).

## Agent interface

**Verified reality (2026-07-16):** the current contract is
`__call__(messages: list[dict]) -> str` (aliased `Agent = Callable` at
`scripts/run_pipeline.py:42`) plus side-channel attributes mutated per call:
`last_usage` (all five agents), `last_thinking` (**Claude only** —
`interface/agents/claude.py:275-291`; Kimi discards `reasoning_content`, Qwen
agents discard thinking), and a `reset_usage()` method (Qwen agents only). The
runner does reset-then-read around each call (`interface/runner.py:316,384-389`)
— single-slot state that is unsafe with N calls in flight. No agent uses an SDK;
all are raw `urllib` POSTs with `interface/agents/http_retry.py` retry. There is
no module-level shared client/rate-limiter state, so concurrency safety hinges
entirely on these per-instance attributes. Formalize:

```
class Reply:            # small dataclass
    text: str
    usage: dict         # input_tokens, output_tokens, total_tokens (+ cache splits)
    thinking: str | None = None
    stop_reason: str | None = None   # provider stop/finish reason, verbatim
    token_truncated: bool = False    # output hit the request max_tokens cap

class Agent(Protocol):
    def generate(self, messages: list[dict]) -> Reply: ...
    def generate_batch(self, batch: list[list[dict]]) -> list[Reply]: ...
```

- `generate` = today's behavior; keep `__call__` as an alias (returning
  `reply.text` and still mutating `last_usage`/`last_thinking`) so existing
  callers (`runner.py`, `_CountingAgent`, every test double) are untouched.
- **Truncation detection (corrected):** both providers' batch results DO include
  a stop signal — Anthropic returns the full Message with `stop_reason`
  (`"max_tokens"` = truncated) and Moonshot returns chat-completions bodies with
  `finish_reason` (`"length"` = truncated). Use that as the primary signal;
  `output_tokens >= request max_tokens` is the fallback only. The original
  draft's "providers' usage lacks finish_reason" was wrong.
- **Naming (corrected):** `truncated` already means *environment* truncation in
  transcripts/`end_reason` (`interface/runner.py:558-560`,
  `pipeline/episode_metrics.py:198-209`). The token-cap flag is therefore named
  `token_truncated` everywhere it is persisted, and the raw `stop_reason` is
  recorded on the query record.
- `generate_batch` returns results in input order, one `Reply` per input.
  Providers return batch results in arbitrary order — map back by `custom_id`.
- Per-provider implementation:
  - **Claude** → Anthropic Message Batches API (`POST /v1/messages/batches`,
    up to 100k requests / 256 MB per batch; batch-endpoint pool ~1000 RPM —
    per-tick batch creation + 30-60s polling is far inside limits). One request
    per maze with a `custom_id`; poll `GET .../batches/{id}` until
    `processing_status == "ended"`; stream results; map by `custom_id`.
    Request shape for Opus 4.8 (hard requirements, 400s otherwise):
    `thinking: {"type": "adaptive"}` + `output_config: {"effort": "xhigh"}`;
    **never** `budget_tokens`, `temperature`, `top_p`, or `top_k`. Set
    `thinking: {"type": "adaptive", "display": "summarized"}` so thinking text
    is loggable (4.8 defaults to `omitted`; billing identical — full thinking
    tokens bill as output either way). Vision/base64 images are supported in
    batch. Growing multi-image histories inflate batch bytes — watch the 256 MB
    bound if MAX_BATCHES grows.
  - **Kimi** → Moonshot Batch API (40% off; platform.kimi.ai — the old
    platform.moonshot.ai domain now redirects). File-based flow: upload a JSONL
    file (`purpose="batch"`, ≤100 MB, one
    `{custom_id, method, url:"/v1/chat/completions", body}` per line) →
    `POST /v1/batches` with a `completion_window` → poll
    `validating → in_progress → finalizing → completed` → download
    `output_file_id` (+ `error_file_id`). All requests in one batch must use the
    same model. Body: `thinking: {"type": "enabled"}`; omit sampling params
    entirely (k2.6 mode-forces temp 1.0 thinking / 0.6 non-thinking and errors
    on other values); **set `max_tokens` explicitly** (k2.6 defaults to 32768).
    Capture `reasoning_content` into `Reply.thinking` (the sync agent currently
    discards it — fix in the batch implementation and, optionally, the sync path).
  - **Qwen** → fan out to the vLLM server it already continuous-batches against
    (thread pool over the served endpoint); not used by R1's lockstep path but
    implemented for interface completeness. Capture vLLM's
    `choices[0].finish_reason` (`"length"`) into `Reply.stop_reason` — the
    two-tier rerun design consumes it.

## Lockstep runner

New module `interface/batch_runner.py`. Does **not** modify `runner.py`; the
per-maze env-step + prompt-assembly logic is extracted from `runner.py` into a
shared helper (e.g. `interface/episode_step.py`) that both runners call, so they
cannot drift. **The extraction seam is the loop body `interface/runner.py:270-560`**,
split as `plan_query(...) -> messages | None` / `apply_reply(reply) -> events,
termination` around the agent call at `:318`, keeping the working-tree history
normalization (`:321-344`) and `logged_agent_messages` handling (`:363-369`)
inside the shared helper.

**Batch granularity (corrected — load-bearing):** the batch unit is the
**query**, not the env step. A maze contributes a request to a round only when
`primitive_buffer` is empty AND `querying.should_query(...)` is true
(`runner.py:273`). Between queries a maze advances **locally** with no model
call: cardinal tokens expand to several primitives (`runner.py:423-427`),
subgoal replies carry multiple actions, and `full_trajectory` queries exactly
once per episode. Parse failures are inline re-queries at the same
`env_step_count` (`runner.py:394-412`) — a maze may occupy several rounds
without stepping. "Lockstep" means one batch round per tick over the
mazes-awaiting-a-query, not same-step-index and not one-request-per-active-maze.
Each `QueryingMode` instance is stateful and per-episode — one per maze, never
shared.

State: a working set of up to `MAX_BATCHES` per-maze episode state machines.
Each **round** (one tick):

1. Drain every active maze locally until it either terminates or needs a query.
2. Collect the next user-message for every maze awaiting a query. Mazes may be
   at different step indices — a freshly pulled maze at step 0 batches alongside
   one at step 30.
3. `agent.generate_batch(...)` over that set. **Strict lockstep:** wait for the
   entire batch, then apply each reply (parse → queue actions → local drain).
4. **Report progress to the coordinator after every round** (per-maze heartbeat
   — see Coordinator integration) — richer than today's on-completion-only
   reporting. **Caveat (corrected):** heartbeats fire only *between* rounds, so a
   round that runs longer than `stale_after_seconds` emits none while it is in
   flight. Per-round heartbeats keep progress *visible*; they do NOT by
   themselves keep long rounds from going stale. A single batch round can run the
   full `batch_deadline_s` (2 h), so the coordinator must be started with
   `--stale-after-seconds` ≥ `batch_deadline_s + batch_cancel_grace_s` (+ slack;
   e.g. 9000), and there must be **exactly one lockstep worker per API model
   group** — otherwise a stale unit is re-assigned to the other worker and
   double-paid.
5. Terminated mazes (solved / failed / stall-K) drop out; **refill from the
   coordinator up to `MAX_BATCHES`**. The batch stays full until the unit pool
   drains, then the tail shrinks.

## Coordinator integration

- The batch runner is an **API worker** that holds up to `MAX_BATCHES` distinct
  active units. The concurrency-aware `assign` already supports this
  (`scripts/distributed_run_pipeline.py:543-607` honors `worker_concurrency`
  from capabilities; CLI `--worker-concurrency` / env `WORKER_CONCURRENCY`).
  **`MAX_BATCHES` is not a new knob — it IS `worker_concurrency`** for the
  lockstep worker role; no separate config key.
- **Invariant (new):** the model group's `max_in_flight` (a fleet-wide,
  per-group coordinator throttle, `distributed_run_pipeline.py:794-814`) must be
  `>= MAX_BATCHES`, or `assign` silently starves the working set. R1 config
  must raise Claude/Kimi `max_in_flight` from 1 to `>= 50`.
- **Per-round heartbeats (corrected mechanism):** the store already accepts
  per-unit heartbeats keyed by `unit_id`
  (`CoordinatorStore.heartbeat`, `:623-625`); what's missing is only the
  driver. The lockstep worker must heartbeat **every held unit_id** each round
  (with its generation count as `progress`) — today heartbeating lives inside
  per-unit threads (`:1219-1223`) and the concurrent loop never does it for a
  shared working set. This makes between-round progress visible, but it does NOT
  cover a single round that runs longer than `stale_after_seconds` (no heartbeat
  flows mid-round) — raise `--stale-after-seconds` above the worst-case round
  and run one lockstep worker per group (see round-loop step 4) so a mid-round
  stale bounce cannot double-assign.
- Refill: on maze completion the worker requests a replacement unit
  (`assign`), keeping the working set at `MAX_BATCHES` until the coordinator's
  pool is empty. The existing thread-per-episode refill loop
  (`run_worker_loop`, `:1313-1337`) is **not** reusable — the lockstep worker is
  a new single-loop driver over N state machines using the same
  assign/heartbeat/upload/fail primitives.
- **Upload compatibility (hard requirement):** each finished unit must contain
  `episode.json`, `run_inputs.json`, `run_score.json`
  (`EXPECTED_RUN_FILES`, `:47`), and `run_inputs.json` must carry an
  `inputs_hash` equal to the unit's `episode_inputs_hash` — upload verification
  rejects mismatches (`_verify_and_extract`). The lockstep worker must produce
  run inputs through the same machinery as `pipeline._run_one_unit`, not
  reimplement it.

## Error handling & durability

- **Strict lockstep head-of-line blocking** accepted: one runaway query (a
  Claude step thinking to 64k) gates its round. Mitigation is future work
  (as-ready).
- **Per-round deadline (new — required):** neither provider publishes a batch
  latency SLA (Anthropic: "most < 1h", hard 24h expiry; Moonshot: completion
  window, e.g. 24h). Today nothing kills a wedged round between coordinator
  staleness (defeated by our own heartbeats) and the whole-VM
  `BATCH_CAP` shutdown — the 6h-watchdog incident, amplified. The runner
  therefore enforces its own per-round deadline (config knob, default 2h): on
  expiry, cancel the provider batch, treat missing results as batch-item
  failures (below), and continue. Size `MAX_RUN_DURATION`/`BATCH_CAP`
  assuming multi-round tail latency, not sync-API latency.
- **Per-round checkpointing:** persist each maze's transcript after every batch
  round so a crashed or very long batch run resumes without re-paying completed
  ticks. **Verified reality:** this is net-new — today `episode.json` is
  written once at episode end and the writer rmtree's the run dir first
  (`interface/episode_log.py:84-140`); no resume exists anywhere. Checkpoint at
  query boundaries only (both action buffers empty) so no mid-drain
  `primitive_buffer`/`action_queue` needs persisting; resume = `backend.reset(seed)`
  + deterministic replay of recorded primitive actions (verified: no
  un-replayable RNG in the step path; stall-K state rebuilds from replayed
  states).
- **Batch-item failure** (a `custom_id` errored/expired): that maze records a
  parse-failure step and continues, identical to a normal bad reply (it counts
  against `max_parse_retries`). No batch aborts on one bad item. Anthropic
  `expired` items are not billed.
- **Cost provenance:** record `pricing_tier: "batch"` in `run_inputs.json` so
  discounted cost is reproducible from artifacts (token usage alone can't tell
  batch from sync).

## Validation smoke

- **Mazes (selected 2026-07-16 from the candidate-maze feature table, maintained
  with the results; all beatable, none in balanced_03, none length-risk, 5
  distinct mechanism signatures):**

  | slot | maze | optimal actions | signature |
  |---|---|---:|---|
  | ~30 | `S4/10x10_dense_0` | 30 | none (navigation) |
  | ~40 | `D1/10x10_dense_wrong_ky_kr_sg_1` | 42 | K→S + wrong key |
  | ~50 | `M5/10x10_corridor_kr_kb_1` | 47 | K→K |
  | ~60 | `M6/10x10_dense_kr_sg_kb_0` | 61 | K→S→K |
  | ~80 | `D1/14x14_dense_wrong_ky_kr_0` | 80 | K + wrong key |

  Sources: `ogbench/ogbench/procgen/maze_jsons/<id>.json` (all tracked in the
  submodule at the pinned sha; submodule shipping to VMs is handled + verified
  since commit 36ebf73). Smoke fixtures: `manifest.r1_smoke_batch.json` +
  a smoke run config patterned on `run_config.r1.json`, validated with
  `scripts/validate_fixtures.py --manifest ...`.
- **Models:** Claude + Kimi through `generate_batch`.
- **Checks:** (a) results map to the right mazes by `custom_id`; (b) mazes
  terminating at different steps drop out cleanly without corrupting others
  (the early-termination concern); (c) per-maze token usage + cost captured,
  `stop_reason` recorded; (d) batched vs a tiny unbatched control confirms the
  ~50% / ~40% price delta and equivalent behavior; (e) checkpoint/resume
  exercised once (kill + resume mid-run); (f) per-round batch latency measured
  (informs the per-round deadline and BATCH_CAP for the full run).
- **Output:** a short report that **updates the R1 budget estimate** with real
  image_only + xhigh numbers (queries/episode and output/query for the actual
  cell, which no prior run measured). Prior central estimate ~$435 sync
  (Claude ~$329 + Kimi ~$107); batch pricing shifts that to roughly
  ~$229 central (~$165 Claude + ~$64 Kimi) before smoke-measured corrections.

## Testing

- Unit: `generate_batch` result-ordering + `custom_id` mapping (mocked API);
  ragged termination (mazes ending at steps 2/5/8 all captured correctly);
  resume-from-checkpoint; `stop_reason`/`token_truncated` propagation;
  per-round deadline expiry handling.
- E2e: scripted-agent lockstep run over 3 mazes driving the shared episode-step
  helper (mirrors `tests/test_cardinal_runner.py`; a scripted batch agent
  implements `generate_batch` returning per-maze scripted `Reply`s). Equivalence
  test: same scripted episode through `runner.py` and through the lockstep
  runner produces identical transcripts.

## Future work (flag in code + docs, do not build now)

- **As-ready + bubble-popping.** Replace strict lockstep with per-ready
  retrieval that steps each maze the moment its result returns and refills each
  batch from a **broader pool**, so stragglers don't leave idle slots (bubbles).
  This is the extension to `MAX_BATCHES` — a bubble-popping initiative. It trades
  larger uniform batches for continuous slot utilization.
