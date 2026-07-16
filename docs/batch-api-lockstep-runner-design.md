# Batch-API step-lockstep runner — design spec

**Status:** design, approved 2026-07-16. Implementation to be done by the team.
**Motivation:** R1 runs Claude Opus 4.8 (xhigh, 64k cap) and Kimi k2.6 over 50
mazes. Output cost dominates. Anthropic's Batch API (50% off) and Moonshot's
Batch API (40% off) cut that roughly in half. The 50 mazes are mutually
independent, so at any tick every active maze contributes one independent query
that can be batched together.

## Scope

- **In:** a new lockstep runner that advances N mazes together and batches each
  tick's queries; a batched/unbatched method pair on the agent interface; Claude
  and Kimi batch-API implementations; coordinator integration; a validation
  smoke.
- **Out:** Qwen (local vLLM, batches server-side already — uses its existing
  path); the as-ready/bubble-popping optimization (future work, below).

## Agent interface

Today each agent is an implicit protocol: `__call__(messages) -> str` plus a
mutated `last_usage` attribute. That side-channel breaks when N calls are in
flight. Formalize it:

```
class Reply:            # small dataclass
    text: str
    usage: dict         # input_tokens, output_tokens, total_tokens
    thinking: str | None = None
    truncated: bool = False   # output_tokens >= request max_tokens (providers'
                              # usage lacks finish_reason; compare against the cap)

class Agent(Protocol):
    def generate(self, messages: list[dict]) -> Reply: ...
    def generate_batch(self, batch: list[list[dict]]) -> list[Reply]: ...
```

- `generate` = today's behavior; keep `__call__` as an alias so existing callers
  (`runner.py`) are untouched.
- `generate_batch` returns results in input order, one `Reply` per input.
- Per-provider implementation:
  - **Claude** → Anthropic Batch API (`POST /v1/messages/batches`). One request
    per maze with a `custom_id`; poll batch until `ended`; retrieve results;
    map back by `custom_id`.
  - **Kimi** → Moonshot Batch API (40% off). Same pattern.
  - **Qwen** → fan out to the vLLM server it already continuous-batches against
    (thread pool over the served endpoint); not used by R1's lockstep path but
    implemented for interface completeness.

## Lockstep runner

New module `interface/batch_runner.py`. Does **not** modify `runner.py`; the
per-maze env-step + prompt-assembly logic is extracted from `runner.py` into a
shared helper (e.g. `interface/episode_step.py`) that both runners call, so they
cannot drift.

State: a working set of up to `MAX_BATCHES` per-maze episode state machines.
Each **round** (one tick):

1. Collect the next user-message for every **active** maze. Mazes may be at
   different step indices — a freshly pulled maze at step 0 batches alongside one
   at step 30. "Lockstep" means one batch round per tick, not same-step-index.
2. `agent.generate_batch(...)` over the whole active set. **Strict lockstep:**
   wait for the entire batch, then step each env with its parsed action.
3. **Report progress to the coordinator after every tick** (per-maze step
   update) — richer than today's on-completion-only reporting; keeps the
   progress watchdog live during long batches.
4. Terminated mazes (solved / failed / stall-K) drop out; **refill from the
   coordinator up to `MAX_BATCHES`**. The batch stays full until the unit pool
   drains, then the tail shrinks.

## Coordinator integration

- The batch runner is an **API worker** that holds up to `MAX_BATCHES` distinct
  active units. This reuses the existing concurrency-aware `assign`
  (worker sends `worker_concurrency`; may hold that many distinct active units —
  see `docs/qwen-served-vllm-concurrency.md`). `MAX_BATCHES` maps to
  `worker_concurrency`.
- New: **per-tick progress reporting.** Extend the worker→coordinator progress
  channel so a maze reports after each batch step, not only on completion.
- Refill: on maze completion the worker requests a replacement unit, keeping the
  working set at `MAX_BATCHES` until the coordinator's pool is empty.
- `MAX_BATCHES` is a new config knob. For R1 (50 mazes, 1 seed) it can be 50
  (one batch); it matters when units > `MAX_BATCHES` (multi-seed, long_tail).

## Error handling & durability

- **Strict lockstep head-of-line blocking** accepted: one runaway query (a
  Claude step thinking to 64k) gates its round. Mitigation is future work
  (as-ready).
- **Per-tick checkpointing:** persist each maze's transcript after every batch
  round so a crashed or very long batch run resumes without re-paying completed
  ticks. Reuse `episode.json` semantics.
- **Batch-item failure** (a `custom_id` errored/expired): that maze records a
  parse-failure step and continues, identical to a normal bad reply. No batch
  aborts on one bad item.

## Validation smoke

- **Mazes:** 5 mazes **not in balanced_03**, spanning the curve at roughly
  30 / 40 / 50 / 60 / ~80 BFS steps (nearest available in the corpus, not hard
  points; go from ~30 up to high-70s/low-80s).
- **Models:** Claude + Kimi through `generate_batch`.
- **Checks:** (a) results map to the right mazes by `custom_id`; (b) mazes
  terminating at different steps drop out cleanly without corrupting others
  (the early-termination concern); (c) per-maze token usage + cost captured;
  (d) batched vs a tiny unbatched control confirms the ~50% / ~40% price delta
  and equivalent behavior.
- **Output:** a short report that **updates the R1 budget estimate** with real
  image_only + xhigh numbers (queries/episode and output/query for the actual
  cell, which no prior run measured).

## Testing

- Unit: `generate_batch` result-ordering + `custom_id` mapping (mocked API);
  ragged termination (mazes ending at steps 2/5/8 all captured correctly);
  resume-from-checkpoint.
- E2e: scripted-agent lockstep run over 3 mazes driving the shared episode-step
  helper (mirrors `tests/test_cardinal_runner.py`).

## Future work (flag in code + docs, do not build now)

- **As-ready + bubble-popping.** Replace strict lockstep with per-ready
  retrieval that steps each maze the moment its result returns and refills each
  batch from a **broader pool**, so stragglers don't leave idle slots (bubbles).
  This is the extension to `MAX_BATCHES` — a bubble-popping initiative. It trades
  larger uniform batches for continuous slot utilization.
