# Qwen two-tier truncation-requeue — design spec

**Status:** design, approved 2026-07-16. Implementation to be done by the team.
**Motivation:** Qwen3.6-27B runs locally on A100 via served vLLM. A large output
budget forces a KV-cache tradeoff: `max_model_len=96000` (needed for a 64k output
cap) collapses concurrency from ~13-16 episodes/server to ~2-3, wrecking the
"~4-5 hour, 26-parallel" plan. But Qwen's *observed* output demand is tiny
(median 142, p90 ~2815, max ~4921; ~4.8% of image_only queries hit the buggy 4k
cap). So: run wide at a small cap, and only re-run the few mazes that actually
truncate, at 64k.

## Scope

- **Qwen-only.** Claude/Kimi are API with no KV constraint — they run at 64k from
  the start, no two-tier.
- A general two-tier mechanism, used by R1.

## Two phases, one fleet, one reload

- **Phase 1 (fast/wide):** `max_model_len=16384`, `max_tokens=8000`, full
  concurrency. Runs **all** mazes. (8k vs the prior proven 4k gives headroom for
  the harder balanced panel; because Qwen's median output is ~142, most sequences
  stay tiny and the concurrency hit from the higher cap is modest — only the rare
  long generations pay the extra KV.)
- **Reload (same fleet):** restart the vLLM servers with `max_model_len=96000`
  and reduced `max_in_flight` / concurrency (~2-3/server — the cost of the big KV
  cache). Serve-args cannot change at runtime; this is a real ~14-min reload
  (see `docs/qwen-served-vllm-concurrency.md`).
- **Phase 2 (deep/narrow):** `max_tokens=64000`, reruns **only the flagged
  mazes**. The flagged set is small (~5%), so phase 2 is short despite low
  concurrency.

## Detection & scan

- **Trigger:** an episode is flagged if **any step has `output_tokens >= cap`**,
  where `cap` is that phase's `max_tokens` (8000 in phase 1).
  This is deliberate and load-bearing: Qwen's truncated steps mostly still report
  `parse_ok=True` (the lenient parser salvages an action from the cut-off
  reasoning), so a parse-failure trigger would miss ~98% of Qwen truncations. The
  cap-hit trigger is the only one that catches the silent, degraded decisions.
- **When:** live as episodes complete in phase 1. Flagged `task_id`s accumulate
  into a **phase-2 rerun manifest** (a subset of the original manifest). Reruns
  are deferred to phase 2 regardless (the server reload forces a phase boundary),
  so a live scan and a post-phase-1 sweep are functionally equivalent; live is
  chosen for earlier visibility.
- **Where:** a small scan step in the pipeline (`scripts/scan_truncations.py` or
  a pipeline function) reading `episode.json` / `episode_runs.jsonl`.

## Result reconciliation & provenance

- Each `episode.json` records the `pass` (1 or 2) and the `max_model_len` /
  `max_tokens` it ran under.
- Final results = phase-1 episodes for unflagged mazes **+ phase-2 episodes
  overwrite phase-1** for flagged mazes (keyed by `task_id`).
- Artifacts land in phase-labeled dirs: `.runs/<run>/qwen_phase1/`,
  `.runs/<run>/qwen_phase2/`, so the outcome is legible from co-located files
  without reverse-engineering.

## Terminal case

64k is the ceiling — there is no third tier. If a maze **still** hits
`output_tokens >= 64000` in phase 2, keep the episode and mark it
`truncated_at_ceiling: true` rather than looping. Report these explicitly so the
results are interpretable (a genuinely unbounded Qwen thinking loop is a finding,
not a bug to retry forever).

## Documentation (explicit requirement — clear & replicable)

- Runbook `docs/qwen-two-tier-rerun.md`: what/why/how, the KV-vs-parallelism
  rationale, the exact reload step and phase-2 serve args, how to re-run.
- Inline comments at the phase-transition points in the run scripts
  (`launch_distributed.sh` / `lib/distributed_start.sh` / the sweep driver).
- Phase-labeled artifacts (above) as self-documenting output.
- Goal: a new person can see and reproduce the two-tier flow from the files.

## Testing

- Unit: truncation-flagging — `output_tokens >= cap` at various step positions
  flags the episode; all-under does not flag; `parse_ok=True` truncations are
  still flagged.
- Scan → rerun-manifest: correct subset of `task_id`s emitted.
- Merge/provenance: phase-2 overwrites phase-1 by `task_id`; `pass` and cap
  recorded on each episode.
- Terminal case: `truncated_at_ceiling` set when phase-2 still hits 64k.

## Config summary

| Phase | max_model_len | max_tokens | concurrency | mazes |
|---|---|---|---|---|
| 1 | 16384 | 8000 | full (~13-16/server) | all |
| 2 | 96000 | 64000 | reduced (~2-3/server) | flagged only |

Claude/Kimi (for reference, not two-tier): 64000 cap from the start.
