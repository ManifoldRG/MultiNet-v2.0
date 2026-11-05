# Run dashboard (Cloud Monitoring)

"MultiNet runs" is a Cloud Monitoring dashboard for fleet runs. Pick a run and
it shows where the run stands: how many episodes are done, running, waiting or
stuck, how that moved over the run, the time left, the API spend so far and
projected, the solve rate per model, why episodes ended, and a live list of
results and problems.

## How it works

```
                        ┌─► events ──► Cloud Logging ──► log-based metric ─┐
coordinator-serve ──────┤   (scripts/run_events.py)   (multinet_episodes_ended) ├─► dashboard
  --cloud-logging       └─► progress, every 30s ──► Cloud Monitoring ──────┘
                            (scripts/run_progress.py)  (custom.googleapis.com/multinet/*)
```

- **The coordinator writes, nothing else.** With `--cloud-logging` it writes
  two things to its own project:
  - **Events**, one log entry each in `multinet-run-events`, at the points
    where it already changes a unit's state: it hands an episode to a worker
    (`episode_started`), that attempt ends (`episode_ended`, with its tokens
    and API cost), a provider is parked on a billing limit or unparked
    (`provider_parked`, `provider_unparked`), and it starts serving
    (`coordinator_started`).
  - **Progress**, every 30 seconds: episodes by state, API spend and time
    left, as three custom metrics. It reads the coordinator's own plan and
    state files (both written atomically), so it never takes the
    coordinator's lock.

  Every worker VM the launchers start (GPU, serial API and lockstep API
  workers) is covered, and workers are unchanged. Heartbeats are untouched.
  Not covered: units run by the separate `coordinator-run-api-client` role
  (the `deploy/print_launch_commands.py` path), which does not take the flag.
- **One way.** Nothing reads the dashboard, the log or the metrics back into a
  run. Looking at them, or deleting them, has no effect on an experiment.
- **Best effort.** Writing an event only queues it; one background thread
  sends events and progress. A Google Cloud problem (unreachable, slow,
  refusing) is logged in `coordinator-serve.log` and never blocks or fails the
  coordinator. A permission error turns that half off for that coordinator
  with one error line.
- **No new dependency.** Auth is the coordinator VM's service account through
  the GCE metadata server, over the standard library.

## Turn it on for a run

```bash
CLOUD_LOGGING=1 ./launch_distributed.sh            # with the usual RUN_CONFIG, MANIFEST, ...
CLOUD_LOGGING=1 ./sweep_run.sh next-batch 2         # sweeps: either coordinator start
```

It is off unless `CLOUD_LOGGING=1`. `start_coordinator` passes
`--cloud-logging` to `coordinator-serve`, like `STALE_AFTER_SECONDS`. To turn
it on for a coordinator that is already running, restart `coordinator-serve`
with the flag (no re-prepare); its first progress update shows the whole run,
and episodes already running show only their end in the event list.

**Check it is working.** As soon as `coordinator-serve` is up (after
`coordinator-prepare`, which can take a while for mechanics mazes),
`artifacts/<RUN_ID>/coordinator-serve.log` on the coordinator says
`Run dashboard: writing ...`, the run appears in the dashboard's `run_id`
picker within a minute, and Logs Explorer with the query
`log_id("multinet-run-events")` shows `[<RUN_ID>] coordinator started: N
episodes planned`. If not, look in `coordinator-serve.log` for lines starting
`run events:` or `run progress:`.

## One-time setup in a project

Pass the project explicitly; nothing here falls back to the gcloud default.

```bash
PROJECT=your-project-id          # deliberately not defaulted
deploy/monitoring/apply.sh "$PROJECT"
```

This creates (or, on a re-run, updates in place):

| Resource | Defined in |
|---|---|
| Log-based counter `multinet_episodes_ended` (labels `run_id`, `model_id`, `outcome`, `end_kind`) | `deploy/monitoring/multinet_episodes_ended.json` |
| Dashboard "MultiNet runs" | `deploy/monitoring/dashboard.json` |

The progress metrics need no setup: a coordinator's first write creates them
(Cloud Monitoring auto-creates custom metric descriptors). Log-based metrics
count only entries written after they exist, so run `apply.sh` before the run
whose *Why episodes ended* chart you want; everything else works either way.

An earlier version of this dashboard also created `multinet_episodes_started`.
Nothing reads it any more; delete it once with
`gcloud logging metrics delete multinet_episodes_started --project "$PROJECT"`.

**Permissions.**

- *Whoever runs `apply.sh`:* `roles/logging.configWriter` (log-based metrics)
  and `roles/monitoring.dashboardEditor`, or project Editor.
- *The coordinator VM:* its service account needs `logging.logEntries.create`
  (`roles/logging.logWriter`) and `monitoring.timeSeries.create`
  (`roles/monitoring.metricWriter`); the default compute service account
  usually has both through Editor. The VM needs the `cloud-platform` access
  scope (or `logging.write` and `monitoring.write`). Coordinators are created
  from the machine image `COORD_IMAGE` (`launch_distributed.sh`), which fixes
  the account and scope. Check once:

  ```bash
  gcloud compute machine-images describe multinet-coordinator-n2-20260629 \
    --project "$PROJECT" --format='yaml(sourceInstanceProperties.serviceAccounts)'
  ```

  Without them, the first write fails with HTTP 403, the coordinator logs
  `run events: OFF ...` or `run progress: OFF ...` once, and the run carries
  on without that half.
- *Team members who view it:* `roles/monitoring.viewer` (charts) and
  `roles/logging.viewer` (event list). Project Viewer includes both. The
  dashboard is inside the Cloud console; it is not a public page.

Open it at `https://console.cloud.google.com/monitoring/dashboards?project=PROJECT_ID`.

## What the dashboard shows

Pick a run in the `run_id` picker at the top; `model_id` narrows everything
but the event list to one model. Without a pick, the tiles add up every run
writing in the selected time range.

| Widget | From | Shows |
|---|---|---|
| **Done, Planned, Stuck** | progress | Episodes finished (solved + unsolved), planned, and parked + gave up (red above 0). |
| **Time left** | progress | Episodes not finished, at the recent pace (below). |
| **API spend so far, Projected** | progress | USD at list price (below). |
| **Episodes by state over the run** | progress | The plan split by state over time. It starts all `waiting` and fills with `solved` and `unsolved`; the right edge is now. |
| **Right now** | progress | The same split at the latest update. |
| **By model** | progress | Episodes by model and state. |
| **Solve rate by model** | progress | Solved over finished, per model, over time. |
| **API spend by model** | progress | Spent and projected USD per model. |
| **Events for this run** | events | Results, problems and parks, newest first, one line each, for example `[r1] unsolved gpt-6-astra on M7_frozen_3: 120/120 steps, 131 queries, 412.6k tokens, $9.87 (max_steps)`. Starts are left out. |
| **Why episodes ended** | events | Ended attempts per 5 minutes by `end_kind`. |

**States.** Every planned episode is in exactly one, so they add up to the
plan:

| State | Coordinator status |
|---|---|
| `solved` / `unsolved` | `verified`, by whether the agent reached the goal. |
| `running` | `assigned`, `running`, `uploaded`: with a worker. |
| `waiting` | `pending`; `failed` or `stale` with attempts left (retried); `resumable` (resumes after an infra kill). |
| `parked` | Would be waiting, but its provider is parked on a billing limit. Unpark it to continue. |
| `gave_up` | Out of attempts, or a failure that is not retried (`terminal`). Finalize reports it. |

**Why episodes ended** (`end_kind`): the episode's `end_reason` for finished
episodes (`success`, `max_steps`, `stalled`, `parse_failed`, ...);
`agent_error` for an API, billing or process kill (outcome `infra_killed`: the
episode resumes); `failed` (worker error or bad upload) and `stale` (the
worker went silent) when nothing was uploaded. Retries and resumes count
again, so this counts attempts, not episodes.

## Estimates, and how they can be wrong

- **Spent** is the list price of the tokens in uploaded episodes: each query's
  recorded usage (input, cached input, cache writes, output) priced by
  `scripts/run_progress.py`. A resumed episode carries its earlier queries, so
  it is counted once. It is a floor: episodes still running and attempts that
  died without uploading are not in it yet, and VM time never is.
- **Projected** is spent plus the episodes not finished, at the model's
  average cost per finished episode so far. Easy mazes tend to finish first,
  so early in a run it runs low. It appears once a model has finished one
  episode.
- **Time left** is the episodes not finished over the pace of the last 30
  minutes (the whole run's pace if it is younger, or if nothing finished in
  that window). Parked episodes count as left.
- **Solve rate** is of the episodes finished so far, not of the plan.

**Prices.** OpenAI models use `OPENAI_PRICES` (`interface/agents/openai_agent.py`),
the table their spend cap uses, at the Flex rate when the run config says
`service_tier: flex` or the episode ran on a Batch API (lockstep workers record
`pricing_tier: batch` in `run_inputs.json`). Claude and Kimi models use
`API_PRICES` in `scripts/run_progress.py`, with their sources and dates, and
Batch at 50% (Claude) or 60% (Kimi) of the list price. A model in neither table
(self-hosted models, and any model not added yet) shows tokens but no spend.
To add one, add a line to `API_PRICES` with its source.

## What it does not know

The dashboard reports what the coordinator saw. It is not the run's results:
the coordinator's `/status` and `coordinator-finalize` are the source of truth.

- **No progress inside an episode.** An episode in progress counts only as
  `running`.
- **Updates lag a little.** Progress is at most 30 seconds old when written and
  reaches the charts within a minute or two; the event list is close to live.
  If a coordinator stops writing (stopped, or Cloud Monitoring unreachable),
  the last update stays on the tiles and the state chart stops at that time.
- **Events are best effort.** If a write fails after retries, those events are
  dropped and the next write that lands starts with an `events_dropped` entry
  giving the count. Events queued in the last seconds before a coordinator is
  killed or its VM stops can be lost without that marker. A worker that
  uploads after its lease went stale produces a second end for that attempt.
- **`wall_seconds`** in an event is coordinator-observed: from the hand-out to
  the result arriving (including upload), not model time.
- **Only flagged runs.** Runs without `CLOUD_LOGGING=1` are absent.

## Data reference

**Events.** Log `projects/PROJECT_ID/logs/multinet-run-events`, resource
`global`. Every entry's `jsonPayload` has `event`, `run_id`, `job_id` and a
one-line `message`.

| Field | `episode_started` | `episode_ended` |
|---|---|---|
| `unit_id`, `task_id`, `model_id`, `model_group`, `provider`, `seed`, `prompt_variant`, `backend` | yes | yes |
| `worker_id`, `worker_host` | worker it went to | worker that held it |
| `attempt` (fresh hand-outs so far), `resume_count` | yes | yes |
| `resumed` | yes | |
| `status`, `outcome`, `end_kind` | | yes |
| `success`, `end_reason`, `steps_used`, `max_steps`, `query_count` | | from the uploaded `episode.json` (null when there is none) |
| `tokens`, `cost_usd` | | the episode's so far (null without an upload; `cost_usd` null without a price) |
| `wall_seconds`, `reason` (why an attempt ended without an episode) | | yes |

`provider_parked` carries `provider`, `groups`, `unit_id` and `reason`;
`provider_unparked` carries `provider` and `groups`. `coordinator_started`
carries `unit_count`, `units` (unit counts by status at start, so a restarted
coordinator shows what is already verified) and `models`. `events_dropped`
carries `count`. Severity is `WARNING` for `infra_killed`, `failed`, `stale`,
`provider_parked` and `events_dropped`, `INFO` otherwise.

**Progress.** Custom metrics, kind GAUGE, resource `global`, one point per
series every 30 seconds:

| Metric (`custom.googleapis.com/multinet/...`) | Labels | Value |
|---|---|---|
| `episodes` | `run_id`, `model_id`, `state` | INT64 episodes in that state |
| `api_cost_usd` | `run_id`, `model_id`, `kind` (`spent`, `projected`) | DOUBLE USD (priced models only) |
| `time_left_minutes` | `run_id` | INT64 minutes (once an episode has finished) |

`tests/test_run_events.py` checks that every label and value the dashboard
filters or groups on is one the events and progress carry.

## What it costs

Nothing runs between runs: no VM, container, scheduler or uptime check. The
dashboard and metrics cost nothing at rest.

| Line | Usage | Cost |
|---|---|---|
| Cloud Logging ingestion | ~650 bytes per entry; an episode attempt writes two: ~1.3 MB per 1,000 episodes | $0: the first 50 GiB per project per month are free, then $0.50/GiB |
| Log retention | `_Default` bucket, 30 days | $0 |
| Progress metrics | (8 series per model + 1) x 2 points a minute while the coordinator serves, 8 bytes a point: ~1 MiB per 5-model run-day | $0 within the free 150 MiB of metric data per billing account per month, then about $0.26/MiB |
| Log-based metric | a handful of series per run (run x model x outcome x end kind) | inside the same free 150 MiB |
| Dashboard and viewing | | $0 |

Labels are kept to run, model and a few fixed values (not maze) so the number
of series stays small. Google changes these rates; re-check before relying on
them.

## Teardown

```bash
gcloud monitoring dashboards list --project "$PROJECT" \
  --filter='displayName="MultiNet runs"' --format='value(name)' \
  | xargs -r -n1 gcloud monitoring dashboards delete --project "$PROJECT" --quiet
gcloud logging metrics delete multinet_episodes_ended --project "$PROJECT" --quiet
for m in episodes api_cost_usd time_left_minutes; do   # gcloud has no command for custom metrics
  curl -sS -X DELETE -H "Authorization: Bearer $(gcloud auth print-access-token)" \
    "https://monitoring.googleapis.com/v3/projects/$PROJECT/metricDescriptors/custom.googleapis.com/multinet/$m"
done
```

Written log entries age out of the `_Default` bucket after 30 days; metric
points after Cloud Monitoring's retention (24 months for custom metrics).
