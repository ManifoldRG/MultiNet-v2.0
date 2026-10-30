# Run dashboard (Cloud Monitoring)

A Cloud Monitoring dashboard, "MultiNet runs", that shows episodes of a fleet
run as they start and end: how many started per model, how many ended and how
(solved, unsolved, infra-killed, failed, stale), and a live list of the events
themselves.

## How it works

```
coordinator-serve --cloud-logging ──► Cloud Logging ──► log-based metrics ──► dashboard
   (scripts/run_events.py)            log: multinet-run-events   (2 counters)   (charts, table, event list)
```

- **The coordinator writes, nothing else.** With `--cloud-logging` the
  coordinator writes one structured log entry when it hands an episode to a
  worker (`episode_started`) and one when that attempt ends (`episode_ended`),
  plus `coordinator_started` with the run's planned units. These are the
  points where the coordinator already changes a unit's state (`assign`,
  `upload`, `fail`, stale lease), so every worker VM the launchers start
  (GPU, serial API and lockstep API workers) is covered, and workers are
  unchanged. Heartbeats are untouched. Not covered: units run by the separate
  `coordinator-run-api-client` role (the `deploy/print_launch_commands.py`
  path), which does not take the flag.
- **One way.** Nothing reads the dashboard or the log back into a run. Looking
  at it, or deleting it, has no effect on an experiment.
- **Best effort.** Writing only queues an event; one background thread sends
  it. A Cloud Logging problem (unreachable, slow, refusing) is logged in
  `coordinator-serve.log` and never blocks or fails the coordinator. A
  permission error turns events off for that coordinator with one error line.
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
with the flag (no re-prepare); episodes already running then show only their end.

**Check it is working.** As soon as `coordinator-serve` is up (after
`coordinator-prepare`, which can take a while for mechanics mazes), Logs
Explorer with the query `log_id("multinet-run-events")` should show
`[<RUN_ID>] coordinator started: N episodes planned`. If it does not, look in
`artifacts/<RUN_ID>/coordinator-serve.log` on the coordinator for lines starting
`run events:`.

## One-time setup in a project

Pass the project explicitly; nothing here falls back to the gcloud default.

```bash
PROJECT=your-project-id          # deliberately not defaulted
deploy/monitoring/apply.sh "$PROJECT"
```

This creates (or, on a re-run, updates in place) three things, defined in
`deploy/monitoring/`:

| Resource | File |
|---|---|
| Log-based counter `multinet_episodes_started` (labels `run_id`, `model_id`) | `multinet_episodes_started.json` |
| Log-based counter `multinet_episodes_ended` (labels `run_id`, `model_id`, `outcome`) | `multinet_episodes_ended.json` |
| Dashboard "MultiNet runs" | `dashboard.json` |

Log-based metrics count only entries written after they exist. Create them
before the run you want charted; the event list shows older entries either way.

**Permissions.**

- *Whoever runs `apply.sh`:* `roles/logging.configWriter` (log-based metrics)
  and `roles/monitoring.dashboardEditor`, or project Editor.
- *The coordinator VM:* its service account needs `logging.logEntries.create`
  (`roles/logging.logWriter`; the default compute service account usually has
  it through Editor), and the VM needs the `logging.write` or `cloud-platform`
  access scope. Coordinators are created from the machine image
  `COORD_IMAGE` (`launch_distributed.sh`), which fixes both. Check once:

  ```bash
  gcloud compute machine-images describe multinet-coordinator-n2-20260629 \
    --project "$PROJECT" --format='yaml(sourceInstanceProperties.serviceAccounts)'
  ```

  or, on a running coordinator, `gcloud compute instances describe <COORD>
  --zone <ZONE> --format='yaml(serviceAccounts)'`. Without them, the first
  write fails with HTTP 403, the coordinator logs `run events: OFF ...` once,
  and the run carries on without events.
- *Team members who view it:* `roles/monitoring.viewer` (charts) and
  `roles/logging.viewer` (event list). Project Viewer includes both. The
  dashboard is inside the Cloud console; it is not a public page.

Open it at `https://console.cloud.google.com/monitoring/dashboards?project=PROJECT_ID`.

## What the dashboard shows

- **Episodes ended in the selected time range**: a table of run, model and
  outcome with counts, over whatever time range the dashboard is set to.
- **Episodes ended, by outcome** and **Episodes started, by model**: counts per
  5 minutes.
- **Run events, newest first**: every entry, one line each, for example
  `[r1-astra] unsolved gpt6_astra on M7_frozen_3: 120/120 steps, 131 queries (max_steps)`.
- A `run_id` filter at the top narrows the charts and table to one run.

Outcomes:

| Outcome | Coordinator status | Meaning |
|---|---|---|
| `solved` | `verified` | Finished; the agent reached the goal. |
| `unsolved` | `verified` | Finished without reaching the goal (`end_reason` says how: `max_steps`, `stalled`, `parse_failed`, ...). |
| `infra_killed` | `resumable` | Ended on `agent_error:*` (API error, billing, dead process); the unit resumes. |
| `failed` | `failed` | The attempt failed (worker error, bad upload, or out of resumes). Retried unless the unit is out of attempts or the failure is final. |
| `stale` | `stale` | The worker stopped heartbeating; the unit is handed out again. |

## What it does not know

The dashboard reports events as the coordinator saw them. It is not the run's
results: the coordinator's `/status` and `coordinator-finalize` are the source
of truth.

- **No step-level progress.** An episode in progress shows only as started.
- **Charts lag.** Log-based metrics usually reach the charts within a few
  minutes; the event list is closer to live.
- **Starts are hand-outs, not units.** A retry after a failure or stale lease,
  and a resume after an infra kill, each count as a new start. So "started"
  can exceed the unit count, and starts minus ends is not exactly "running".
  A worker that uploads after its lease went stale produces a second end for
  that attempt (`stale`, then its real outcome).
- **Delivery is best effort.** If a write fails after retries, those events are
  dropped and the next write that lands starts with an `events_dropped` entry
  giving the count. Events queued in the last seconds before a coordinator is
  killed or its VM stops can be lost without that marker.
- **`wall_seconds`** is coordinator-observed: from the hand-out to the result
  arriving (including upload), not model time.
- **Only flagged runs.** Runs without `CLOUD_LOGGING=1`, and runs before the
  metrics were created (charts only), are absent.
- **The `run_id` filter does not apply to the event list**, which shows all
  runs; each line starts with its run id.

## Events

Log `projects/PROJECT_ID/logs/multinet-run-events`, resource `global`. Every
entry's `jsonPayload` has `event`, `run_id`, `job_id` and a one-line `message`.

| Field | `episode_started` | `episode_ended` |
|---|---|---|
| `unit_id`, `task_id`, `model_id`, `model_group`, `provider`, `seed`, `prompt_variant`, `backend` | yes | yes |
| `worker_id`, `worker_host` | worker it went to | worker that held it |
| `attempt` (fresh hand-outs so far), `resume_count` | yes | yes |
| `resumed` | yes | |
| `status`, `outcome` | | yes |
| `success`, `end_reason`, `steps_used`, `max_steps`, `query_count` | | from the uploaded `episode.json` (null when there is none) |
| `wall_seconds`, `reason` (why an attempt ended without an episode) | | yes |

`coordinator_started` carries `unit_count`, `units` (unit counts by status at
start, so a restarted coordinator shows what is already verified) and `models`.
`events_dropped` carries `count`.

Severity is `WARNING` for `infra_killed`, `failed`, `stale` and
`events_dropped`, `INFO` otherwise.

To chart something new, add a log-based metric over these fields (for example
a distribution of `steps_used`) and a widget to `dashboard.json`, then re-run
`apply.sh`. `tests/test_run_events.py` checks that every metric label and
dashboard grouping names a field the events carry.

## What it costs

Nothing runs between runs: no VM, container, scheduler or uptime check. The
dashboard and log-based metrics cost nothing at rest. Measured entry size is
about 650 bytes; an episode attempt writes two.

| Line | Usage | Cost |
|---|---|---|
| Cloud Logging ingestion | ~1.3 MB per 1,000 episodes | $0: the first 50 GiB per project per month are free, then $0.50/GiB |
| Log retention | `_Default` bucket, 30 days | $0 |
| Log-based metrics | two counters, a handful of series per run (run × model × outcome) | $0 within the free 150 MiB of metric data per billing account per month |
| Dashboard and viewing | | $0 |

Labels are kept to run, model and outcome (not maze) so the number of
metric series stays small. Google changes these rates; re-check before
relying on them.

## Teardown

```bash
gcloud monitoring dashboards list --project "$PROJECT" \
  --filter='displayName="MultiNet runs"' --format='value(name)' \
  | xargs -r -n1 gcloud monitoring dashboards delete --project "$PROJECT" --quiet
gcloud logging metrics delete multinet_episodes_started --project "$PROJECT" --quiet
gcloud logging metrics delete multinet_episodes_ended --project "$PROJECT" --quiet
```

Written entries age out of the `_Default` bucket after 30 days.
