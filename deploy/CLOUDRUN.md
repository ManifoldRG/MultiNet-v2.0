# Deploying the maze API to Cloud Run

The backend behind the playable maze on multinet.ai. FastAPI, one container,
no database.

## What you are actually creating

**No VM.** Cloud Run is serverless: you hand Google a container image and it
returns an HTTPS URL. Nothing to SSH into, no OS to patch, no disk to manage.
Where this doc says "instance" it means a running copy of the container, which
is Cloud Run's own term - not a Compute Engine machine.

**Who can reach it.** `--allow-unauthenticated` makes the service public to
the internet. That is required, because the caller is an anonymous visitor's
browser on multinet.ai. Two things follow:

- `MULTINET_CORS_ORIGINS` is *not* a security control. It stops other websites
  from calling the API from a browser; it stops nothing from calling it with
  curl. Treat the endpoint as fully public.
- The only exposure is the maze itself. The API starts and steps MiniGrid
  episodes and reads a bundled results CSV. It has no database, no secrets, no
  write path to anything, and no access to the rest of the project. Worst case
  from abuse is compute cost and games evicted early by the LRU cap.

**What it costs.** Roughly **$9-11/month**, and close to flat regardless of
traffic - about 99% of it is the charge for keeping one container warm. Full
breakdown in "What it costs" below. `--max-instances=1` also caps the blast
radius: a traffic spike or a scraper cannot fan out into a surprise bill, it
just queues.

**Which project.** Pass `--project` explicitly on every command below rather
than relying on whatever `gcloud config` happens to be set to. This service is
public and unauthenticated, so it is worth putting it somewhere you are happy
to have that, rather than inheriting a default project set for other work.

```bash
PROJECT=your-project-id       # deliberately not defaulted - pick it
REGION=us-central1
```

## The constraint that decides everything

`GameRegistry` (`demo/api/registry.py`) keeps live games in a Python dict in
process memory. There is no shared store. Consequences:

- **`--max-instances=1` is mandatory.** With two instances, a player's second
  move can land on a container that has never heard of their game and gets a
  404. This is the one setting you cannot relax without adding Redis or sticky
  sessions.
- **`--min-instances=1`**, so nobody pays a cold start mid-game and the
  registry is not wiped by a scale-to-zero.
- **Deploying drops in-flight games.** A new revision is a new container.
  Deploy when nobody is mid-maze, or accept that a few players see a 404 and
  have to restart.

Session cleanup is request-driven (`_purge()` runs inside `start` and `get`),
not a background task, so Cloud Run's default "CPU only during requests" is
fine and keeps idle cost to memory alone.

## First deploy

Build and run it on your laptop first. Neither step touches GCP, and the
image carries a build-time import check, so anything structurally wrong fails
here in a couple of minutes rather than after a Cloud Build round trip.

```bash
cd MultiNet-v2.0            # the Dockerfile is at the repo root

docker build -t multinet-maze .
docker run --rm -p 8080:8080 \
  -e MULTINET_CORS_ORIGINS=http://127.0.0.1:8899 \
  multinet-maze
```

With that running, open the local site against it and play a maze end to end:

```
http://127.0.0.1:8899/?api=http://127.0.0.1:8080
```

Only once that works, deploy:

```bash
gcloud run deploy multinet-maze \
  --project "$PROJECT" \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --min-instances=1 \
  --max-instances=1 \
  --memory=512Mi \
  --cpu=1 \
  --concurrency=80 \
  --set-env-vars MULTINET_CORS_ORIGINS=https://multinet.ai,MULTINET_MAX_GAMES=256
```

`--source .` picks up the root `Dockerfile` rather than guessing a buildpack.
It uploads the build context to Cloud Build, so the first deploy will prompt to
enable the Cloud Build and Artifact Registry APIs if they are not already on.

## Sizing

Measured in this image, not estimated: **62MB resident at idle**, 0.13% CPU
idle, ~0.4MB per live game, 73ms to start a game, 4ms per move. At
`MULTINET_MAX_GAMES=256` the ceiling is roughly 164MB, so 512Mi has real
headroom. Raise memory before raising the game cap.

(An earlier ~113MB figure came from the desktop pygame path, not this image.
The container never opens a display - see the SDL dummy drivers in the
Dockerfile - and is correspondingly lighter.)

`--concurrency=80` is the Cloud Run default and is not the bottleneck - moves
are single-digit milliseconds. With `--max-instances=1`, 80 simultaneous
in-flight *requests* is the hard ceiling; concurrent *players* can be far
higher, since a player is idle between moves.

Note `_evict_overflow` uses `>=`, so `MULTINET_MAX_GAMES=256` actually holds
255 games. Harmless, just don't be surprised by the off-by-one.

## Environment

| Variable | Value | Why |
|---|---|---|
| `MULTINET_CORS_ORIGINS` | `https://multinet.ai` | Defaults to localhost only; without this every browser request fails. Add other origins comma-separated. |
| `MULTINET_MAX_GAMES` | `256` | Default 64 is a dev cap. |
| `MULTINET_R1_RESULTS_CSV` | *(unset)* | Only if the results table is mounted somewhere other than the vendored `demo/data/` copy. |
| `MULTINET_SETTINGS_EDITABLE` | *(unset)* | Leave off. Frozen config is the R1 protocol. |
| `PORT` | *(injected)* | Cloud Run sets it; the Dockerfile honours it. |

## Verify

```bash
URL=$(gcloud run services describe multinet-maze \
        --project "$PROJECT" --region "$REGION" \
        --format='value(status.url)')

curl -s "$URL/api/health"                        # {"ok":true}
curl -s "$URL/api/game/tasks" | head -c 200      # 42 tasks

# CORS preflight must echo the site origin back, or the maze silently fails.
curl -si -X OPTIONS "$URL/api/game/start" \
  -H "Origin: https://multinet.ai" \
  -H "Access-Control-Request-Method: POST" \
  -H "Access-Control-Request-Headers: Content-Type" | grep -i access-control
```

Then point the site at it: set `API_DEFAULT` in
`multinet-website/static/js/play.js` to `$URL` (it must be https - an http
endpoint is blocked as mixed content from multinet.ai) and bump the asset
cache-buster.

Before changing the default you can test the deployed backend against the
local site with the `?api=` override:

```
http://127.0.0.1:8899/?api=https://multinet-maze-xxxx.run.app
```

## Where to watch it

Replace `PROJECT_ID`. Note this is **not** under Compute Engine - there is no VM.

| What | Where |
|---|---|
| The service | `https://console.cloud.google.com/run?project=PROJECT_ID` |
| Metrics / logs / revisions | `https://console.cloud.google.com/run/detail/us-central1/multinet-maze/metrics?project=PROJECT_ID` |
| Cost reports (filter Service = Cloud Run) | `https://console.cloud.google.com/billing/reports` |
| Budget alerts | `https://console.cloud.google.com/billing/budgets` |
| Build history | `https://console.cloud.google.com/cloud-build/builds?project=PROJECT_ID` |
| Stored images | `https://console.cloud.google.com/artifacts?project=PROJECT_ID` |

On the service's **Metrics** tab, the one to watch is **container instance
count**. It must stay pinned at 1. If it ever reads 2, `--max-instances` did
not take and players will start getting random 404s as their moves land on a
container that never saw their game.

Billing data lags up to 24 hours, so an empty report on deploy day is normal.

**Set a budget alert before you deploy.** Scope it to the project, set ~$25/mo,
alert at 50/90/100%. The service is public and unauthenticated, so this is the
practical safety net - the real risk is cost, not compromise.

## What it costs

Measured, not estimated. Response sizes came from the running container:
`/start` 4.7KB, `/action` 3.7KB average, `gridImage` 3.4KB (an encoded grid,
not a PNG). A 60-move game is about **38KB** of egress.

At 1 vCPU / 512Mi, `min=max=1`, request-based billing, us-central1. A month is
~2,628,000 seconds and `--min-instances=1` is billed for all of them.

| Line | Calculation | Cost |
|---|---|---|
| Idle CPU | (2,628,000 - 180,000 free) x $0.0000025 | $6.12 |
| Memory | (1,314,000 - 360,000 free) x $0.0000025 | $2.39 |
| Requests | 610k at 10k games - under the 2M free tier | $0.00 |
| Active CPU uplift | ~3,200 vCPU-s x $0.0000215 | $0.07 |
| Egress | 10k games x 38KB = 380MB | ~$0.05 |
| **Total** | | **~$8.60** |

**~99% of that is idle charge.** You are paying to keep a container warm, not
to serve traffic: 10,000 games costs about twelve cents. At 100,000 games a
month it is still only ~$11.40 (requests +$1.64, egress +$0.46, CPU +$0.69).

Two caveats. The free tier is per *billing account*, not per service - if
something else consumes it the baseline rises to $9.86. And Google changes
these rates, so re-check before relying on the number.

**Trimming.** The container measures 62MB resident and 0.13% CPU at idle, well
under what is provisioned. `--cpu=0.5 --memory=256Mi` would cut this to roughly
$4-5/month. Deploy at 1 vCPU / 512Mi anyway for launch week: 256Mi leaves only
~100MB above the 255-game ceiling, and half a vCPU makes a spike queue instead
of absorb. Watch the metrics for a week, then trim - it is a one-line redeploy,
and the $5 saved is worth less than a maze that stutters on launch day.

## Teardown

```bash
gcloud run services delete multinet-maze --project "$PROJECT" --region "$REGION"
```

That stops all charges for the service. Images built by `--source .` remain in
Artifact Registry and cost pennies; delete the `cloud-run-source-deploy` repo
too if you want it fully gone.
