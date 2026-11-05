#!/usr/bin/env bash
# Create or update the run dashboard and its log-based metric in one GCP
# project (deploy/MONITORING.md). Safe to re-run: the metric and the dashboard
# named "MultiNet runs" are updated in place, never duplicated. The progress
# metrics (custom.googleapis.com/multinet/*) need no setup: a coordinator's
# first write creates them. Nothing here touches a VM or a run, and none of it
# costs anything at rest.
#
#   deploy/monitoring/apply.sh PROJECT_ID
set -euo pipefail

PROJECT="${1:?usage: deploy/monitoring/apply.sh PROJECT_ID (deliberately not defaulted)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DASHBOARD_NAME="MultiNet runs"

METRIC=multinet_episodes_ended
if gcloud logging metrics describe "$METRIC" --project "$PROJECT" >/dev/null 2>&1; then
  gcloud logging metrics update "$METRIC" --project "$PROJECT" --config-from-file "$HERE/$METRIC.json"
else
  gcloud logging metrics create "$METRIC" --project "$PROJECT" --config-from-file "$HERE/$METRIC.json"
fi

existing="$(gcloud monitoring dashboards list --project "$PROJECT" \
  --filter="displayName=\"$DASHBOARD_NAME\"" --format='value(name)')"
if [[ "$(printf '%s' "$existing" | grep -c .)" -gt 1 ]]; then
  echo "More than one dashboard is named \"$DASHBOARD_NAME\"; delete the extras first:" >&2
  printf '%s\n' "$existing" >&2
  exit 1
elif [[ -n "$existing" ]]; then
  # The API only takes an update that carries the dashboard's current etag.
  etag="$(gcloud monitoring dashboards describe "$existing" --project "$PROJECT" --format='value(etag)')"
  gcloud monitoring dashboards update "$existing" --project "$PROJECT" \
    --config "$(sed "1s/^{/{\"etag\": \"$etag\",/" "$HERE/dashboard.json")"
else
  gcloud monitoring dashboards create --project "$PROJECT" --config-from-file "$HERE/dashboard.json"
fi

echo "Dashboard: https://console.cloud.google.com/monitoring/dashboards?project=$PROJECT"
echo "Events:    https://console.cloud.google.com/logs/query;query=log_id(%22multinet-run-events%22)?project=$PROJECT"
