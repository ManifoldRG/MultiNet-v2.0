#!/usr/bin/env bash
# Create or update the run dashboard and its two log-based metrics in one GCP
# project (deploy/MONITORING.md). Safe to re-run: existing metrics and the
# dashboard named "MultiNet runs" are updated in place, never duplicated.
# Nothing here touches a VM or a run, and none of it costs anything at rest.
#
#   deploy/monitoring/apply.sh PROJECT_ID
set -euo pipefail

PROJECT="${1:?usage: deploy/monitoring/apply.sh PROJECT_ID (deliberately not defaulted)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DASHBOARD_NAME="MultiNet runs"

for metric in multinet_episodes_started multinet_episodes_ended; do
  if gcloud logging metrics describe "$metric" --project "$PROJECT" >/dev/null 2>&1; then
    gcloud logging metrics update "$metric" --project "$PROJECT" --config-from-file "$HERE/$metric.json"
  else
    gcloud logging metrics create "$metric" --project "$PROJECT" --config-from-file "$HERE/$metric.json"
  fi
done

existing="$(gcloud monitoring dashboards list --project "$PROJECT" \
  --filter="displayName=\"$DASHBOARD_NAME\"" --format='value(name)')"
if [[ "$(printf '%s' "$existing" | grep -c .)" -gt 1 ]]; then
  echo "More than one dashboard is named \"$DASHBOARD_NAME\"; delete the extras first:" >&2
  printf '%s\n' "$existing" >&2
  exit 1
elif [[ -n "$existing" ]]; then
  gcloud monitoring dashboards update "$existing" --project "$PROJECT" --config-from-file "$HERE/dashboard.json"
else
  gcloud monitoring dashboards create --project "$PROJECT" --config-from-file "$HERE/dashboard.json"
fi

echo "Dashboard: https://console.cloud.google.com/monitoring/dashboards?project=$PROJECT"
echo "Events:    https://console.cloud.google.com/logs/query;query=log_id(%22multinet-run-events%22)?project=$PROJECT"
