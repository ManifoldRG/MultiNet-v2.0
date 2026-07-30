#!/usr/bin/env bash
# R1 Kimi make-up rerun — three parallel arms. See docs/r1-kimi-rerun-package.md.
#
#   Arm 1  resume M6 from step 143 under the AS-RUN harness (no DROP),
#          via the e0542b5 worktree + scripts/resume_from_archive.py
#   Arm 2  M6 fresh from step 1 under the DROP-enabled branch
#   Arm 3  D2 (wrong-key decoy) fresh under the DROP-enabled branch
#
# All three run as independent processes with independent artifact roots, so a
# crash in one never strands the others.
#
# PAID RUN. Gates, in order: Moonshot balance precheck (manual), committed
# clean tree on fix/drop-key-bookkeeping, free replay-verify pass.
#
# Usage:  MOONSHOT_API_KEY=... scripts/launch_kimi_rerun.sh <artifacts-root>
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="${1:?usage: launch_kimi_rerun.sh <artifacts-root>}"
ASRUN_SHA=e0542b5
ASRUN_TREE=/tmp/multinet-r1-asrun
# The suite maximum R1 was launched with (RUN_NOTES_incidents_and_findings.md:
# "launched with DIFFICULTY_MAX=3000"). It normalizes difficulty_weight, which
# feeds the composite, so it MUST match R1 or the rerun rows are not comparable.
# Cross-checked against the archive: M6 1120.367/0.373456 = 3000.0 exactly, and
# D2 631.234/0.210411 = 3000.0. The shipped scorer config leaves it unset (and
# the sweep default of 1000 is below M6's static score, so it would hard-fail).
DIFFICULTY_MAX=3000
ARCHIVE="$ROOT/Multinet-v2-results/r1-20260717/runs/r1_M6_14x14_dense_kr_sg_kb_1/minigrid/kimi-k2.6/seed_0/default"

# ---------------------------------------------------------------- gates
: "${MOONSHOT_API_KEY:?MOONSHOT_API_KEY is required (and check the Moonshot
console balance FIRST — the batch precheck once killed a leg at \$0 spent)}"

branch="$(git branch --show-current)"
[ "$branch" = "fix/drop-key-bookkeeping" ] || {
  echo "FATAL: on branch '$branch', expected fix/drop-key-bookkeeping" >&2; exit 1; }
git diff --quiet && git diff --cached --quiet || {
  echo "FATAL: working tree not clean (code-sync invariant)" >&2; exit 1; }

[ -d "$ASRUN_TREE" ] || git worktree add "$ASRUN_TREE" "$ASRUN_SHA"

echo "== free gate: deterministic replay of the archived episode =="
python scripts/resume_from_archive.py \
  --archive-dir "$ARCHIVE" --repo-root "$ASRUN_TREE" \
  --out-dir "$OUT/arm1_resume_m6_verify" --mode replay-verify

mkdir -p "$OUT"/{arm1_resume_m6,fresh_m6,fresh_d2,logs}

# ---------------------------------------------------------------- arms
echo "== launching arm 1 (resume M6, as-run harness, no DROP) =="
nohup python scripts/resume_from_archive.py \
  --archive-dir "$ARCHIVE" --repo-root "$ASRUN_TREE" \
  --out-dir "$OUT/arm1_resume_m6" --mode continue --max-new-queries 200 \
  --timeout 2400 \
  > "$OUT/logs/arm1_resume.log" 2>&1 &
ARM1=$!

# Arms 2 and 3 run as SEPARATE processes with separate artifact roots: the
# local pipeline is sequential (max_in_flight is a distributed-only knob) and
# has no per-episode exception isolation, so one process per maze both halves
# wall clock and stops a crash in one arm from stranding the other.
echo "== launching arm 2 (fresh M6, DROP-enabled harness) =="
nohup python -m scripts.run_pipeline \
  --run-config gridworld/fixtures/run_config.r1.kimi_rerun_m6.json \
  --manifest gridworld/fixtures/manifest.r1_kimi_rerun.json \
  --seeds 0 --artifacts-root "$OUT/fresh_m6" \
  --difficulty-max-static-score "$DIFFICULTY_MAX" \
  > "$OUT/logs/arm2_fresh_m6.log" 2>&1 &
ARM2=$!

echo "== launching arm 3 (fresh D2 wrong-key, DROP-enabled harness) =="
nohup python -m scripts.run_pipeline \
  --run-config gridworld/fixtures/run_config.r1.kimi_rerun_d2.json \
  --manifest gridworld/fixtures/manifest.r1_kimi_rerun.json \
  --seeds 0 --artifacts-root "$OUT/fresh_d2" \
  --difficulty-max-static-score "$DIFFICULTY_MAX" \
  > "$OUT/logs/arm3_fresh_d2.log" 2>&1 &
ARM3=$!

echo "arm1 pid=$ARM1  arm2 pid=$ARM2  arm3 pid=$ARM3 — waiting..."
FAIL=0
wait "$ARM1" || { echo "ARM 1 EXITED NONZERO — see logs/arm1_resume.log";   FAIL=1; }
wait "$ARM2" || { echo "ARM 2 EXITED NONZERO — see logs/arm2_fresh_m6.log"; FAIL=1; }
wait "$ARM3" || { echo "ARM 3 EXITED NONZERO — see logs/arm3_fresh_d2.log"; FAIL=1; }

# ---------------------------------------------------------------- summary
echo "== outcomes =="
python3 - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
for label, p in [
    ("arm1 resume M6 (no DROP)", out / "arm1_resume_m6" / "episode.json"),
    ("arm2 fresh M6 (DROP)", next(iter((out / "fresh_m6").glob(
        "runs/r1_M6*/minigrid/kimi-k2.6/seed_0/*/episode.json")), None)),
    ("arm3 fresh D2 (DROP)", next(iter((out / "fresh_d2").glob(
        "runs/r1_D2*/minigrid/kimi-k2.6/seed_0/*/episode.json")), None)),
]:
    if p and Path(p).is_file():
        ep = json.load(open(p))
        steps = [t for t in ep["transcript"] if t["kind"] == "step"]
        drops = sum(1 for s in steps if s.get("action") == "DROP")
        print(f"{label}: end={ep.get('end_reason')} success={ep.get('success')} "
              f"steps={ep.get('steps_used')} DROP-actions={drops}")
    else:
        print(f"{label}: NO EPISODE ARTIFACT")
EOF
exit "$FAIL"
