#!/usr/bin/env bash
# R1 Kimi make-up rerun — three parallel arms. See docs/r1-kimi-rerun-package.md.
#
#   Arm 1  resume M6 from step 143 under the AS-RUN harness (no DROP),
#          via the e0542b5 worktree + scripts/resume_from_archive.py
#   Arm 2  M6 fresh from step 1 under the DROP-enabled branch
#   Arm 3  D2 (wrong-key decoy) fresh under the DROP-enabled branch
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

mkdir -p "$OUT"/{arm1_resume_m6,fresh,logs}

# ---------------------------------------------------------------- arms
echo "== launching arm 1 (resume M6, as-run harness, no DROP) =="
nohup python scripts/resume_from_archive.py \
  --archive-dir "$ARCHIVE" --repo-root "$ASRUN_TREE" \
  --out-dir "$OUT/arm1_resume_m6" --mode continue --max-new-queries 200 \
  > "$OUT/logs/arm1_resume.log" 2>&1 &
ARM1=$!

echo "== launching arms 2+3 (fresh M6 + D2, DROP-enabled harness) =="
nohup python -m scripts.run_pipeline \
  --run-config gridworld/fixtures/run_config.r1.kimi_rerun.json \
  --manifest gridworld/fixtures/manifest.r1_kimi_rerun.json \
  --seeds 0 --artifacts-root "$OUT/fresh" \
  > "$OUT/logs/arms23_fresh.log" 2>&1 &
ARM23=$!

echo "arm1 pid=$ARM1  arms2+3 pid=$ARM23 — waiting..."
FAIL=0
wait "$ARM1"  || { echo "ARM 1 EXITED NONZERO — see logs/arm1_resume.log";  FAIL=1; }
wait "$ARM23" || { echo "ARMS 2+3 EXITED NONZERO — see logs/arms23_fresh.log"; FAIL=1; }

# ---------------------------------------------------------------- summary
echo "== outcomes =="
python3 - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
for label, p in [
    ("arm1 resume M6 (no DROP)", out / "arm1_resume_m6" / "episode.json"),
    ("arm2 fresh M6 (DROP)", next(iter((out / "fresh").glob(
        "runs/r1_M6*/minigrid/kimi-k2.6/seed_0/*/episode.json")), None)),
    ("arm3 fresh D2 (DROP)", next(iter((out / "fresh").glob(
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
