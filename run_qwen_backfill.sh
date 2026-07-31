#!/usr/bin/env bash
# run_qwen_backfill.sh — finish the Qwen run locally: the 3 units the cloud fleet missed.
#
#   baseline_thinking (thinking ON) : validation_10_v02_winding_corridor  [standard]
#   cond_prompt       (thinking OFF): conditional_s_s5_14x14_corridor_1    [standard] + [minimal]
#
# It auto-generates minimal 2-maze subset manifests + Qwen-only run-configs so ONLY those
# 3 units run (no re-doing the other mazes or the API models). Outputs land as
# episode_runs.jsonl under artifacts/qwen_backfill/<unit>/.
#
# Prereqs: a free GPU (~40 GB) with Qwen/Qwen3.6-27B cached; the vLLM venv.
# Usage:   bash run_qwen_backfill.sh        # from the repo root
#          QWEN_VENV=/path/to/venv bash run_qwen_backfill.sh   # override the venv
set -euo pipefail
cd "$(dirname "$0")"                                   # repo root
source "${QWEN_VENV:-.venv-qwen-vllm}/bin/activate"
source lib/vllm_serve_args.sh                          # vllm_serve_args(): two-tier serve args

OUT=artifacts/qwen_backfill; GEN="$OUT/_gen"; mkdir -p "$GEN"
BASE_RC=gridworld/fixtures/run_config.conditional_baseline_thinking_claude_kimi_qwen.json
PROMPT_RC=gridworld/fixtures/run_config.conditional_prompt_claude_kimi_qwen.json
MANIFEST=gridworld/fixtures/manifest.conditional_eval.json

echo "[1/3] generating minimal 2-maze subsets + Qwen-only run-configs -> $GEN"
python3 - "$MANIFEST" "$BASE_RC" "$PROMPT_RC" "$GEN" <<'PY'
import json, sys, pathlib
manifest, base_rc, prompt_rc, gen = sys.argv[1:5]; gen = pathlib.Path(gen)
man = json.load(open(manifest))
def subset(tids, name):
    m = dict(man); m["tasks"] = [t for t in man["tasks"] if t["task_id"] in tids]
    assert len(m["tasks"]) == len(tids), f"{name}: wanted {tids}, got {[t['task_id'] for t in m['tasks']]}"
    json.dump(m, open(gen/f"manifest.{name}.json", "w"), indent=1)
def qwen_only(src, sub_manifest, name):
    rc = json.load(open(src))
    rc["models"] = {k: v for k, v in rc["models"].items() if v.get("provider") == "qwen_vllm_api"}
    assert rc["models"], f"no qwen model in {src}"
    rc["manifest"] = str(gen/f"manifest.{sub_manifest}.json")
    for v in rc["models"].values(): v["tasks"] = ["all"]
    json.dump(rc, open(gen/f"run_config.{name}.json", "w"), indent=1)
subset({"validation_10_v02_winding_corridor"}, "winding")
subset({"conditional_s_s5_14x14_corridor_1"}, "s5")
qwen_only(base_rc,   "winding", "baseline_qwen")
qwen_only(prompt_rc, "s5",      "prompt_qwen")
print("  ok")
PY

echo "[2/3] ensuring vLLM (Qwen/Qwen3.6-27B) is serving on :8000"
if ! curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
  echo "      launching vLLM (~14 min to load; log: $OUT/vllm.log)"
  # TWO-TIER PHASE-TRANSITION POINT: serve args from vllm_serve_args() (phase-1
  # defaults reproduce today's line; QWEN_MAX_MODEL_LEN=96000 etc. -> phase 2).
  nohup env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    vllm serve Qwen/Qwen3.6-27B --served-model-name Qwen/Qwen3.6-27B $(vllm_serve_args) \
      > "$OUT/vllm.log" 2>&1 &
  for _ in $(seq 1 180); do curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break; sleep 10; done
fi
curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1 || { echo "vLLM never became ready — see $OUT/vllm.log"; exit 1; }
echo "      vLLM ready."

echo "[3/3] running the 3 missing units"
run() {  # <run-config> <manifest> <variant> <out-subdir>
  echo "  --> $4"
  python -m scripts.run_pipeline --run-config "$1" --manifest "$2" \
    --conditions Prompt --prompt-variant "$3" --seeds 0 --force \
    --artifacts-root "$OUT/$4" --run-set-id "$4"
}
run "$GEN/run_config.baseline_qwen.json" "$GEN/manifest.winding.json" standard baseline_thinking_winding
run "$GEN/run_config.prompt_qwen.json"   "$GEN/manifest.s5.json"      standard cond_prompt_s5_standard
run "$GEN/run_config.prompt_qwen.json"   "$GEN/manifest.s5.json"      minimal  cond_prompt_s5_minimal

echo
echo "DONE. Results:"
for d in baseline_thinking_winding cond_prompt_s5_standard cond_prompt_s5_minimal; do
  j="$OUT/$d/episode_runs.jsonl"
  [ -f "$j" ] && echo "  $d: $(wc -l < "$j") episode(s) -> $j" || echo "  $d: NO OUTPUT (check logs)"
done
echo "These backfill winding_corridor (baseline_thinking) + s5 (cond_prompt std+min) to 135/135 + 45/45."
