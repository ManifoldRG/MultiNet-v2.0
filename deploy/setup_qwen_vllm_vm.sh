#!/usr/bin/env bash
# Setup for a Qwen3.6 offline-vLLM A100 worker:
#   venv -> vLLM -> editable repo -> FP8 weights -> vLLM smoke.
# Idempotent: re-running skips work already cached by pip/Hugging Face.
set -euo pipefail

MODEL="Qwen/Qwen3.6-27B-FP8"
VENV="./.venv-qwen-vllm"
MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION=0.88
HF_DOWNLOAD_WORKERS="${HF_DOWNLOAD_WORKERS:-4}"
DRY_RUN=0
LOG="deploy/setup_qwen_vllm_vm.log"
HF_INCLUDE_PATTERNS=("*.safetensors" "*.index.json" "*.json" "*.jinja" "*.txt")

usage() { grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --venv) VENV="$2"; shift 2;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2;;
    --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

run() {
  echo "+ $*"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@" || {
      local status=$?
      echo "Command failed with exit $status: $*" >&2
      exit "$status"
    }
  fi
}

log() { echo "[setup_qwen_vllm_vm] $*"; }

ensure_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found; install/repair the NVIDIA driver first." >&2
    exit 1
  fi
  run nvidia-smi
}

ensure_build_deps() {
  run sudo apt-get install -y build-essential git python3-venv python3-dev
}

ensure_venv() {
  local py="python3"
  if [[ -n "${CONDA_PREFIX:-}" && -x /usr/bin/python3 ]]; then
    py="/usr/bin/python3"
  fi
  unset PYTHONPATH || true
  if [[ -d "$VENV" && ! -x "$VENV/bin/python" ]]; then
    run rm -rf "$VENV"
  fi
  if [[ ! -d "$VENV" ]]; then
    # Reuse the Deep Learning image's preinstalled PyTorch/CUDA stack instead of
    # downloading another full torch wheel set into the venv.
    run "$py" -m venv --system-site-packages "$VENV"
  fi
  # shellcheck disable=SC1091
  if [[ "$DRY_RUN" -eq 0 ]]; then source "$VENV/bin/activate"; fi
  run python -m pip install -U pip wheel setuptools
}

install_runtime() {
  # Qwen3.6 documents vLLM >=0.19.0. Pin the floor for now: newer vLLM releases
  # pull newer torch stacks and change the runtime surface while we are validating.
  run python -m pip install -U "vllm==0.19.0" "huggingface_hub[cli]>=0.36,<0.37"
  run python -m pip install -e ".[dev]"
}

download_weights() {
  export HF_HUB_ENABLE_HF_TRANSFER=0
  export HF_HUB_DISABLE_XET=1
  log "downloading $MODEL"
  run hf download "$MODEL" --include "${HF_INCLUDE_PATTERNS[@]}" --max-workers "$HF_DOWNLOAD_WORKERS"
}

verify() {
  run python -c "import vllm; print('vllm', getattr(vllm, '__version__', 'unknown'))"
  run python deploy/smoke_qwen_vllm.py \
    --model "$MODEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-tokens 128 \
    --enforce-eager \
    --local-files-only
}

main() {
  mkdir -p "$(dirname "$LOG")"
  log "model=$MODEL venv=$VENV max_model_len=$MAX_MODEL_LEN gpu_mem=$GPU_MEMORY_UTILIZATION dry_run=$DRY_RUN"
  ensure_gpu
  ensure_build_deps
  ensure_venv
  install_runtime
  download_weights
  verify
  log "DONE. Snapshot this VM for one offline-vLLM Qwen worker per GPU."
}

main "$@" > >(tee -a "$LOG") 2>&1
