#!/usr/bin/env bash
# Setup for a Qwen3.6 offline-vLLM A100 worker:
#   venv -> vLLM -> editable repo -> weights -> vLLM smoke.
# Idempotent: re-running skips work already cached by pip/Hugging Face.
set -euo pipefail

MODEL="Qwen/Qwen3.6-27B"   # FP16 on the qwen-fp16-80 A100-80GB image (FP8 retired)
VLLM_SPEC="${VLLM_SPEC:-vllm}"
HF_HUB_SPEC="${HF_HUB_SPEC:-huggingface_hub[cli]}"
VENV="./.venv-qwen-vllm"
MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION=0.88
DTYPE="auto"
ENFORCE_EAGER=1
HF_DOWNLOAD_WORKERS="${HF_DOWNLOAD_WORKERS:-4}"
DRY_RUN=0
LOG="deploy/setup_qwen_vllm_vm.log"
HF_INCLUDE_PATTERNS=("*.safetensors" "*.index.json" "*.json" "*.jinja" "*.txt")

usage() { grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --vllm-spec) VLLM_SPEC="$2"; shift 2;;
    --hf-hub-spec) HF_HUB_SPEC="$2"; shift 2;;
    --venv) VENV="$2"; shift 2;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2;;
    --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2;;
    --dtype) DTYPE="$2"; shift 2;;
    --enforce-eager) ENFORCE_EAGER=1; shift;;
    --no-enforce-eager) ENFORCE_EAGER=0; shift;;
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
  # Current vLLM wheels bring their own compatible Torch/CUDA stack, so use the
  # latest vLLM by default. Override VLLM_SPEC/--vllm-spec for bisects.
  run python -m pip install -U "$VLLM_SPEC" "$HF_HUB_SPEC"
  # The Deep Learning VM exposes system-site torchaudio. If vLLM upgrades Torch
  # in the venv, force a matching torchaudio wheel into the venv as well.
  local torch_version
  torch_version="$(python -c "import torch; print(torch.__version__.split('+')[0])")"
  run python -m pip install --force-reinstall --no-deps "torchaudio==$torch_version"
  run python -m pip install -e ".[dev]"
}

download_weights() {
  local include_args=()
  local pattern
  for pattern in "${HF_INCLUDE_PATTERNS[@]}"; do
    include_args+=(--include "$pattern")
  done
  export HF_HUB_ENABLE_HF_TRANSFER=0
  export HF_HUB_DISABLE_XET=1
  log "downloading $MODEL"
  run hf download "$MODEL" "${include_args[@]}" --max-workers "$HF_DOWNLOAD_WORKERS"
}

verify() {
  local eager_arg
  if [[ "$ENFORCE_EAGER" -eq 1 ]]; then
    eager_arg="--enforce-eager"
  else
    eager_arg="--no-enforce-eager"
  fi
  run python -c "import vllm; print('vllm', getattr(vllm, '__version__', 'unknown'))"
  run python deploy/smoke_qwen_vllm.py \
    --model "$MODEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --dtype "$DTYPE" \
    --max-tokens 128 \
    "$eager_arg" \
    --local-files-only
}

main() {
  mkdir -p "$(dirname "$LOG")"
  log "model=$MODEL vllm_spec=$VLLM_SPEC hf_hub_spec=$HF_HUB_SPEC venv=$VENV max_model_len=$MAX_MODEL_LEN gpu_mem=$GPU_MEMORY_UTILIZATION dtype=$DTYPE enforce_eager=$ENFORCE_EAGER dry_run=$DRY_RUN"
  ensure_gpu
  ensure_build_deps
  ensure_venv
  install_runtime
  download_weights
  verify
  log "DONE. Create a machine image from this VM for one offline-vLLM Qwen worker per GPU."
}

main "$@" > >(tee -a "$LOG") 2>&1
