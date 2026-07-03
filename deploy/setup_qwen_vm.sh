#!/usr/bin/env bash
# Single invasive setup for a Qwen3.6 A100 VM (Ubuntu/apt):
#   driver/CUDA -> venv -> torch(cu128) + transformers + fast kernels -> weights -> verify.
# Idempotent: re-running skips work already done. Use --dry-run to preview.
set -euo pipefail

MODEL="27b"            # 27b | moe | both
VENV="./.venv-qwen"
CUDA_VERSION="12.8"
SKIP_DRIVER=0
DRY_RUN=0
LOG="deploy/setup_qwen_vm.log"
HF_INCLUDE_PATTERNS=("*.safetensors" "*.index.json" "*.json" "*.jinja" "*.txt")

QWEN_27B="Qwen/Qwen3.6-27B"
QWEN_MOE="Qwen/Qwen3.6-35B-A3B"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
TORCH_PKGS=("torch==2.9.1+cu128" "torchvision" "torchaudio")

usage() { grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --venv) VENV="$2"; shift 2;;
    --cuda) CUDA_VERSION="$2"; shift 2;;
    --skip-driver) SKIP_DRIVER=1; shift;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

run() {
  echo "+ $*"
  if [[ "$DRY_RUN" -eq 0 ]]; then "$@"; fi
}

log() { echo "[setup_qwen_vm] $*"; }

ensure_driver() {
  if [[ "$SKIP_DRIVER" -eq 1 ]]; then log "skip-driver set; not touching driver/CUDA"; return; fi
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    log "nvidia-smi healthy; skipping driver install"
  else
    log "installing NVIDIA driver via apt (ubuntu-drivers autoinstall)"
    run sudo apt-get update
    run sudo apt-get install -y ubuntu-drivers-common
    run sudo ubuntu-drivers autoinstall
    log "driver installed; a reboot may be required before nvidia-smi works"
  fi
  if command -v nvcc >/dev/null 2>&1; then
    log "nvcc present: $(nvcc --version 2>/dev/null | tail -1 || true)"
  else
    log "installing CUDA toolkit ${CUDA_VERSION} (for building flash-attn / causal-conv1d)"
    local pkg="cuda-toolkit-${CUDA_VERSION/./-}"
    run sudo apt-get install -y "${pkg}" || log "WARN: ${pkg} not available via apt; install the toolkit manually if kernel builds fail"
  fi
}

ensure_build_deps() {
  run sudo apt-get install -y build-essential ninja-build git python3-venv python3-dev
}

ensure_venv() {
  # GCP Deep Learning images (e.g. common-cu129) auto-activate a conda base env.
  # A venv built/activated on top of it stacks with conda: pip and imports resolve
  # against /opt/conda instead of the venv. Build from the system interpreter and
  # drop conda/PYTHONPATH so the venv is the only thing on the path.
  local py="python3"
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    log "conda base active ($CONDA_PREFIX); isolating venv from it"
    if [[ -x /usr/bin/python3 ]]; then py="/usr/bin/python3"; fi
  fi
  unset PYTHONPATH || true
  # Rebuild a partial/broken venv left by a prior run that died mid-setup.
  if [[ -d "$VENV" && ! -x "$VENV/bin/python" ]]; then
    log "incomplete venv at $VENV; recreating"
    run rm -rf "$VENV"
  fi
  if [[ ! -d "$VENV" ]]; then run "$py" -m venv "$VENV"; fi
  # shellcheck disable=SC1091
  if [[ "$DRY_RUN" -eq 0 ]]; then source "$VENV/bin/activate"; fi
  run python -m pip install -U pip wheel setuptools
}

install_torch() {
  run python -m pip install -U --index-url "$TORCH_INDEX" "${TORCH_PKGS[@]}"
}

install_runtime() {
  run python -m pip install -U transformers accelerate bitsandbytes pillow einops "huggingface_hub[cli]"
}

install_kernels() {
  # Warn-only: the model still runs (slower) without these. Install each kernel
  # separately so one failing build does not skip the others. All three import
  # torch at build time, so they need --no-build-isolation.
  run python -m pip install -U --no-build-isolation flash-linear-attention || log "WARN: flash-linear-attention build failed (continuing)"
  run python -m pip install -U --no-build-isolation causal-conv1d || log "WARN: causal-conv1d build failed (continuing)"
  run python -m pip install -U --no-build-isolation flash-attn || log "WARN: flash-attn build failed (continuing)"
}

download_weights() {
  local repos=()
  case "$MODEL" in
    27b) repos=("$QWEN_27B");;
    moe) repos=("$QWEN_MOE");;
    both) repos=("$QWEN_27B" "$QWEN_MOE");;
    *) echo "Unknown --model: $MODEL" >&2; exit 2;;
  esac
  export HF_HUB_ENABLE_HF_TRANSFER=0
  for repo in "${repos[@]}"; do
    log "downloading $repo"
    run hf download "$repo" --include "${HF_INCLUDE_PATTERNS[@]}" --max-workers 16
  done
}

verify() {
  run python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"
  local smoke_model="$QWEN_27B"
  [[ "$MODEL" == "moe" ]] && smoke_model="$QWEN_MOE"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    # Real load + decode so a broken driver/weights/model-class fails here rather
    # than being snapshotted. --self-test (no model load) is for CI / dry-run only.
    run python deploy/smoke_qwen.py --model "$smoke_model"
  else
    log "dry-run: would run 'python deploy/smoke_qwen.py --model $smoke_model'"
  fi
}

main() {
  mkdir -p "$(dirname "$LOG")"
  log "model=$MODEL venv=$VENV cuda=$CUDA_VERSION dry_run=$DRY_RUN"
  ensure_driver
  ensure_build_deps
  ensure_venv
  install_torch
  install_runtime
  install_kernels
  download_weights
  verify
  log "DONE. If nvidia-smi/CUDA are healthy and the smoke prints tok/s, snapshot this image for the other Qwen runners."
}

main "$@" > >(tee -a "$LOG") 2>&1
