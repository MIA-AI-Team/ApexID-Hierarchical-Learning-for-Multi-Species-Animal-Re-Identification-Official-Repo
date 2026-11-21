#!/bin/bash
#SBATCH --job-name=apexid-train
#SBATCH --account=aisc
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --constraint=ARCH:X86
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=48:00:00
#SBATCH --output=apexid_train_%j.out
#SBATCH --error=apexid_train_%j.err

set -euo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: submit this file with sbatch; do not run directly on lx01."
  exit 2
fi

# Slurm copies the batch script into /var/spool/slurmd/job*/ — never use
# BASH_SOURCE as the repo root. Override with REPO_ROOT if your checkout differs.
REPO_ROOT="${REPO_ROOT:-$HOME/ApexID}"
if [ ! -d "$REPO_ROOT" ]; then
  echo "ERROR: REPO_ROOT does not exist: $REPO_ROOT"
  exit 3
fi
if [ ! -f "$REPO_ROOT/requirements.txt" ] && [ ! -f "$REPO_ROOT/requirements-split-v2.txt" ] && [ ! -f "$REPO_ROOT/requirements-lock.txt" ]; then
  echo "ERROR: REPO_ROOT does not look like an ApexID checkout: $REPO_ROOT"
  exit 3
fi
CONFIG="${CONFIG:-configs/cluster_home.yaml}"
export APEXID_DATA_ROOT="${APEXID_DATA_ROOT:-$REPO_ROOT/ApexID_Dataset}"
export APEXID_SPLIT_ROOT="${APEXID_SPLIT_ROOT:-$APEXID_DATA_ROOT/open_set_splits}"
export APEXID_OUTPUT_ROOT="${APEXID_OUTPUT_ROOT:-$REPO_ROOT/outputs}"

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "REPO_ROOT=$REPO_ROOT"
echo "CONFIG=$CONFIG"

WORKDIR="${SLURM_TMPDIR:-/tmp}/apexid_train_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"

UV="${UV_BIN:-$(command -v uv || true)}"
if [ -z "$UV" ]; then
  echo "ERROR: uv is not on PATH; set UV_BIN to its executable."
  exit 3
fi

VENV="$WORKDIR/venv"
$UV venv --no-project "$VENV"
PY="$VENV/bin/python"

$UV pip install --python "$PY" --torch-backend="${APEXID_TORCH_BACKEND:-auto}" torch==2.10.0 torchvision==0.25.0
$UV pip install --python "$PY" --no-deps -r "$REPO_ROOT/requirements-lock.txt"
$PY -c "import cv2; print('cv2', cv2.__version__)"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"

ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"
if [ ! -f "$ENV_FILE" ] && [ -f "$HOME/.env.apexid" ]; then
  ENV_FILE="$HOME/.env.apexid"
fi
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
  echo "Loaded env from $ENV_FILE"
else
  echo "No env file at $ENV_FILE (optional; scp .env to ~/ApexID/.env)"
fi

cd "$REPO_ROOT"
nvidia-smi || true

TRAIN_ARGS=(--config "$CONFIG")
if [ "${REQUIRE_CLEAN_GIT:-0}" = "1" ]; then
  TRAIN_ARGS+=(--require-clean-git)
fi
$UV run --no-project -p "$PY" python scripts/train.py "${TRAIN_ARGS[@]}"
