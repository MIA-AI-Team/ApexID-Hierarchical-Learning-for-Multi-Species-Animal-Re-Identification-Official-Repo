#!/bin/bash
#SBATCH --job-name=apexid-train
#SBATCH --account=aisc
#SBATCH --partition=aisc-batch
#SBATCH --constraint=ARCH:X86
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=48:00:00
#SBATCH --output=/sc/home/%u/logs/apexid_train_%j.out
#SBATCH --error=/sc/home/%u/logs/apexid_train_%j.err

set -euo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: submit this file with sbatch; do not run directly on lx01."
  exit 2
fi

REPO_ROOT="$HOME/ApexID"
CONFIG="${CONFIG:-configs/cluster_home.yaml}"

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "REPO_ROOT=$REPO_ROOT"
echo "CONFIG=$CONFIG"

WORKDIR="${SLURM_TMPDIR:-/tmp}/apexid_train_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"

cp -f "$HOME/.local/bin/uv" "$WORKDIR/uv"
chmod +x "$WORKDIR/uv"
UV="$WORKDIR/uv"

VENV="$WORKDIR/venv"
$UV venv --no-project "$VENV"
PY="$VENV/bin/python"

$UV pip install --python "$PY" --torch-backend=cu124 -r "$REPO_ROOT/requirements.txt"
# GPU nodes are headless; opencv-python needs libGL.so.1. Reinstall cv2 via headless only.
$UV pip uninstall --python "$PY" -y opencv-python opencv-python-headless 2>/dev/null || true
$UV pip install --python "$PY" opencv-python-headless
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

$UV run --no-project -p "$PY" python scripts/train.py --config "$CONFIG"
