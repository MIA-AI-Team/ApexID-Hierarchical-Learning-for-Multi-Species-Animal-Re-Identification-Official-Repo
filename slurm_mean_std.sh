#!/bin/bash
#SBATCH --job-name=apexid-mean-std
#SBATCH --account=aisc
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --constraint=ARCH:X86
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=apexid_mean_std_%j.out
#SBATCH --error=apexid_mean_std_%j.err

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
DATA_ROOT="${APEXID_DATA_ROOT:-$REPO_ROOT/ApexID_Dataset}"
SPLIT_DIR="${APEXID_SPLIT_ROOT:-$DATA_ROOT/open_set_splits_v2}"
# Set SKIP_ADDITIONAL_CONFIGS=1 to only update $CONFIG (recommended for gray3 dsstats).
SKIP_ADDITIONAL_CONFIGS="${SKIP_ADDITIONAL_CONFIGS:-0}"
PROVENANCE_OUTPUT="${PROVENANCE_OUTPUT:-$SPLIT_DIR/normalization_stats.json}"

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "REPO_ROOT=$REPO_ROOT"
echo "CONFIG=$CONFIG"
echo "DATA_ROOT=$DATA_ROOT"
echo "SPLIT_DIR=$SPLIT_DIR"
echo "SKIP_ADDITIONAL_CONFIGS=$SKIP_ADDITIONAL_CONFIGS"
echo "PROVENANCE_OUTPUT=$PROVENANCE_OUTPUT"

WORKDIR="${SLURM_TMPDIR:-/tmp}/apexid_mean_std_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"

UV="${UV_BIN:-$(command -v uv || true)}"
if [ -z "$UV" ]; then
  echo "ERROR: uv is not on PATH; set UV_BIN to its executable."
  exit 3
fi

VENV="$WORKDIR/venv"
$UV venv --no-project "$VENV"
PY="$VENV/bin/python"

# Use the canonical exact lock on CPU; H100 validation is a separate handoff.
$UV pip install --python "$PY" --torch-backend=cpu torch==2.10.0 torchvision==0.25.0
$UV pip install --python "$PY" --no-deps -r "$REPO_ROOT/requirements-lock.txt"

cd "$REPO_ROOT"

MEAN_STD_ARGS=(
  --config "$CONFIG"
  --csv "$SPLIT_DIR/train.csv"
  --root-dir "$DATA_ROOT"
  --split train
  --write-config
  --manifest "$SPLIT_DIR/split_manifest.csv"
  --provenance-output "$PROVENANCE_OUTPUT"
)

if [ "$SKIP_ADDITIONAL_CONFIGS" != "1" ]; then
  for extra in \
    configs/default.yaml \
    configs/cluster_pretrained.yaml \
    configs/cluster_subcenter_dynamic.yaml; do
    if [ -f "$REPO_ROOT/$extra" ]; then
      MEAN_STD_ARGS+=(--additional-config "$extra")
    else
      echo "Skipping missing additional config: $extra"
    fi
  done
fi

$UV run --no-project -p "$PY" python -m apexid.utils.calculate_mean_std "${MEAN_STD_ARGS[@]}"

echo "Updated transform mean/std in:"
echo "$CONFIG"
grep -A 8 "transforms:" "$CONFIG" | head -12
if [ "$SKIP_ADDITIONAL_CONFIGS" != "1" ]; then
  for config_path in \
    configs/default.yaml \
    configs/cluster_pretrained.yaml \
    configs/cluster_subcenter_dynamic.yaml; do
    if [ -f "$REPO_ROOT/$config_path" ]; then
      echo "$config_path"
      grep -A 6 "transforms:" "$config_path" | head -8
    fi
  done
fi
sha256sum "$PROVENANCE_OUTPUT"
