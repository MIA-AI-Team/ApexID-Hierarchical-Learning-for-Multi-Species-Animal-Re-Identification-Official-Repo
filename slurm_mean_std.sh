#!/bin/bash
#SBATCH --job-name=apexid-mean-std
#SBATCH --account=aisc
#SBATCH --partition=aisc-batch
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

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
CONFIG="${CONFIG:-configs/cluster_home.yaml}"
DATA_ROOT="${APEXID_DATA_ROOT:-$REPO_ROOT/ApexID_Dataset}"
SPLIT_DIR="${APEXID_SPLIT_ROOT:-$DATA_ROOT/open_set_splits_v2}"

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "REPO_ROOT=$REPO_ROOT"
echo "CONFIG=$CONFIG"
echo "DATA_ROOT=$DATA_ROOT"
echo "SPLIT_DIR=$SPLIT_DIR"

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

$UV run --no-project -p "$PY" python -m apexid.utils.calculate_mean_std \
  --config "$CONFIG" \
  --csv "$SPLIT_DIR/train.csv" \
  --root-dir "$DATA_ROOT" \
  --split train \
  --write-config \
  --additional-config configs/default.yaml \
  --additional-config configs/cluster_pretrained.yaml \
  --additional-config configs/cluster_subcenter_dynamic.yaml \
  --manifest "$SPLIT_DIR/split_manifest.csv" \
  --provenance-output "$SPLIT_DIR/normalization_stats.json"

echo "Updated transform mean/std in all four training configs:"
for config_path in \
  "$CONFIG" \
  configs/default.yaml \
  configs/cluster_pretrained.yaml \
  configs/cluster_subcenter_dynamic.yaml; do
  echo "$config_path"
  grep -A 6 "transforms:" "$config_path" | head -8
done
sha256sum "$SPLIT_DIR/normalization_stats.json"
