#!/bin/bash
#SBATCH --job-name=apexid-splits
#SBATCH --account=aisc
#SBATCH --partition=aisc-batch
#SBATCH --constraint=ARCH:X86
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=48:00:00
#SBATCH --output=/sc/home/%u/logs/apexid_splits_%j.out
#SBATCH --error=/sc/home/%u/logs/apexid_splits_%j.err

set -euo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: submit this file with sbatch; do not run directly on lx01."
  exit 2
fi

REPO_ROOT="$HOME/ApexID"
METADATA="${METADATA:-/sc/projects/sci-aisc/apex-id/extracted/apexid-dataset/metadata.csv}"
DATA_ROOT="${DATA_ROOT:-/sc/projects/sci-aisc/apex-id/extracted/apexid-dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/open_set_splits_v2}"
FEATURE_CACHE="${FEATURE_CACHE:-$REPO_ROOT/split_v2_feature_cache}"
EXPECTED_METADATA_SHA256="fd8a921f7ab5566b02fa166bfbbfc948375ee68226b15c29caab8b7fc3ebacf0"

WORKDIR="${SLURM_TMPDIR:-/tmp}/apexid_splits_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"

cp -f "$HOME/.local/bin/uv" "$WORKDIR/uv"
chmod +x "$WORKDIR/uv"
UV="$WORKDIR/uv"

VENV="$WORKDIR/venv"
$UV venv --no-project "$VENV"
PY="$VENV/bin/python"

$UV pip install --python "$PY" --torch-backend=cu124 -r "$REPO_ROOT/requirements-split-v2.txt"
$UV pip uninstall --python "$PY" -y opencv-python opencv-python-headless 2>/dev/null || true
$UV pip install --python "$PY" opencv-python-headless==4.13.0.92

cd "$REPO_ROOT"
nvidia-smi

ACTUAL_METADATA_SHA256=$(sha256sum "$METADATA" | cut -d ' ' -f 1)
if [ "$ACTUAL_METADATA_SHA256" != "$EXPECTED_METADATA_SHA256" ]; then
  echo "ERROR: metadata SHA-256 mismatch: $ACTUAL_METADATA_SHA256"
  exit 3
fi

SPLIT_ARGS=(
  --metadata "$METADATA"
  --data-root "$DATA_ROOT"
  --output-dir "$OUTPUT_DIR"
  --feature-cache "$FEATURE_CACHE"
  --baseline-dir "$DATA_ROOT/open_set_splits"
  --device cuda
  --batch-size "${BATCH_SIZE:-16}"
)
if [ "${RECOMPUTE_FEATURES:-0}" = "1" ]; then
  SPLIT_ARGS+=(--recompute-features)
fi
if [ "${NEAR_DUPLICATE_AUDIT:-1}" = "1" ]; then
  SPLIT_ARGS+=(--near-duplicate-audit)
fi
if [ "${OVERWRITE_V2:-0}" = "1" ]; then
  SPLIT_ARGS+=(--overwrite-v2)
fi

$UV run --no-project -p "$PY" python data/make_open_set_splits.py \
  "${SPLIT_ARGS[@]}"

$UV run --no-project -p "$PY" python data/make_open_set_splits.py \
  --metadata "$METADATA" \
  --output-dir "$OUTPUT_DIR" \
  --validation-only

echo "Done. Output files:"
ls -la "$OUTPUT_DIR"
