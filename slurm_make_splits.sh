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
#SBATCH --output=apexid_splits_%j.out
#SBATCH --error=apexid_splits_%j.err

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
METADATA="${METADATA:-$REPO_ROOT/metadata.csv}"
DATA_ROOT="${APEXID_DATA_ROOT:-$REPO_ROOT/ApexID_Dataset}"
OUTPUT_DIR="${APEXID_SPLIT_ROOT:-$DATA_ROOT/open_set_splits_v2}"
FEATURE_CACHE="${FEATURE_CACHE:-$OUTPUT_DIR.feature_cache}"
EXPECTED_METADATA_SHA256="fd8a921f7ab5566b02fa166bfbbfc948375ee68226b15c29caab8b7fc3ebacf0"

WORKDIR="${SLURM_TMPDIR:-/tmp}/apexid_splits_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"

UV="${UV_BIN:-$(command -v uv || true)}"
if [ -z "$UV" ]; then
  echo "ERROR: uv is not on PATH; set UV_BIN to its executable."
  exit 3
fi

VENV="$WORKDIR/venv"
$UV venv --no-project "$VENV"
PY="$VENV/bin/python"

$UV pip install --python "$PY" --torch-backend="${APEXID_TORCH_BACKEND:-auto}" -r "$REPO_ROOT/requirements-split-v2.txt"
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
