#!/bin/bash
#SBATCH --job-name=apexid-splits
#SBATCH --account=aisc
#SBATCH --partition=aisc-batch
#SBATCH --constraint=ARCH:X86
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/sc/home/%u/logs/apexid_splits_%j.out
#SBATCH --error=/sc/home/%u/logs/apexid_splits_%j.err

set -euo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: submit this file with sbatch; do not run directly on lx01."
  exit 2
fi

REPO_ROOT="$HOME/ApexID"
DATA_SOURCE="/sc/projects/sci-aisc/apex-id/extracted/apexid-dataset/metadata.csv"
OUTPUT_DIR="$HOME/ApexID/open_set_splits"

WORKDIR="${SLURM_TMPDIR:-/tmp}/apexid_splits_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"

cp -f "$HOME/.local/bin/uv" "$WORKDIR/uv"
chmod +x "$WORKDIR/uv"
UV="$WORKDIR/uv"

VENV="$WORKDIR/venv"
$UV venv --no-project "$VENV"
PY="$VENV/bin/python"

$UV pip install --python "$PY" pillow

cd "$REPO_ROOT"

$UV run --no-project -p "$PY" python data/make_open_set_splits.py \
  --source "$DATA_SOURCE" \
  --output-dir "$OUTPUT_DIR" \
  --skip-image-check

echo "Done. Output files:"
ls -la "$OUTPUT_DIR"
