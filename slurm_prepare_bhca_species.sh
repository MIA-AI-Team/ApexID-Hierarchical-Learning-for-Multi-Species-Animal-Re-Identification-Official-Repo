#!/bin/bash
#SBATCH --job-name=apexid-bhca-species
#SBATCH --account=aisc
#SBATCH --partition=pot-hpi-aisc-batch
#SBATCH --constraint=ARCH:X86
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=apexid_bhca_species_%j.out
#SBATCH --error=apexid_bhca_species_%j.err

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
if [ ! -f "$REPO_ROOT/scripts/prepare_bhca_species_metadata.py" ]; then
  echo "ERROR: prepare_bhca_species_metadata.py missing under $REPO_ROOT/scripts"
  exit 3
fi

SPLIT_DIR="${APEXID_SPLIT_ROOT:-$REPO_ROOT/open_set_splits_v2}"
REPORT_DIR="${BHCA_REPORT_DIR:-$REPO_ROOT/outputs/bhca_species_preflight}"
# Set BHCA_IN_PLACE=0 to write *.bhca.csv next to each source instead of replacing.
BHCA_IN_PLACE="${BHCA_IN_PLACE:-1}"

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "REPO_ROOT=$REPO_ROOT"
echo "SPLIT_DIR=$SPLIT_DIR"
echo "REPORT_DIR=$REPORT_DIR"
echo "BHCA_IN_PLACE=$BHCA_IN_PLACE"

WORKDIR="${SLURM_TMPDIR:-/tmp}/apexid_bhca_species_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR" "$REPORT_DIR"

# Copy uv into the job workdir — ~/.local/bin/uv is often not executable on lx01.
UV_SRC="${UV_BIN:-$HOME/.local/bin/uv}"
if [ ! -x "$UV_SRC" ] && [ ! -f "$UV_SRC" ]; then
  echo "ERROR: uv not found at $UV_SRC (set UV_BIN)."
  exit 3
fi
cp -f "$UV_SRC" "$WORKDIR/uv"
chmod +x "$WORKDIR/uv"
UV="$WORKDIR/uv"

VENV="$WORKDIR/venv"
$UV venv --no-project "$VENV"
PY="$VENV/bin/python"

# CSV enrichment only needs pandas/numpy/PyYAML (no torch).
$UV pip install --python "$PY" pandas==2.2.3 numpy==2.1.3 PyYAML==6.0.2

cd "$REPO_ROOT"

CSVS=(train.csv val.csv val_known.csv val_unknown.csv)
for csv_name in "${CSVS[@]}"; do
  input="$SPLIT_DIR/$csv_name"
  if [ ! -f "$input" ]; then
    echo "ERROR: missing split CSV: $input"
    exit 3
  fi
  stem="${csv_name%.csv}"
  report="$REPORT_DIR/${stem}_species_preflight.json"

  if [ "$BHCA_IN_PLACE" = "1" ]; then
    echo "Enriching in-place: $input"
    $UV run --no-project -p "$PY" python scripts/prepare_bhca_species_metadata.py \
      --input "$input" \
      --in-place \
      --report "$report"
  else
    output="$SPLIT_DIR/${stem}_bhca.csv"
    echo "Enriching $input -> $output"
    $UV run --no-project -p "$PY" python scripts/prepare_bhca_species_metadata.py \
      --input "$input" \
      --output "$output" \
      --report "$report"
  fi
done

echo "BHCA species enrichment finished."
ls -lah "$REPORT_DIR"
