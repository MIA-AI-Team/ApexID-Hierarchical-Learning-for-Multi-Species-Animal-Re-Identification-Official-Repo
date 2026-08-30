# ApexID Portable Cluster Runbook

The supplied Slurm jobs use portable working-directory logs and derive the
repository root from the script location. Site-specific account, partition, and
constraint directives may be overridden with `sbatch` options. No committed
configuration contains a user home directory.

## Required variables

Set paths for the mounted prepared corpus and results:

```bash
export APEXID_DATA_ROOT=/mounted/ApexID_Dataset
export APEXID_SPLIT_ROOT="$APEXID_DATA_ROOT/open_set_splits_v2"
export APEXID_OUTPUT_ROOT=/mounted/apexid-results
export UV_BIN=/path/to/uv
```

Active YAMLs retain the V1-relative split fallback when
`APEXID_SPLIT_ROOT` is unset. Final-paper runs must set it explicitly to the
frozen V2 directory. Prepared images, metadata, split manifests, normalization
statistics, and model weights are external inputs and are not bundled here.

Use submission-time paths for logs when desired:

```bash
sbatch --output=/mounted/logs/train_%j.out \
  --error=/mounted/logs/train_%j.err slurm_train_h100.sh
```

## Split V2 on one H100

The splitter uses its dedicated `requirements-split-v2.txt` lock. The default
authoritative input is repository-root `metadata.csv`; override `METADATA` only
for a different mount. It verifies SHA-256
`fd8a921f7ab5566b02fa166bfbbfc948375ee68226b15c29caab8b7fc3ebacf0`.

```bash
SPLIT_JOB=$(sbatch --parsable \
  --export=ALL,METADATA=/mounted/metadata.csv \
  slurm_make_splits.sh)
```

Only set `RECOMPUTE_FEATURES=1` or `OVERWRITE_V2=1` when deliberately replacing
those artifacts. The job writes the six legacy CSVs, `split_manifest.csv`,
`split_summary.json`, and optional perceptual-audit CSVs, then runs
validation-only mode. V1 remains untouched.

Do not proceed unless validation reports zero train↔`val_known` exact-content
crossings and zero shared temporal observation groups. Perceptual candidates
must be reviewed and explained.

## Train-only normalization on CPU

```bash
MEAN_JOB=$(sbatch --parsable --dependency=afterok:${SPLIT_JOB} \
  slurm_mean_std.sh)
```

This job installs the canonical lock with CPU PyTorch 2.10.0/TorchVision 0.25.0,
reads only `train.csv`, writes `normalization_stats.json`, and updates the four
440×440 RGB configs. The statistics record train/manifest hashes, row count,
preprocessing, command, code revision, mean, and standard deviation.

## Offline and data-pipeline smoke tests

The first command needs no data, downloads, or private labels:

```bash
python scripts/repro_smoke.py --output /tmp/apexid-repro-smoke.json
python scripts/debug_data_pipeline.py --config configs/cluster_home.yaml
```

## H100 training handoff

The CPU lock is verified. The exact CUDA/H100 combination is not yet verified.
On the target node, set `APEXID_TORCH_BACKEND` to the site's supported uv
backend (or leave `auto`), install the same lock through the Slurm job, and
return `environment.json`, `run_manifest.json`, `nvidia-smi`, and smoke/test
logs before claiming CUDA reproducibility.

```bash
TRAIN_JOB=$(sbatch --parsable --dependency=afterok:${MEAN_JOB} \
  --export=ALL,CONFIG=configs/cluster_home.yaml \
  slurm_train_h100.sh)
```

Training writes `config_resolved.yaml`, `environment.json`, and
`run_manifest.json`. Add `--require-clean-git` after committing the intended
scientific changes when producing a release run. Persistent worker-internal RNG
and hardware-dependent CUDA kernels may still prevent bitwise equivalence.

## Deployment-copy verification

```bash
for script in slurm_make_splits.sh slurm_mean_std.sh slurm_train_h100.sh; do
  base64 -d "$script.b64.txt" > "/tmp/$script"
  cmp "$script" "/tmp/$script"
done
```

Return Split V2 summaries/audits, normalization statistics, Slurm logs, and the
training environment/manifest. This implementation does not itself execute the
private-corpus H100 jobs.
