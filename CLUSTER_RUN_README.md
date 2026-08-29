# ApexID Split V2 H100 Runbook

This runbook completes the observation-aware split, perceptual audit,
train-only normalization, and data-pipeline smoke test on the HPI/AISC Slurm
cluster. The split job uses one H100; normalization stays CPU-only. V1 remains
untouched.

## Inputs and outputs

The cluster defaults used by the supplied jobs are environment variables, so
no user home directory is committed to a training YAML:

| Variable | Default in the job | Purpose |
|---|---|---|
| `METADATA` | shared `apexid-dataset/metadata.csv` | authoritative six-column input |
| `DATA_ROOT` | shared `apexid-dataset` | resolves `images/...` paths |
| `OUTPUT_DIR` | `$REPO_ROOT/open_set_splits_v2` | new V2 artifacts |
| `FEATURE_CACHE` | `$REPO_ROOT/split_v2_feature_cache` | validated memory-mapped DINO cache |
| `SPLIT_DIR` | `$REPO_ROOT/open_set_splits_v2` | normalization input/output |
| `CONFIG` | `configs/cluster_home.yaml` | primary config receiving mean/std |

The split job verifies that metadata SHA-256 is exactly
`fd8a921f7ab5566b02fa166bfbbfc948375ee68226b15c29caab8b7fc3ebacf0`.
Override a path at submission time only when the mounted layout differs:

```bash
sbatch --export=ALL,METADATA=/mounted/metadata.csv,DATA_ROOT=/mounted/apexid \
  ~/ApexID/slurm_make_splits.sh
```

## 1. Prepare the checkout

From the login node:

```bash
mkdir -p ~/logs
cd ~/ApexID
git status --short
sha256sum /sc/projects/sci-aisc/apex-id/extracted/apexid-dataset/metadata.csv
bash -n slurm_make_splits.sh slurm_mean_std.sh slurm_train_h100.sh
```

If shell files were transferred through a text-only channel, their deployment
copies decode exactly as follows:

```bash
base64 -d slurm_make_splits.sh.b64.txt > /tmp/slurm_make_splits.sh
base64 -d slurm_mean_std.sh.b64.txt > /tmp/slurm_mean_std.sh
cmp slurm_make_splits.sh /tmp/slurm_make_splits.sh
cmp slurm_mean_std.sh /tmp/slurm_mean_std.sh
```

The scientific pins are in `requirements-split-v2.txt` and `data/split_v2.yaml`:
WildlifeDatasets commit `67380d7951b89f02e48c84a7d3546188b9dbec8a`,
`timm==1.0.27`, DINOv2 revision
`4741e1cafbf45415e77074bb0cb42dba76c8684a`, and checkpoint SHA-256
`0424a5d1b515278cba3c6640ccbeaacc41de59d3a93df0dd5e494285eea2b355`.

## 2. Submit Split V2 on one H100

The default run creates a fresh cache when none exists, rejects a stale or
incomplete cache, performs the V2 and V1 perceptual audits, writes atomically,
then executes validation-only mode.

```bash
SPLIT_JOB=$(sbatch --parsable ~/ApexID/slurm_make_splits.sh)
echo "$SPLIT_JOB"
squeue -j "$SPLIT_JOB"
```

Only when deliberately replacing an existing cache or V2 destination:

```bash
SPLIT_JOB=$(sbatch --parsable \
  --export=ALL,RECOMPUTE_FEATURES=1,OVERWRITE_V2=1 \
  ~/ApexID/slurm_make_splits.sh)
```

These flags are explicit because silent cache reuse and accidental destination
replacement are prohibited.

Monitor:

```bash
tail -f ~/logs/apexid_splits_${SPLIT_JOB}.out
tail -f ~/logs/apexid_splits_${SPLIT_JOB}.err
sacct -j "$SPLIT_JOB" --format=JobID,State,Elapsed,AllocTRES,MaxRSS,ExitCode
```

Successful output must contain the six legacy-schema CSVs plus
`split_manifest.csv`, `split_summary.json`, `near_duplicate_candidates.csv`,
and `v1_near_duplicate_candidates.csv`:

```bash
cd ~/ApexID
python data/make_open_set_splits.py \
  --metadata /sc/projects/sci-aisc/apex-id/extracted/apexid-dataset/metadata.csv \
  --output-dir open_set_splits_v2 \
  --validation-only
python -m json.tool open_set_splits_v2/split_summary.json | less
```

Do not proceed unless the summary reports zero train↔`val_known` exact-content
crossings and zero shared temporal observation groups. Review and explain every
remaining perceptual candidate; perceptual hashes are an audit signal, not an
automatic label.

## 3. Run train-only normalization on CPU

Submit after the split succeeds:

```bash
MEAN_JOB=$(sbatch --parsable --dependency=afterok:${SPLIT_JOB} \
  ~/ApexID/slurm_mean_std.sh)
echo "$MEAN_JOB"
```

The job reads only `open_set_splits_v2/train.csv`, writes
`normalization_stats.json`, and applies the same result to all four 440×440 RGB
training configs:

- `configs/default.yaml`
- `configs/cluster_home.yaml`
- `configs/cluster_pretrained.yaml`
- `configs/cluster_subcenter_dynamic.yaml`

`normalization_stats.json` records the split-manifest hash, training CSV hash,
row count, preprocessing, exact command, code revision, mean, and standard
deviation.

```bash
tail -f ~/logs/apexid_mean_std_${MEAN_JOB}.out
python -m json.tool ~/ApexID/open_set_splits_v2/normalization_stats.json
```

## 4. Point a local cluster config at V2

Split paths are environment-specific. Keep them in a local experiment copy,
not as a committed `/sc/home/<user>` path. Copy the desired training YAML and
set these four keys to the generated directory:

```yaml
data:
  train_path: /sc/home/USER/ApexID/open_set_splits_v2/train.csv
  val_known_path: /sc/home/USER/ApexID/open_set_splits_v2/val_known.csv
  val_unknown_path: /sc/home/USER/ApexID/open_set_splits_v2/val_unknown.csv
  val_path: /sc/home/USER/ApexID/open_set_splits_v2/val.csv
  root_dir: /sc/projects/sci-aisc/apex-id/extracted/apexid-dataset
```

Replace `USER` only in the local copy, for example
`configs/cluster_v2.local.yaml`, and submit with
`CONFIG=configs/cluster_v2.local.yaml`.

## 5. Data-pipeline smoke test and training

Run the smoke test on a compute allocation with the same environment as
training:

```bash
cd ~/ApexID
python scripts/debug_data_pipeline.py --config configs/cluster_v2.local.yaml
```

Then train only after split, normalization, candidate review, and smoke test
have succeeded:

```bash
TRAIN_JOB=$(sbatch --parsable --dependency=afterok:${MEAN_JOB} \
  --export=ALL,CONFIG=configs/cluster_v2.local.yaml \
  ~/ApexID/slurm_train_h100.sh)
echo "$TRAIN_JOB"
```

## 6. Return the completion artifacts

Copy back these files so `SPLIT_V2_REPORT.md` can be finalized with actual
corpus values:

```text
open_set_splits_v2/split_summary.json
open_set_splits_v2/split_manifest.csv
open_set_splits_v2/normalization_stats.json
open_set_splits_v2/near_duplicate_candidates.csv
open_set_splits_v2/v1_near_duplicate_candidates.csv
logs/apexid_splits_<jobid>.out
logs/apexid_splits_<jobid>.err
logs/apexid_mean_std_<jobid>.out
logs/apexid_mean_std_<jobid>.err
```

Also return the four modified config YAMLs. The report is complete only after
the actual V2 counts/hashes, V1/V2 comparison, normalization numbers, smoke-test
result, and candidate dispositions are recorded.
