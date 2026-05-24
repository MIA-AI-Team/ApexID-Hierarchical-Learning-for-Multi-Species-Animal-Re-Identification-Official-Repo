# ApexID

ApexID is a PyTorch project for animal re-identification. It trains an
embedding model, applies metric/classification heads such as ArcFace and
SphereFace2, validates with retrieval metrics, and saves checkpoints,
experiment history, and optional embedding-space plots.

Run commands from the repository root:

```bash
cd /home/seifelkerdany/PC/MIA/ApexID
```

## Project Map

- `apexid/`: Python package with datasets, transforms, models, losses, training
  engine, evaluation metrics, and utilities.
- `configs/`: YAML experiment configuration. `configs/default.yaml` is the
  default entry point used by scripts and tests.
- `scripts/`: command-line entry points for training and data-pipeline checks.
- `ApexID_Dataset/`: ignored local dataset root used by the default config.
- `data/`: local dataset preparation notes and scripts.
- `notebooks/`: exploratory notebook examples.
- `tests/`: pytest suite covering data, model, engine, metric, script, and
  utility behavior.
- `outputs/` and `wandb/`: generated runtime artifacts. These are not source
  modules and are not documented per run folder.

Each important folder has its own `README.md` explaining what it owns, how it
connects to the rest of the project, and which commands are useful for that
folder.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

The requirements include PyTorch, TorchVision, timm, OpenCV, Albumentations,
scikit-learn, UMAP, matplotlib, W&B, and WildlifeDatasets packages.

## Train

The default configuration expects these dataset paths:

```yaml
data:
  train_path: ApexID_Dataset/open_set_splits/train.csv
  val_known_path: ApexID_Dataset/open_set_splits/val_known.csv
  val_unknown_path: ApexID_Dataset/open_set_splits/val_unknown.csv
  val_path: ApexID_Dataset/open_set_splits/val.csv
  root_dir: ApexID_Dataset
```

The train and validation CSV files must contain at least:

```text
identity,path
```

For the default ApexID dataset, CSV image paths such as `images/...` are
resolved relative to `root_dir: ApexID_Dataset`.

Start training:

```bash
python scripts/train.py --config configs/default.yaml
```

Training builds datasets, transforms, a P x K sampler, an ApexID model, the
configured head/loss, optimizer, scheduler, validator, checkpoint writer, early
stopping, optional W&B logging, and optional UMAP embedding plots.

## Open-Set Validation

The default training config evaluates the full open-set validation splits every
epoch:

- `val_known.csv`: known identities held out from training images.
- `val_unknown.csv`: identities that do not appear in `train.csv`.
- `val.csv`: the mixed open-set split, normally
  `val_known.csv + val_unknown.csv`.

When `val.csv` is exactly the known split followed by the unknown split, the
validator reuses the known and unknown embeddings instead of running the mixed
loader again. It also reuses gallery-match scores for mixed metrics. This keeps
full validation exact while avoiding duplicated work.

Exact metrics can still be expensive on large galleries because `known/mAP`,
CMC/top-k retrieval, rejection metrics, threshold selection, and clustering all
depend on gallery or validation-set size. For faster approximate validation, set
positive `evaluation.sample_limits` values in `configs/default.yaml`; leave them
as `null` to keep full-split validation.

## Validate The Data Pipeline

Before a long run, inspect the CSV-to-label flow and one DataLoader batch:

```bash
python scripts/debug_data_pipeline.py --config configs/default.yaml
```

Skip batch loading when you only want metadata and label-map checks:

```bash
python scripts/debug_data_pipeline.py --config configs/default.yaml --skip-batch
```

## Calculate Mean And Std

Calculate normalization statistics for the configured training CSV:

```bash
python -m apexid.utils.calculate_mean_std --config configs/default.yaml --split train
```

Calculate statistics for validation:

```bash
python -m apexid.utils.calculate_mean_std --config configs/default.yaml --split val
```

Write training statistics back into `configs/default.yaml`:

```bash
python -m apexid.utils.calculate_mean_std --config configs/default.yaml --split train --write-config
```

Use an explicit CSV instead of the configured split:

```bash
python -m apexid.utils.calculate_mean_std \
  --config configs/default.yaml \
  --csv animals/fine_tune_train.csv \
  --root-dir animals
```

## Track Experiments With Weights & Biases

Log in once on the machine:

```bash
wandb login
```

Enable tracking in `configs/default.yaml`:

```yaml
tracking:
  enabled: true
  provider: wandb
  project: ApexID
  entity: MIA-Robotics
  run_name: null
  group: null
  tags: []
  mode: online
  log_artifacts: metadata
  block_on_artifact_upload: false
  artifact_finish_timeout: 300
```

Tips:

- Set `run_name` for a human-readable run name. Otherwise the experiment name is
  used.
- Add strings to `tags`, for example `["osnet", "sphereface2"]`.
- Use `mode: offline` to test logging locally without uploading. Local W&B files
  are written under `wandb/`, which is ignored by Git.
- `log_artifacts: none` logs metrics only and skips all W&B file uploads. This is
  the recommended setting on slow networks such as Kaggle.
- Use `log_artifacts: metadata` to upload small files such as best-run JSON
  metadata and UMAP plots when embedding plotting is enabled.
- Use `log_artifacts: best` when you also want the best `.pt` weights uploaded to
  W&B.
- `block_on_artifact_upload: false` keeps training moving while W&B uploads run
  in the background. W&B may still wait for pending uploads during `finish()`.
- If logging appears stuck for a long time after validation, the usual cause is
  uploading large checkpoint artifacts over a slow connection, not scalar metric
  logging.

The trainer logs epoch-level metrics such as `train_loss`, `val_loss`,
`val_top1`, `val_top5`, `val_mAP`, `val_micro_f1`, `lr`,
open-set metrics such as `known/mAP`, `unknown/ARI`, `rejection/AUROC`,
`mixed/NormAcc`, diagnostics cadence state, validation reuse flags, validation
sample counts when sampling is enabled, and early-stopping state.

## Test

Run the full suite:

```bash
pytest
```

Run focused suites:

```bash
pytest tests/data
pytest tests/engine
pytest tests/models
pytest tests/scripts
pytest tests/utils
pytest tests/test_default_config.py
```

## Contributor Notes

- Keep configuration-driven behavior in sync with `configs/default.yaml`.
- Keep README command examples runnable from the repository root.
- Do not document generated folders per run or per image identity; document the
  source folder that owns the behavior instead.
- When adding a module, add or update the nearest folder README and include a
  targeted pytest command if tests exist.
