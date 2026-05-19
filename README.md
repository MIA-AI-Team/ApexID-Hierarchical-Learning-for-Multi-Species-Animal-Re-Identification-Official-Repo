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
- `animals/`: local CSV metadata and image tree used by the default config.
- `data/`: placeholder for external or temporary datasets that should stay
  separate from the committed `animals/` layout.
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
  train_path: animals/fine_tune_train.csv
  val_path: animals/val_all.csv
  root_dir: animals
```

The train and validation CSV files must contain at least:

```text
identity,path
```

Start training:

```bash
python scripts/train.py --config configs/default.yaml
```

Training builds datasets, transforms, a P x K sampler, an ApexID model, the
configured head/loss, optimizer, scheduler, validator, checkpoint writer, early
stopping, optional W&B logging, and optional UMAP embedding plots.

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
  log_artifacts: best
```

Tips:

- Set `run_name` for a human-readable run name. Otherwise the experiment name is
  used.
- Add strings to `tags`, for example `["osnet", "sphereface2"]`.
- Use `mode: offline` to test logging locally without uploading. Local W&B files
  are written under `wandb/`, which is ignored by Git.
- `log_artifacts: best` uploads the best model files saved by early stopping,
  not every epoch checkpoint.

The trainer logs epoch-level metrics such as `train_loss`, `train_acc`,
`val_loss`, `val_acc`, `val_top1`, `val_top5`, `val_mAP`, `val_micro_f1`, `lr`,
validation sample counts, and early-stopping state.

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
