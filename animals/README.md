# animals

This folder contains the local dataset metadata and image tree used by the
default ApexID experiment configuration. It is the concrete dataset root for
`configs/default.yaml`, where `data.root_dir` points to `animals`.

## Main Files

- `fine_tune_train.csv`: training split used by `data.train_path`.
- `val_all.csv`: validation split used by `data.val_path`; includes `val_type`
  values such as known validation samples.
- `metadata.csv`: broader metadata table containing train and test image rows.
- `sample_submission.csv`: example submission-style mapping from `image_id` to a
  predicted cluster label.
- `images/`: image files grouped by source dataset, split, and identity folder.

The CSV files include more metadata than the model needs. The required columns
for the ApexID dataset loader are:

```text
identity,path
```

`path` values are relative to this folder. For example, the dataset resolves
`images/LynxID2025/train/...jpg` as `animals/images/LynxID2025/train/...jpg`.

## How It Connects

`scripts/train.py` loads this folder through `configs/default.yaml`:

```yaml
data:
  train_path: animals/fine_tune_train.csv
  val_path: animals/val_all.csv
  root_dir: animals
```

The data package standardizes column names, loads images with OpenCV, converts
them to RGB, applies configured transforms, and builds labels for classification
and retrieval metrics.

## Common Commands

Audit CSV columns, label mappings, and one transformed batch:

```bash
python scripts/debug_data_pipeline.py --config configs/default.yaml
```

Run metadata checks without loading a batch:

```bash
python scripts/debug_data_pipeline.py --config configs/default.yaml --skip-batch
```

Calculate normalization statistics from the training split:

```bash
python -m apexid.utils.calculate_mean_std --config configs/default.yaml --split train
```

Write calculated training statistics back into the config:

```bash
python -m apexid.utils.calculate_mean_std --config configs/default.yaml --split train --write-config
```

## Contributor Notes

- Keep identity values globally unique across source datasets. The dataset loader
  rejects identity collisions across source/species columns.
- Keep CSV paths relative to `animals/` when using the default config.
- Do not add README files inside each `images/...` identity folder; those folders
  are raw dataset content, not separate source modules.
- If you change CSV column names, make sure `apexid.data.datasets.ApexID_Dataset`
  can still standardize them or update the config/code together.
