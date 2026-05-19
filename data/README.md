# data

This folder is reserved for external, temporary, or locally prepared datasets
that should remain separate from the committed `animals/` sample/data tree.

## Main Files

This folder currently has no required source files. It exists as a conventional
place for local data work that should not become part of the Python package.

## How It Connects

ApexID does not automatically read from `data/`. To use files placed here, point
`configs/default.yaml` or a custom config at them:

```yaml
data:
  train_path: data/my_train.csv
  val_path: data/my_val.csv
  root_dir: data
```

The dataset loader still expects CSV metadata with image paths and identities.
At minimum, the standardized columns must be:

```text
identity,path
```

Common accepted aliases include `image_path`, `img_path`, `filepath`,
`file_path`, `id`, `individual_id`, and `label`.

## Common Commands

Train against a config that points into this folder:

```bash
python scripts/train.py --config configs/default.yaml
```

Calculate mean/std for an explicit CSV under this folder:

```bash
python -m apexid.utils.calculate_mean_std \
  --config configs/default.yaml \
  --csv data/my_train.csv \
  --root-dir data
```

Debug a config after changing dataset paths:

```bash
python scripts/debug_data_pipeline.py --config configs/default.yaml
```

## Contributor Notes

- Do not place Python package code in this folder; package code belongs under
  `apexid/`.
- Large local datasets should generally stay untracked unless intentionally
  committed.
- Prefer documenting reusable dataset conventions here and dataset-specific
  details in the nearest dataset root README.
