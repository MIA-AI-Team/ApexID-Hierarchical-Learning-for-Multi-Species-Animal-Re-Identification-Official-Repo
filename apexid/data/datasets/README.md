# apexid/data/datasets

This folder contains dataset classes that convert CSV metadata into PyTorch
samples for ApexID.

## Main Files

- `dataset.py`: defines `ApexID_Dataset`.
- `__init__.py`: exports `ApexID_Dataset`.

## How It Connects

`ApexID_Dataset` accepts a CSV path or pandas DataFrame, standardizes common
column aliases, validates image path and identity columns, builds deterministic
label maps, loads images with OpenCV, converts BGR to RGB, applies optional
Albumentations transforms, and returns image/classification/metric labels.

The class builds two label systems:

- `label_to_idx`: known training identities for classification or margin heads.
- `metrics_label_map`: retrieval identities used for validation metrics,
  including open-set validation identities.

It also builds `identity_to_indices`, which is consumed by
`apexid.data.samplers.PKBatchSampler`.

## Common Commands

Run dataset tests:

```bash
pytest tests/data/test_dataset.py
```

Run the full data suite:

```bash
pytest tests/data
```

Use the dataset directly:

```python
from apexid.data import build_train_transforms
from apexid.data.datasets import ApexID_Dataset

dataset = ApexID_Dataset(
    data_csv="animals/fine_tune_train.csv",
    root_dir="animals",
    transform=build_train_transforms(),
)
image, cls_label, metric_label = dataset[0]
```

## Contributor Notes

- Keep identity labels globally unique across source/species metadata.
- Preserve the distinction between classification labels and metric labels.
- When adding accepted CSV aliases, update both tests and folder documentation.
