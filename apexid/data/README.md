# apexid/data

This package owns data loading, image preprocessing, normalization statistics,
wavelet channel construction, and batch sampling for ApexID experiments.

## Main Files

- `datasets/dataset.py`: `ApexID_Dataset`, a CSV/DataFrame-backed PyTorch
  dataset.
- `transforms.py`: Albumentations train/eval transform builders, path
  resolution, and dataset mean/std calculation.
- `DWT.py`: Haar wavelet detail-channel utilities.
- `samplers/pk_sampler.py`: deterministic P x K batch sampler for metric
  learning and re-identification batches.
- `__init__.py`: public exports for transform helpers and config defaults.

## How It Connects

`scripts/train.py` builds train and validation datasets from the paths in
`configs/default.yaml`. The train dataset provides `identity_to_indices` to
`PKBatchSampler`, which creates batches with multiple identities and multiple
images per identity. The validation dataset preserves metric labels for
retrieval metrics in `apexid.evaluation`.

The required standardized CSV columns are:

```text
identity,path
```

Accepted aliases include `image_path`, `img_path`, `filepath`, `file_path`,
`id`, `individual_id`, and `label`.

`ApexID_Dataset.__getitem__` returns:

```python
image, cls_label, metric_label
```

- `image`: transformed tensor, usually `(3, H, W)` or `(6, H, W)` in wavelet
  mode.
- `cls_label`: classification label for trainable heads, with unknown validation
  identities optionally mapped to the configured ignore index.
- `metric_label`: identity label used by retrieval metrics, where validation
  identities remain distinct.

## Transforms

Training transforms resize, apply light affine augmentation, convert to float,
optionally append wavelet channels, normalize, and convert to a tensor.
Evaluation transforms use deterministic preprocessing only.

Relevant config keys:

```yaml
data:
  image_size: [440, 440]
  wavelet: false
  transforms:
    mean: [...]
    std: [...]
```

When `data.wavelet: true`, the transform appends grayscale Haar `LH`, `HL`, and
`HH` detail bands to RGB, producing six input channels. Mean/std must also have
six values.

## Common Commands

Run data tests:

```bash
pytest tests/data
```

Debug the configured data flow:

```bash
python scripts/debug_data_pipeline.py --config configs/default.yaml
```

Calculate training mean/std:

```bash
python -m apexid.utils.calculate_mean_std --config configs/default.yaml --split train
```

Import the public data helpers:

```python
from apexid.data import build_train_transforms, build_eval_transforms
from apexid.data import calculate_dataset_mean_std
from apexid.data.datasets import ApexID_Dataset
from apexid.data.samplers import PKBatchSampler
```

## Contributor Notes

- Keep transform behavior aligned with mean/std calculation. If preprocessing
  changes before normalization, update `calculate_dataset_mean_std`.
- Keep train/eval transforms shape-compatible with `model.input_channels`.
- Preserve deterministic label mapping and sampler behavior; tests rely on it.
- Add new dataset-specific behavior under `datasets/` and new sampling behavior
  under `samplers/`.
