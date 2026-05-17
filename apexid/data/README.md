# ApexID Data Package

This folder contains the data loading, preprocessing, normalization, and batch
sampling code used for animal re-identification experiments.

The package is built around a simple CSV format:

```csv
identity,path
animal_001,/path/to/image_001.jpg
animal_001,/path/to/image_002.jpg
animal_002,/path/to/image_003.jpg
```

The two required concepts are:

- `identity`: the animal identity used for classification, metric learning, and
  retrieval metrics.
- `path`: the image path loaded from disk with OpenCV.

Several common aliases are accepted and normalized internally. For image paths,
`image_path`, `img_path`, `filepath`, and `file_path` are treated as `path`. For
identities, `id`, `individual_id`, and `label` are treated as `identity`.

## Files

- `datasets/dataset.py`: CSV-backed PyTorch dataset.
- `transforms.py`: train/eval transform builders, wavelet channel construction,
  CLAHE enhancement, and dataset mean/std calculation.
- `samplers/pk_sampler.py`: deterministic P x K batch sampler for metric
  learning.
- `__init__.py`: public exports for the data package.

## Dataset

`ApexID_Dataset` loads rows from a CSV or a pandas DataFrame. Each row is one
image sample.

Input table shape:

```text
(num_samples, num_columns)
```

Required columns after standardization:

```text
identity, path
```

Image loading behavior:

- Images are loaded with `cv2.imread(..., cv2.IMREAD_COLOR)`.
- OpenCV loads images in BGR order, so the dataset converts them to RGB.
- Before transforms, each image has shape:

```text
(height, width, 3)
```

`__getitem__` returns:

```python
image, cls_label, metric_label
```

Where:

- `image` is the transformed image. With the default transform settings, the
  tensor shape is `(3, 440, 440)`.
- `cls_label` is a scalar `torch.long` tensor used by classification or margin
  heads.
- `metric_label` is a scalar `torch.long` tensor used for retrieval and
  verification metrics.

The dataset maintains two label systems:

- `label_to_idx`: classification labels for known training identities.
- `metrics_label_map`: metric labels where every identity should stay distinct,
  including validation identities in open-set evaluation.

The dataset also builds:

```python
identity_to_indices
```

This maps each known training identity to its dataset row indices and is consumed
by the P x K sampler.

## Transforms

The transform builders read their defaults from:

```text
configs/default.yaml
```

Relevant config keys:

```yaml
data:
  image_size: [440, 440]
  wavelet: false
  transforms:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
```

Use these two public functions:

```python
from apexid.data import build_train_transforms, build_eval_transforms

train_tfms = build_train_transforms()
eval_tfms = build_eval_transforms()
```

### Training Pipeline

`build_train_transforms` creates an Albumentations pipeline that:

1. Resizes the image to `data.image_size`.
2. Applies CLAHE local contrast enhancement.
3. Applies light geometric augmentation with `A.Affine`.
4. Converts the image to float values in `[0, 1]`.
5. Optionally appends wavelet detail channels.
6. Normalizes using configured mean/std.
7. Converts the image to a PyTorch tensor with `ToTensorV2`.

Input shape:

```text
(height, width, 3)
```

Output shape when `wavelet: false`:

```text
(3, configured_height, configured_width)
```

Output shape when `wavelet: true`:

```text
(6, configured_height, configured_width)
```

### Evaluation Pipeline

`build_eval_transforms` uses deterministic preprocessing only:

1. Resize.
2. CLAHE local contrast enhancement.
3. Float conversion or wavelet channel construction.
4. Normalize.
5. Convert to tensor.

There is no random augmentation in the eval pipeline.

## Dataset Mean And Std

Use `calculate_dataset_mean_std` to compute normalization statistics from your
own training data:

```python
from apexid.data import calculate_dataset_mean_std

mean, std = calculate_dataset_mean_std(
    data_csv="train.csv",
    root_dir="/path/to/images",
)
```

The function:

- Reads the CSV or DataFrame.
- Resolves image paths using `root_dir` when needed.
- Loads images with OpenCV and converts them to RGB.
- Resizes images to `data.image_size`.
- Applies the same CLAHE enhancement used by train/eval transforms.
- Applies wavelet channel construction if `data.wavelet` is enabled.
- Computes per-channel mean and standard deviation over all pixels.

Return values:

```python
mean, std
```

When `wavelet: false`, both lists have length `3`.

When `wavelet: true`, both lists have length `6`.

After computing these values, put them into `configs/default.yaml` under:

```yaml
data:
  transforms:
    mean: [...]
    std: [...]
```

## CLAHE

Both train and eval transforms apply CLAHE after resize and before normalization.
The current fixed parameters are:

```python
clip_limit = 4.0
tile_grid_size = (5, 5)
```

CLAHE is deterministic in this pipeline (`p=1.0`). Because normalization happens
after CLAHE, `calculate_dataset_mean_std` also applies CLAHE before computing
statistics. This keeps the configured mean/std aligned with the actual images
seen by the model.

## Wavelet Mode

The config key:

```yaml
data:
  wavelet: true
```

enables extra frequency-detail channels.

The current implementation computes one-level grayscale Haar detail bands:

- `LH`
- `HL`
- `HH`

These are appended to the RGB image:

```text
RGB + LH + HL + HH
```

So the transform output has 6 channels instead of 3.

Important: when `wavelet: true`, `data.transforms.mean` and
`data.transforms.std` must each contain 6 values. Use
`calculate_dataset_mean_std` with `wavelet: true` to generate the correct
statistics.

## P x K Sampler

`PKBatchSampler` builds batches for re-identification and metric learning.

Example:

```python
from torch.utils.data import DataLoader
from apexid.data.samplers import PKBatchSampler

sampler = PKBatchSampler(
    dataset=train_dataset,
    num_identities=8,
    num_instances=4,
    seed=42,
)

loader = DataLoader(
    train_dataset,
    batch_sampler=sampler,
    num_workers=4,
)
```

Each yielded batch contains:

```text
P * K samples
```

For the example above:

```text
8 identities * 4 images per identity = 32 samples
```

Sampler behavior:

- Identities are sampled without replacement inside a batch.
- Samples for an identity are sampled without replacement when enough images are
  available.
- If an identity has fewer than `K` images, sampling for that identity uses
  replacement.
- The sampler is deterministic from the configured seed.
- Calling `set_epoch(epoch)` changes batches in a reproducible way across epochs.

## Reproducibility

The data package uses the shared seed utilities from:

```python
apexid.utils.seed
```

The default seed is `42`. The P x K sampler uses NumPy's generator with:

```python
seed + epoch
```

This makes batches repeatable for the same seed and epoch while still changing
between epochs.

## Public Imports

Common imports:

```python
from apexid.data import (
    DEFAULT_CONFIG_PATH,
    build_train_transforms,
    build_eval_transforms,
    calculate_dataset_mean_std,
)

from apexid.data.datasets import ApexID_Dataset
from apexid.data.samplers import PKBatchSampler
```

## Typical Usage

```python
from apexid.data import build_train_transforms
from apexid.data.datasets import ApexID_Dataset
from apexid.data.samplers import PKBatchSampler

train_transforms = build_train_transforms()

train_dataset = ApexID_Dataset(
    data_csv="train.csv",
    root_dir="/path/to/images",
    transform=train_transforms,
)

batch_sampler = PKBatchSampler(
    dataset=train_dataset,
    num_identities=8,
    num_instances=4,
    seed=42,
)
```

This produces batches suitable for training ApexID with metric-learning or
margin-based classification heads.
