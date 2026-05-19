# apexid/utils

This package contains shared utilities used by scripts, data processing,
training, and visualization.

## Main Files

- `config.py`: YAML config loading and default config path.
- `device.py`: CUDA/CPU device selection and device-info helpers.
- `seed.py`: reproducibility helpers for Python, NumPy, PyTorch, workers, and
  generators.
- `calculate_mean_std.py`: CLI and helper for dataset normalization statistics.
- `lr_range_test.py`: learning-rate range test and artifact writing.
- `embedding_umap.py`: UMAP embedding plots and metadata JSON generation.
- `__init__.py`: public utility exports.

## How It Connects

Scripts use `load_config`, training uses seeding/device helpers, transforms use
config loading, the trainer can run LR range testing and embedding plotting, and
the root README commands expose the mean/std CLI directly.

## Common Commands

Run utility tests:

```bash
pytest tests/utils
```

Calculate training mean/std:

```bash
python -m apexid.utils.calculate_mean_std --config configs/default.yaml --split train
```

Write calculated statistics into the config:

```bash
python -m apexid.utils.calculate_mean_std --config configs/default.yaml --split train --write-config
```

Import common utilities:

```python
from apexid.utils.config import load_config
from apexid.utils.device import get_device
from apexid.utils.seed import seed_everything
```

## Contributor Notes

- Utility code should stay dependency-light unless the feature requires an
  optional package such as UMAP or matplotlib.
- Keep CLI defaults aligned with `configs/default.yaml`.
- When adding generated artifacts, write them under configured output folders,
  not inside source package directories.
