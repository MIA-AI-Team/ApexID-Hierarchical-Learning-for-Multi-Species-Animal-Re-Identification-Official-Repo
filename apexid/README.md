# apexid

This is the Python package for ApexID. It contains the reusable code used by
training scripts, tests, notebooks, and future inference/evaluation workflows.

## Main Modules

- `data/`: CSV-backed datasets, image transforms, wavelet channel generation,
  mean/std calculation, and P x K batch sampling.
- `models/`: ApexID embedding model, OSNet backbones, trainable heads, and
  factory functions that build components from config.
- `losses/`: objective implementations used by model factories and heads.
- `engine/`: training loop, validation loop, checkpointing, early stopping, and
  experiment logging.
- `evaluation/`: re-identification metrics such as top-k accuracy, mAP, and
  nearest-neighbor micro-F1.
- `utils/`: config loading, device selection, seeding, dataset statistics, LR
  range testing, and embedding UMAP plotting.
- `__init__.py`: public package exports for common model builders.

## How It Connects

`scripts/train.py` is the main orchestrator. It loads a config, builds datasets
from `apexid.data`, constructs model/head/loss components from `apexid.models`,
trains through `apexid.engine.Trainer`, validates with `apexid.engine.Validator`
and `apexid.evaluation`, and writes optional outputs using utility helpers.

## Common Commands

Run the full package test suite:

```bash
pytest tests
```

Train using the package through the script entry point:

```bash
python scripts/train.py --config configs/default.yaml
```

Import top-level builders:

```python
from apexid import ApexID, build_components, build_head, build_loss, build_model
```

## Contributor Notes

- Keep reusable behavior inside `apexid/`; scripts should remain thin wrappers.
- Add or update tests under `tests/` when changing package behavior.
- Keep config keys, README examples, and factory defaults synchronized.
