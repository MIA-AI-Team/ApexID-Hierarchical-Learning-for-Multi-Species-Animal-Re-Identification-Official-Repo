# apexid/engine

This package owns the training and validation runtime for ApexID experiments.

## Main Files

- `trainer.py`: `Trainer`, the epoch loop that handles forward/backward passes,
  validation cadence, scheduler steps, checkpointing, early stopping, logging,
  and optional embedding plots.
- `validator.py`: `Validator`, validation loss/accuracy calculation and
  embedding collection for retrieval metrics.
- `checkpoint.py`: checkpoint state creation, latest-checkpoint save/load,
  backbone-only weight save/load, and RNG state restoration.
- `early_stopping.py`: best-model tracking and full/backbone artifact saving.
- `loggers.py`: experiment logger protocol and no-op logger.
- `wandb_logger.py`: W&B-backed logger and logger factory.
- `__init__.py`: public exports for the engine package.

## How It Connects

`scripts/train.py` builds a backbone, head, loss, optimizer, scheduler,
DataLoaders, and then passes them to `Trainer`. The trainer calls `Validator` at
the configured interval, feeds validation embeddings into `apexid.evaluation`,
writes checkpoints under the configured output directory, and optionally logs
metrics/artifacts through W&B.

Validation retrieval metrics use the configured query chunk size to avoid
allocating the full all-pairs similarity matrix at once.

## Common Commands

Run engine tests:

```bash
pytest tests/engine
```

Train through the engine:

```bash
python scripts/train.py --config configs/default.yaml
```

Import core engine helpers:

```python
from apexid.engine import Trainer, Validator, EarlyStopping
from apexid.engine import save_training_checkpoint, load_training_checkpoint
```

## Contributor Notes

- Keep checkpoint formats backward-compatible when possible; resume training
  depends on saved model, optimizer, scheduler, history, metrics, and RNG state.
- Keep validation metric keys aligned with config names such as
  `mixed/NormAcc`, `known/mAP`, and the legacy `val_*` aliases.
- Preserve `classification_ignore_index` and `metric_ignore_index` behavior for
  open-set validation.
