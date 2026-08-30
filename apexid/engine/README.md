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

Open-set validation uses gallery, known, unknown, and mixed loaders. When the
mixed split is exactly `val_known.csv + val_unknown.csv`, `Validator` reuses the
known and unknown embeddings instead of running a separate mixed forward pass.
The evaluator also reuses gallery-match scores for mixed metrics, so full
validation remains exact while avoiding duplicated work.

## Checkpoints and Resume

- `last_checkpoint.pt` is checkpoint format version 2 and is the resumable state.
  It stores the full model, AdamW/other optimizer state, scheduler, completed and
  next epoch, history, latest metrics, process RNG states, early-stopping state,
  and explicit DataLoader-generator states.
- `best_full_model.pt` and `best_backbone.pt` are model-selection artifacts for
  the best value of the configured early-stopping metric. They are not resume
  checkpoints.
- `best_metadata.json` records the monitored metric and best epoch.

Early-stopping resume restores its best metric/score, best epoch, patience
counter, stop flag, and loss compatibility field before the next validation.
The active monitor, direction, patience, and minimum delta must match the saved
configuration; incompatible resumes fail instead of silently changing decisions.

Historical checkpoints without early-stopping state remain loadable, but emit a
`LegacyCheckpointWarning`. Their missing state is not inferred from history, so
model-selection and patience behavior cannot be claimed equivalent to an
uninterrupted run. The training CLI deliberately keeps this warning visible.

The P x K sampler remains deterministic from seed plus epoch. Explicit
DataLoader generators are serialized recursively. Bitwise data trajectories are
still not guaranteed with persistent worker processes, stochastic state held
inside workers, or hardware-dependent CUDA kernels; deterministic single-process
or non-persistent-worker pipelines are covered by resume regression tests.

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
  depends on saved model, optimizer, scheduler, history, metrics, RNG,
  early-stopping, and DataLoader-generator state.
- Keep validation metric keys aligned with config names such as
  `mixed/NormAcc`, `known/mAP`, and the legacy `val_*` aliases.
- Preserve `classification_ignore_index` and `metric_ignore_index` behavior for
  open-set validation.
- Preserve mixed-split reuse only when dataset metadata confirms the mixed split
  order matches known followed by unknown.
