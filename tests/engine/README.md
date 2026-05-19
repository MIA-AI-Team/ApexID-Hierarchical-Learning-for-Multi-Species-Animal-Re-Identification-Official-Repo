# tests/engine

This folder tests the training and validation engine.

## Main Files

- `test_checkpoint.py`: checkpoint state, save/load, and resume-related behavior.
- `test_early_stopping.py`: best-metric tracking and best-artifact saving.
- `test_label_contracts.py`: classification and metric label ignore-index
  contracts.
- `test_trainer.py`: trainer epoch behavior, metrics, scheduler/checkpoint
  integration, and validation flow.
- `test_validator.py`: validation loss, logits, label handling, and metric
  collection.
- `test_wandb_logger.py`: W&B logger construction and artifact/metric logging
  behavior.

## How It Connects

These tests protect `apexid.engine`, which is used by `scripts/train.py` to run
experiments, save checkpoints, resume training, validate embeddings, and log
metrics.

## Common Commands

Run all engine tests:

```bash
pytest tests/engine
```

Run checkpoint tests:

```bash
pytest tests/engine/test_checkpoint.py
```

Run trainer and validator tests together:

```bash
pytest tests/engine/test_trainer.py tests/engine/test_validator.py
```

## Contributor Notes

- Keep test expectations aligned with metric names logged by the trainer.
- Add resume/checkpoint tests when checkpoint state changes.
- Preserve explicit tests for known and unknown identity label contracts.
