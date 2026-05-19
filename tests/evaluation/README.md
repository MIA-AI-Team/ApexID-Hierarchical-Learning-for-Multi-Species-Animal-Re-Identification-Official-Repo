# tests/evaluation

This folder tests re-identification metrics.

## Main Files

- `test_reid_metrics.py`: top-k accuracy, mAP, micro-F1, nearest-neighbor
  predictions, ignore-index behavior, and metric edge cases.

## How It Connects

These tests protect `apexid.evaluation.reid_metrics`, which is called by
`apexid.engine.Validator` during validation and feeds metrics such as `val_mAP`
into training history, logs, and early stopping.

## Common Commands

Run evaluation tests:

```bash
pytest tests/evaluation
```

Run the metric test file directly:

```bash
pytest tests/evaluation/test_reid_metrics.py
```

## Contributor Notes

- Add edge-case tests when metric filtering, ranking, or `nan` handling changes.
- Keep metric output keys stable unless the trainer/config/tests are updated
  together.
