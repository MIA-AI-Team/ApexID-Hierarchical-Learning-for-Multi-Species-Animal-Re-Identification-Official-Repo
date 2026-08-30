# tests/evaluation

This folder tests re-identification and open-set validation metrics.

## Main Files

- `test_reid_metrics.py`: top-k accuracy, mAP, micro-F1, nearest-neighbor
  predictions, ignore-index behavior, and metric edge cases.
- `test_open_set_metrics.py`: known retrieval, unknown clustering, rejection,
  mixed open-set metrics, DIR/OSCR helpers, and diagnostics cadence.
- `test_benchmark_metrics.py`: standard item-averaged B³ analytical cases,
  randomized reference comparisons, input validation, and threshold objectives.
- `test_evaluate_apexid_model.py`: sealed benchmark inference, retrieval,
  clustering diagnostics, and final-evaluation integrity.

## How It Connects

These tests protect `apexid.evaluation`, which is called by
`apexid.engine.Validator` during validation and feeds metrics such as
`known/mAP`, `mixed/NormAcc`, and legacy `val_*` aliases into training history,
logs, and early stopping.

## Common Commands

Run evaluation tests:

```bash
pytest tests/evaluation
```

Run the metric test file directly:

```bash
pytest tests/evaluation/test_benchmark_metrics.py
```

## Contributor Notes

- Add edge-case tests when metric filtering, ranking, or `nan` handling changes.
- Keep metric output keys stable unless the trainer/config/tests are updated
  together.
