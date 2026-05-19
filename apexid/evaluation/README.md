# apexid/evaluation

This package contains metric functions for re-identification validation.

## Main Files

- `reid_metrics.py`: embedding normalization, leave-one-out top-k accuracy, mAP,
  nearest-neighbor predictions, micro-F1, and `compute_reid_metrics`.
- `__init__.py`: exports the metric helpers.

## How It Connects

`apexid.engine.Validator` collects validation embeddings and metric labels, then
calls `compute_reid_metrics`. The resulting values are added to epoch metrics
and can drive early stopping through config keys such as `val_mAP`.

Metrics operate on embeddings shaped:

```text
(num_samples, embedding_dim)
```

and labels shaped:

```text
(num_samples,)
```

Embeddings are L2-normalized before cosine-similarity ranking. Optional
`ignore_index` values are filtered before retrieval metrics are computed.

## Common Commands

Run metric tests:

```bash
pytest tests/evaluation
```

Use metrics directly:

```python
from apexid.evaluation import compute_reid_metrics

metrics = compute_reid_metrics(embeddings, labels, ignore_index=None)
```

## Contributor Notes

- Return `nan` rather than misleading scores when a metric has no valid queries.
- Keep metric names stable because training history, early stopping, and W&B
  logging use those keys.
- Add tests for edge cases such as single-sample inputs, ignored labels, and
  identities without positive gallery matches.
