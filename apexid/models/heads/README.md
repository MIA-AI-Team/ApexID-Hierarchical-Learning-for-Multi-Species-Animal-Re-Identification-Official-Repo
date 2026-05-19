# apexid/models/heads

This folder contains trainable classification and metric-learning heads that sit
on top of ApexID embeddings.

## Main Files

- `arcface.py`: `ArcFaceHead`, additive angular-margin logits with
  `predict_logits` for validation.
- `dynamic_arcface.py`: `DynArcFaceHead`, ArcFace-style logits with dynamic
  class margins.
- `sphereface2_head.py`: `SphereFace2Head`, SphereFace2 loss/head wrapper with
  validation logits.
- `sphereface2_focal.py`: `SphereFace2FocalHead`, focal-weighted SphereFace2
  variant.
- `__init__.py`: exports public head classes.

## How It Connects

`apexid.models.model_factory.build_head` selects these classes using the config
`head.name`. Heads receive normalized embeddings from `ApexID` and either return
class logits for an external loss or compute their own scalar loss.

`Validator` uses `predict_logits` when a head provides it, so validation accuracy
is computed from non-margin or evaluation-safe logits.

## Common Commands

Run head tests:

```bash
pytest tests/models
```

Instantiate a head directly:

```python
from apexid.models.heads import ArcFaceHead

head = ArcFaceHead(in_features=128, out_features=50, s=64.0, m=0.5)
```

## Contributor Notes

- Add new head aliases to `HEAD_REGISTRY` in `model_factory.py`.
- If a head computes its own loss, update `INTERNAL_LOSS_HEADS`.
- Provide `predict_logits` when training-time logits include margins that should
  not be used directly for validation metrics.
