# apexid/losses

This package contains loss implementations used by ApexID model factories and
trainable heads.

## Main Files

- `dcmip_loss.py`: `DCMIPLoss`, a contrastive objective with class-proxy and
  momentum-instance memory components.
- `sphereface2.py`: `SphereFace2`, the base SphereFace2 objective/head
  implementation used by SphereFace2 model heads.
- `__init__.py`: exports `DCMIPLoss` and `SphereFace2`.

## How It Connects

`apexid.models.model_factory.build_loss` creates external objectives such as
`cross_entropy` or `dcmip`. SphereFace2 heads compute their own loss internally,
so `build_components` rejects a non-null external loss when a SphereFace2 head is
selected.

## Common Commands

Run loss/head tests that cover SphereFace2 behavior:

```bash
pytest tests/models/test_sphereface2_focal.py
```

Import losses directly:

```python
from apexid.losses import DCMIPLoss, SphereFace2
```

## Contributor Notes

- Keep loss tensor shapes compatible with the trainer's
  `_compute_loss_and_logits` helper.
- When a loss owns trainable state or memory buffers, document how it should be
  initialized and saved.
- Add focused tests for numerical edge cases and label validation.
