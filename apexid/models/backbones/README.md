# apexid/models/backbones

This folder contains feature extractors that produce ApexID embeddings.

## Main Files

- `osnet.py`: OSNet architecture variants adapted for configurable input
  channels and embedding dimensions.
- `registry.py`: backbone registration and lookup helpers.
- `__init__.py`: exports registry helpers and OSNet builders.

## How It Connects

`ApexID` calls `build_backbone(backbone_name, **kwargs)` during model
construction. `configs/default.yaml` selects the default backbone:

```yaml
model:
  backbone:
    name: osnet_x0_25
```

Registered names include `osnet_x1_0`, `osnet_x0_75`, `osnet_x0_5`,
`osnet_x0_25`, and `osnet_ibn_x1_0`.

## Common Commands

Run input-channel tests:

```bash
pytest tests/models/test_input_channels.py
```

List available backbones in Python:

```python
from apexid.models.backbones import available_backbones

print(available_backbones())
```

## Contributor Notes

- New backbones must accept `embedding_dim` and `input_channels` when used by
  `ApexID`.
- Register new builders with `register_backbone`.
- Keep output shape `(batch_size, embedding_dim)` so heads and metrics stay
  compatible.
