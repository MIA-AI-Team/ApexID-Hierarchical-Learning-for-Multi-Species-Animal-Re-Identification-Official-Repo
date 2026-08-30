# apexid/models

This package contains the embedding model, backbone registry, trainable heads,
and factory functions used to build training components from config.

## Main Files

- `apexid_model.py`: `ApexID`, a wrapper around a registered backbone that
  returns L2-normalized embeddings.
- `model_factory.py`: `build_model`, `build_head`, `build_loss`, and
  `build_components` for config-driven construction.
- `backbones/`: OSNet and TorchVision ConvNeXt implementations plus the
  backbone registry.
- `heads/`: ArcFace, dynamic ArcFace, SphereFace2, and SphereFace2 focal heads.
- `__init__.py`: public model exports.

## How It Connects

`scripts/train.py` calls `build_components(cfg, num_classes)`. The model factory
reads `model`, `head`, `loss`, and `data.wavelet` config sections, builds the
embedding model, chooses input channels, constructs the selected head, and
returns an external loss when the head does not compute its own objective.
Backbone names are selected through `model.backbone.name`; ConvNeXt aliases
include `convnext_tiny`, `convnext_small`, `convnext_base`, `convnext_large`,
and pretrained-default `_pt` variants.

Pretrained wrappers pin provider revisions and verify weight SHA-256 where the
artifact is external to TorchVision. TorchVision weight enums record their
resolved URL and cached checksum. The resulting model provenance is included in
training/evaluation run manifests; see `backbones/README.md` and
`THIRD_PARTY_PROVENANCE.md`.

## Common Commands

Run model tests:

```bash
pytest tests/models
```

Build components from config:

```python
from apexid.utils.config import load_config
from apexid.models import build_components

cfg = load_config(config_path="configs/default.yaml")
model, head, criterion = build_components(cfg, num_classes=100)
```

Select ConvNeXt in YAML without changing the surrounding model/head contract:

```yaml
model:
  name: apexid
  embedding_dim: 512
  input_channels: 3
  backbone:
    name: convnext_tiny_pt
    kwargs:
      pretrained: true
```

## Contributor Notes

- Register new backbones in `backbones/registry.py`.
- Add new heads to `HEAD_REGISTRY` in `model_factory.py`.
- Keep `model.input_channels` compatible with `data.wavelet` and transform
  output.
- Preserve normalized embedding output from `ApexID.forward`.
