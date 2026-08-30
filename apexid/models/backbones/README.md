# apexid/models/backbones

This folder contains feature extractors that produce ApexID embeddings.

## Main Files

- `osnet.py`: OSNet architecture variants adapted for configurable input
  channels and embedding dimensions.
- `osnet_pt.py`: pretrained OSNet wrappers and the six-channel wavelet adapter.
- `dinov3_convnext.py`: Hugging Face DINOv3 ConvNeXt wrappers with immutable
  revision and full-file SHA-256 verification support.
- `convnext.py`: TorchVision ConvNeXt wrappers with ApexID embedding projection,
  random-init and ImageNet-pretrained options.
- `registry.py`: backbone registration and lookup helpers.
- `__init__.py`: exports registry helpers, OSNet builders, and ConvNeXt builders.

## How It Connects

`ApexID` calls `build_backbone(backbone_name, **kwargs)` during model
construction. `configs/default.yaml` selects the default backbone:

```yaml
model:
  backbone:
    name: osnet_x1_0
```

Registered names include `osnet_x1_0`, `osnet_x0_75`, `osnet_x0_5`,
`osnet_x0_25`, and `osnet_ibn_x1_0`.

TorchVision ConvNeXt backbones are also registered:

```text
convnext_tiny
convnext_small
convnext_base
convnext_large
convnext_tiny_pt
convnext_small_pt
convnext_base_pt
convnext_large_pt
```

The `_pt` aliases default to ImageNet pretrained weights. Pretrained ConvNeXt
supports RGB input directly and six-channel wavelet input through the same
`WaveletStem` adapter used by pretrained OSNet. Random-init ConvNeXt can replace
the first convolution for any positive `input_channels`.

Pretrained artifacts are auditable rather than moving downloads. OSNet uses
`kaiyangzhou/osnet` revision
`a5c5cc037c24235cda3b21085b93ad77c9616224` and verifies per-variant SHA-256.
The active DINOv3 Small config uses revision
`421020015aa4266363db803b6049a529999afb1e` and SHA-256
`8182492b73345ceb449ed6be0de486d778a73afb273ba2684de785a71ef1da9d`.
TorchVision ConvNeXt records the installed TorchVision version, weight enum,
resolved URL, cached-file SHA-256, and filename hash prefix.

## Common Commands

Run backbone tests:

```bash
pytest tests/models/test_input_channels.py tests/models/test_convnext.py
```

List available backbones in Python:

```python
from apexid.models.backbones import available_backbones

print(available_backbones())
```

Use ConvNeXt from config:

```yaml
model:
  backbone:
    name: convnext_tiny_pt
    kwargs:
      pretrained: true
```

## Contributor Notes

- New backbones must accept `embedding_dim` and `input_channels` when used by
  `ApexID`.
- Register new builders with `register_backbone`.
- Keep output shape `(batch_size, embedding_dim)` so heads and metrics stay
  compatible.
- Do not L2-normalize inside a backbone; `ApexID.forward` owns embedding
  normalization.
- Pretrained fine-tuning profiles are configured in `configs/default.yaml`
  under `finetune` and applied automatically for `*_pt` backbones. See
  [`configs/README.md`](../../configs/README.md#pretrained-fine-tuning).
