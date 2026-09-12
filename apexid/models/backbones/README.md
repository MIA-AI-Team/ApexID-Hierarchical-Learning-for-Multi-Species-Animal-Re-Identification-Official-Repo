# apexid/models/backbones

This folder contains feature extractors that produce ApexID embeddings.

## Main Files

- `osnet.py`: OSNet architecture variants adapted for configurable input
  channels and embedding dimensions.
- `osnet_pt.py`: pretrained OSNet wrappers and the six-channel wavelet adapter.
- `dinov3_convnext.py`: Hugging Face DINOv3 ConvNeXt wrappers with immutable
  revision and full-file SHA-256 verification support.
- `dinov3_vit.py`: Hugging Face DINOv3 ViT wrappers that return the native CLS
  token (Identity when `embedding_dim` matches hidden size), pin revision and
  SHA-256, and reject spatial sizes that are not multiples of patch 16.
- `convnext.py`: TorchVision ConvNeXt wrappers with ApexID embedding projection,
  random-init and ImageNet-pretrained options.
- `convnextv2.py`: Hugging Face ConvNeXt V2 wrappers using GAP `pooler_output`
  (Identity when `embedding_dim` matches hidden size) with revision and SHA-256
  pins. Default `_pt` repos are ImageNet-22k @ 384.
- `registry.py`: backbone registration and lookup helpers.
- `__init__.py`: exports registry helpers and OSNet, ConvNeXt, ConvNeXt V2, and DINOv3 builders.

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
The active DINOv3 ConvNeXt Small pin uses revision
`421020015aa4266363db803b6049a529999afb1e` and SHA-256
`8182492b73345ceb449ed6be0de486d778a73afb273ba2684de785a71ef1da9d`.
DINOv3 ViT-B/16 LVD-1689M uses revision
`5931719e67bbdb9737e363e781fb0c67687896bc` and SHA-256
`9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b`.
TorchVision ConvNeXt records the installed TorchVision version, weight enum,
resolved URL, cached-file SHA-256, and filename hash prefix.

## Common Commands

Run backbone tests:

```bash
pytest tests/models/test_input_channels.py tests/models/test_convnext.py tests/models/test_convnextv2.py tests/models/test_dinov3_vit.py
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

ConvNeXt V2 aliases (Hugging Face ImageNet-22k @ 384 by default):

```text
convnextv2_tiny
convnextv2_base
convnextv2_large
convnextv2_tiny_pt
convnextv2_base_pt
convnextv2_large_pt
```

Use `AutoModel` pooled features, not `ConvNextV2ForImageClassification`. Set
`embedding_dim` to the backbone hidden size (768 tiny / 1024 base / 1536 large)
for native embeddings. The cluster recipe trains at 440x440 gray3 (CNN, so
spatial size is not locked to the 384 checkpoint). Override `model_name` for
1k-224 checkpoints if needed.

```yaml
model:
  embedding_dim: 1024
  backbone:
    name: convnextv2_base_pt
    kwargs:
      pretrained: true
      revision: 53ec0af6e1bd11bf7eba3e9131d29b69625accf3
      expected_sha256: 70dd4eb0d73cbad0f93236564e093cec20ab76a877d2e4d155512097ebf06fd9
```

DINOv3 ViT aliases:

```text
dinov3_vits16
dinov3_vits16plus
dinov3_vitb16
dinov3_vitl16
dinov3_vits16_pt
dinov3_vits16plus_pt
dinov3_vitb16_pt
dinov3_vitl16_pt
```

The `_pt` aliases load Hugging Face LVD-1689M weights with SDPA attention.
Set `embedding_dim` to the ViT hidden size (384 / 768 / 1024) to keep native
CLS embeddings without an extra Linear projection. Input height and width must
be multiples of 16.

```yaml
model:
  embedding_dim: 768
  backbone:
    name: dinov3_vitb16_pt
    kwargs:
      pretrained: true
      revision: 5931719e67bbdb9737e363e781fb0c67687896bc
      expected_sha256: 9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b
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
