# Image encoders

| Module | Provider | RGB size | Embedding dimensions |
| --- | --- | --- | --- |
| `convnext.py` | TorchVision ConvNeXt-Small ImageNet-1K | 440 × 440 | 2048 |
| `miewid.py` | MIEW-ID v3 from Conservation X Labs | 440 × 440 | 2152 |
| `megadescriptor.py` | MegaDescriptor-L-384 from BVRA | 384 × 384 | 1536 |

The `*_pt` aliases load pretrained weights by default. The evaluation encoder loader disables provider weight retrieval when restoring a supplied ApexID checkpoint.
