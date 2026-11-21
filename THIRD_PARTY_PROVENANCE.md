# Third-party provenance

The [Apache-2.0 license](LICENSE) covers the ApexID source code. External datasets, pretrained weights, and their provider code are subject to their respective terms.

## Pretrained encoders

| Artifact | Upstream revision |
| --- | --- |
| ConvNeXt-Small | TorchVision 0.25.0 `ConvNeXt_Small_Weights.IMAGENET1K_V1` |
| MIEW-ID v3 | [conservationxlabs/miewid-msv3](https://huggingface.co/conservationxlabs/miewid-msv3/tree/4f1d7f2b521149e5fe34bb85f377248ce9971a7d) at `4f1d7f2b521149e5fe34bb85f377248ce9971a7d` |
| MegaDescriptor-L | [BVRA/MegaDescriptor-L-384](https://huggingface.co/BVRA/MegaDescriptor-L-384/tree/33b3c6f4ee0c386a4126cc3dcd23843920613fa1) at `33b3c6f4ee0c386a4126cc3dcd23843920613fa1` |

The MIEW-ID v3 loader checks the SHA-256 of `model.safetensors` against `adff92b39678f37eb74861c6399a741639a8907ec2382738e903d6120727b348`. The MegaDescriptor-L loader uses timm's Swin checkpoint conversion. Training records the resolved upstream artifact information in its run manifest.

## Data preparation

The observation-aware splitter uses frozen DINOv2 embeddings for its similarity fallback. The [data guide](data/README.md) and frozen split metadata document the model, thresholds, and source catalogues. Some thresholds were taken from WildlifeDatasets v1.0.9, commit `67380d7951b89f02e48c84a7d3546188b9dbec8a`. The splitter implements the grouping locally.

The paper's prepared training and evaluation images and private jaguar labels are obtained separately under their sources' terms.
