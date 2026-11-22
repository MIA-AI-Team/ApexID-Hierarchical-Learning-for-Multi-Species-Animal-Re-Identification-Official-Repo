# Third-party provenance

The [MIT License](LICENSE) covers the ApexID source code and the ConvNeXt-Small ApexID weights. External datasets, upstream pretrained weights, and their provider code are subject to their respective terms. The fine-tuned MegaDescriptor-L weights keep the upstream CC BY-NC 4.0 license. The upstream MIEW-ID v3 checkpoint declares no license, so its fine-tuned weights carry whatever terms Conservation X Labs and Wild Me set.

## Pretrained encoders

| Artifact | Upstream revision |
| --- | --- |
| ConvNeXt-Small | TorchVision 0.25.0 `ConvNeXt_Small_Weights.IMAGENET1K_V1` |
| MIEW-ID v3 | [conservationxlabs/miewid-msv3](https://huggingface.co/conservationxlabs/miewid-msv3/tree/4f1d7f2b521149e5fe34bb85f377248ce9971a7d) at `4f1d7f2b521149e5fe34bb85f377248ce9971a7d` |
| MegaDescriptor-L | [BVRA/MegaDescriptor-L-384](https://huggingface.co/BVRA/MegaDescriptor-L-384/tree/33b3c6f4ee0c386a4126cc3dcd23843920613fa1) at `33b3c6f4ee0c386a4126cc3dcd23843920613fa1` |

The MIEW-ID v3 loader checks the SHA-256 of `model.safetensors` against `adff92b39678f37eb74861c6399a741639a8907ec2382738e903d6120727b348`. The MegaDescriptor-L loader uses timm's Swin checkpoint conversion. Training records the resolved upstream artifact information in its run manifest.

## Data preparation

The observation-aware splitter uses frozen DINOv2 embeddings for its similarity fallback. The [data guide](data/README.md) and frozen split metadata document the model, thresholds, and source catalogues. Some thresholds were taken from WildlifeDatasets v1.0.9 (AGPL-3.0-or-later), commit `67380d7951b89f02e48c84a7d3546188b9dbec8a`. Only the threshold values are reused; the splitter implements the grouping locally and contains no WildlifeDatasets code. The DINOv2 model card and pretrained configuration declare different licenses (Apache-2.0 and CC-BY-NC-4.0); the splitter only runs the model and does not redistribute it.

## Datasets

The [ApexID Dataset card](https://huggingface.co/datasets/Made-In-Alexandria-AI/ApexID-Dataset) lists every training source with its license and original location. Segmented images are redistributed only where the source license allows it; the rest must be obtained from the original authors. The private jaguar evaluation images and labels are not distributed.
