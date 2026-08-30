# Third-Party Provenance

ApexID source is Apache-2.0. That license does not grant rights to external
datasets, pretrained weights, model-provider code, or their trademarks. Users
must obtain and comply with those materials separately.

## Dependency environment

`requirements.txt` is the readable input. `requirements-lock.txt` is the exact
Python 3.13.5 training/evaluation environment validated on CPU; the splitter
keeps its separate `requirements-split-v2.txt`. Important upstream licenses
include:

| Component | Version/source | Upstream license/provenance |
|---|---|---|
| PyTorch / TorchVision | 2.10.0 / 0.25.0 | BSD-3-Clause |
| timm | 1.0.27 | Apache-2.0 |
| Transformers / Hugging Face Hub | 5.5.0 / 1.13.0 | Apache-2.0 |
| OpenCV headless | 4.13.0.92 | Apache-2.0 |
| NumPy / pandas / SciPy / scikit-learn | locked versions | BSD-family licenses |
| Albumentations | 2.0.8 | MIT |
| PyYAML | 6.0.2 | MIT |
| Matplotlib | 3.10.0 | Matplotlib/PSF-compatible license |
| W&B SDK | 0.29.0 | MIT |
| WildlifeDatasets | commit `67380d7951b89f02e48c84a7d3546188b9dbec8a` (v1.0.9) | AGPL-3.0-or-later |

WildlifeDatasets is scientific provenance for the frozen Split V2 thresholds;
the splitter does not import its code. Its commit remains in the readable input
and dedicated Split V2 environment. The canonical training lock intentionally
does not expand its unused downloader graph (`gdown`, Kaggle, GUI OpenCV, and
dataset clients).

This table is a release aid, not legal advice. The installed distributions'
license files and metadata remain authoritative.

## Pretrained artifacts

| Artifact | Provider and immutable revision | Verified weight SHA-256 |
|---|---|---|
| DINOv3 ConvNeXt Small LVD-1689M | [`facebook/dinov3-convnext-small-pretrain-lvd1689m`](https://huggingface.co/api/models/facebook/dinov3-convnext-small-pretrain-lvd1689m?blobs=true), `421020015aa4266363db803b6049a529999afb1e` | `8182492b73345ceb449ed6be0de486d778a73afb273ba2684de785a71ef1da9d` |
| OSNet x1.0 | [`kaiyangzhou/osnet`](https://huggingface.co/api/models/kaiyangzhou/osnet?blobs=true), `a5c5cc037c24235cda3b21085b93ad77c9616224` | `fe2d63f9157c28a4a8d8ca29bec12d5b2988ac0346d712025789ea9174968e79` |
| OSNet x0.75 | same | `37add23d2ebc7a22584203d6c22976fdda0f7f29c7b7d79257ac22ae20aec809` |
| OSNet x0.5 | same | `f10ea9e26ee55b1a65cee3c6e88aa1ae105fe981af3e37692fa68a72fc64bf8e` |
| OSNet x0.25 | same | `f54941a66bad4ddd07f2907f498c810ce639ce7a1abeaf2a151f8da118d84693` |
| OSNet IBN x1.0 | same | `4e6fbaa3f5e31170aae76372a9ebbe36551e4011a4ed88b139d50bc98729e23a` |

DINOv3 weights are governed by the provider's separate DINOv3 license, not
ApexID's Apache-2.0 license. Inspect the model repository license at the frozen
revision before redistribution or commercial use. OSNet weights likewise
remain third-party artifacts.

TorchVision ConvNeXt uses the installed enum's `IMAGENET1K_V1` artifact. Each
run records TorchVision version, enum, resolved URL, filename hash prefix, and
the full cached-file SHA-256 when present.

## External data

The prepared ApexID corpus, benchmark private ground truth, original wildlife
datasets, and Split V2 image artifacts are not distributed under ApexID's
source license. Run manifests hash mounted inputs but do not confer data rights.
