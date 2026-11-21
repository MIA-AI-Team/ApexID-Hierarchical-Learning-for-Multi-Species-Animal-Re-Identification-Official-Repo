# ApexID

Code for *ApexID: Hierarchical Learning for Multi-Species Animal Re-Identification*. ApexID learns a single embedding for animal re-identification by combining global identity discrimination with balanced animal-category supervision.

## Experiment matrix

The submission compares **ConvNeXt-Small**, **MIEW-ID v3**, and **MegaDescriptor-L** with **ArcFace**, **Dynamic ArcFace**, **SphereFace2**, and **ApexID Hierarchical Loss**. Evaluation also includes the two original animal Re-ID checkpoints. The ConvNeXt-Small category-weight sweep uses the same encoder and training objective as the main model.

See [experiment presets](configs/README.md) for the matrix, the supplementary sweep, and checkpoint compatibility.

## Setup

Use Python 3.12 in an isolated environment. From the repository root, install the CPU build for development and tests.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0 torchvision==0.25.0
python -m pip install -r requirements.txt -c requirements-lock.txt
python -m pip check
```

On PowerShell, activate with `.\.venv\Scripts\Activate.ps1`. For GPU training, install a CUDA build of the same PyTorch and TorchVision versions for your system before installing the remaining requirements.

## Data and training

The [data guide](data/README.md) documents the prepared foreground corpus, the frozen training and validation splits, and the observation-aware partitioner. The checked-in V2 metadata contains 90,921 training images, 28,070 known-identity validation images, and 13,542 held-out-identity validation images.

Mount the matching foreground images and set `APEXID_DATA_ROOT` to the directory containing the manifest-relative paths. The default is `ApexID_Dataset/`. Presets use the checked-in `open_set_splits_v2/` by default and save runs under `outputs/`. Set `APEXID_SPLIT_ROOT` or `APEXID_OUTPUT_ROOT` to override those locations.

```bash
python scripts/debug_data_pipeline.py --config configs/default.yaml --skip-batch
python scripts/train.py --config configs/default.yaml
```

The default preset trains ConvNeXt-Small with ApexID Hierarchical Loss. The first pretrained load requires access to the upstream weights or a local cache. Use each checkpoint's matching encoder and preprocessing for evaluation.

## External evaluation

Follow the [evaluation guide](docs/evaluation.md) for preparing manifests, extracting normalized embeddings, running the five-seed observation-disjoint retrieval protocol, and running the dataset-adapted open-set protocol. The guide covers the available supplementary analyses and the required inputs.

The prepared image corpus, private jaguar images and labels, and trained ApexID weights are separate inputs. See [third-party provenance](THIRD_PARTY_PROVENANCE.md) for upstream artifact references.

## Checks

```bash
python -m pytest -q
```
