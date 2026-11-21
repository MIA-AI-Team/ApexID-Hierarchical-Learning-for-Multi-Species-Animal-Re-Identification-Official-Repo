# Dataset preparation

## Use the frozen training partitions

The [`open_set_splits_v2`](../open_set_splits_v2) directory contains the observation-aware partitions from the submission. Set `APEXID_DATA_ROOT` to the root of the curated image tree. Each manifest path is relative to that root and starts with `images/`.

| File | Images | Dataset-scoped identities |
| --- | ---: | ---: |
| `train.csv` | 90,921 | 11,677 |
| `val_known.csv` | 28,070 | 10,768 |
| `val_unknown.csv` | 13,542 | 1,671 |

`val.csv` concatenates the two validation partitions in that order. The primary partitions cover 19 source datasets and contain 96,435 distinct observation groups. `val_unknown_extra.csv` records held-out overflow groups left outside the primary validation partition. `unlabeled_ood.csv` records singleton and unlabeled rows excluded from training and validation.

The six training CSV columns are `identity`, `path`, `date`, `orientation`, `species`, and `dataset`, in that order. The paper defines identities by `(dataset, identity)`. The released identity strings are already unique across datasets. If you prepare a new corpus, scope repeated local labels to their dataset before training. The `species` field stores the shared animal category, including broad categories such as `whale`.

`split_manifest.csv` records split membership, image hashes, observation IDs, assignment methods, and the DINOv2 revision. `split_summary.json` records the configuration, generation provenance, and counts. Its `output_hashes` describe the current checked-in files in LF form. `generation_output_hashes` records the earlier run before category harmonization and LF checkout. Validate the release without image files from the repository root.

```bash
python -m pip install -r requirements-split-v2.txt
python data/make_open_set_splits.py --validation-only --output-dir open_set_splits_v2
```

## Rebuild split membership from curated images

Start from a curated `metadata.csv` with the same six columns and the matching segmented RGB image tree. The audited generation input contained 158,293 rows. Its SHA-256 is `fd8a921f7ab5566b02fa166bfbbfc948375ee68226b15c29caab8b7fc3ebacf0`.

```bash
python data/make_open_set_splits.py --metadata metadata.csv --data-root ApexID_Dataset --output-dir ApexID_Dataset/open_set_splits_v2 --feature-cache ApexID_Dataset/split_v2_feature_cache
python data/make_open_set_splits.py --metadata metadata.csv --data-root ApexID_Dataset --output-dir ApexID_Dataset/open_set_splits_v2 --validation-only
```

[`split_v2.yaml`](split_v2.yaml) pins the seed, similarity thresholds, and model revision. For known identities, it assigns whole observations using capture dates, verified filename or session groups, or within-identity DINOv2 groups with exact-copy links. The builder selects held-out identities per source and records every eligible and excluded row. Use `--verify-images` to decode the referenced images during validation. Use `--near-duplicate-audit` to export perceptual-hash candidates after splitting.

The released training CSVs already contain the harmonized animal categories. For new splits, `scripts/prepare_bhca_species_metadata.py` can fill missing HappyWhale `whale` or `dolphin` labels from matching identities and image names in the frozen manifest. Pass `--happywhale-species` and the original source CSV to validate or extend coverage.

The paper's upstream preparation includes dataset-specific conversions, within-identity MegaDescriptor duplicate filtering, SAM 3 segmentation, and V-Rush and MobileSAM manual review. The split builder takes the resulting curated metadata and images as inputs. The original images, segmentation scripts, manual annotations, and curator logs are absent from this checkout. Obtain images under their source terms. See the evaluation notebooks for the separate external closed-set and open-set preparations.
