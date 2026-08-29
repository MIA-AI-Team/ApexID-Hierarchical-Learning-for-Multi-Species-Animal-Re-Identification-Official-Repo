# ApexID Animal Re-Identification Dataset

This project contains a cleaned multi-dataset animal re-identification metadata table and an open-set train/validation split designed for animal ReID model training.

The final training-ready master metadata file is:

```text
metadata.csv
```

The authoritative input is the repository-root `metadata.csv`. The audited
corpus has 158,293 rows and SHA-256
`fd8a921f7ab5566b02fa166bfbbfc948375ee68226b15c29caab8b7fc3ebacf0`.
`ApexID_Dataset/metadata.csv` is currently byte-identical, but Split V2 defaults
to the root file so there is one reproducibility anchor.

## Observation-Aware Split V2

Split V2 is generated into a new directory and does not modify the legacy V1
files:

```text
ApexID_Dataset/open_set_splits_v2/
├── train.csv
├── val_known.csv
├── val_unknown.csv
├── val.csv
├── val_unknown_extra.csv
├── unlabeled_ood.csv
├── split_manifest.csv
├── split_summary.json
├── near_duplicate_candidates.csv       # optional
└── normalization_stats.json            # added by the CPU normalization job
```

The six training CSVs retain the exact six-column schema below, and `val.csv`
is required to equal `val_known.csv` followed by `val_unknown.csv`. The manifest
adds source indices, canonical dataset-scoped identities, strict parsed dates,
image SHA-256 values, assignment methods, stable observation/group IDs, seed,
and frozen feature provenance.

The assignment hierarchy is:

1. Hash every image. Same-identity exact duplicates are hard must-links;
   identical bytes with different canonical `(dataset, identity)` labels abort
   the run with a conflict report.
2. Preserve dataset-wise unknown-identity selection at seed 42 and ratio 0.125.
   Cap each unknown identity near 100 images without splitting a hard group;
   overflow goes to `val_unknown_extra`.
3. For fully dated known identities with at least two dates, keep same-date
   observations together and select the closest chronological validation
   suffix while retaining earlier training data.
4. For identities fully covered by verified explicit-source parsers, select a
   deterministic whole-group subset closest to the 0.225 image target.
5. Send partial, missing, malformed, single-date, and otherwise unresolved
   identities through DINOv2 similarity fallback. All multi-image connected
   components stay in training, following WildlifeReID-10k Algorithm 1; only
   singleton observation components may enter `val_known`. An identity with no
   independent validation observation remains train-only.

Dates are accepted only as `%Y-%m-%d` or `%d-%m-%Y`. Blank dates remain missing;
all other nonblank values are malformed. The authoritative metadata contains
11,857 ISO rows, 39,347 Czech-format rows, 107,089 missing rows, and zero
malformed rows at the audited revision.

Verified explicit-source parsers:

| Dataset | Stable source evidence |
|---|---|
| CTai | `img-id` before the object/prepared index |
| Giraffes | original image basename before crop index and prepared hash |
| IPanda50 | `vNNN` video or explicit timestamp session |
| NDD20 | original numeric filename prefix |
| NyalaData | numeric capture ID shared by left/right and duplicate copies |

Scientific defaults live in `data/split_v2.yaml`, with exact runtime package
versions in `requirements-split-v2.txt`. Similarity uses frozen
`timm/vit_large_patch14_dinov2.lvd142m` features at the pinned Hugging Face
revision and checkpoint hash. Images are RGB, resized bilinearly to 518×518,
converted to float32 `[0,1]`, normalized with ImageNet mean/std, and L2
normalized to 1,024 dimensions. The cache is memory-mappable and is rejected if
its ordered paths, image hashes, metadata hash, model/checkpoint,
preprocessing, shape/dtype, or artifact hashes differ.

Canonical local run:

```bash
python data/make_open_set_splits.py \
  --metadata metadata.csv \
  --data-root ApexID_Dataset \
  --output-dir ApexID_Dataset/open_set_splits_v2 \
  --feature-cache ApexID_Dataset/split_v2_feature_cache \
  --near-duplicate-audit

python data/make_open_set_splits.py \
  --metadata metadata.csv \
  --output-dir ApexID_Dataset/open_set_splits_v2 \
  --validation-only
```

An occupied destination is refused. `--overwrite-v2` authorizes an atomic
replacement of that exact V2 directory; it never clears arbitrary CSVs and
never touches V1. `--source` remains an alias for `--metadata`.

The H100 run, actual V1/V2 comparison, and train-only normalization handoff are
tracked in `SPLIT_V2_REPORT.md` and `CLUSTER_RUN_README.md`.

## Legacy Split V1 Reference

The existing V1 open-set files remain:

```text
open_set_splits/train.csv
open_set_splits/val_known.csv
open_set_splits/val_unknown.csv
open_set_splits/val_unknown_extra.csv
open_set_splits/unlabeled_ood.csv
open_set_splits/val.csv
```

## Final Schema

All final metadata and split CSVs use this schema:

| Column | Meaning |
|---|---|
| `identity` | Individual animal identity label. |
| `path` | Relative image path from the configured dataset root. With `root_dir: ApexID_Dataset`, values such as `images/...` resolve under `ApexID_Dataset/images/...`. |
| `date` | Capture date when available. Mixed source quality: most ISO dates are `YYYY-MM-DD`, while CzechLynx contains non-ISO `DD-MM-YYYY` values. |
| `orientation` | View/orientation label when available. CzechLynx also contains annotation-like JSON values in this column inherited from source metadata. |
| `species` | Species label when available. Some upstream datasets have blank species. |
| `dataset` | Source dataset name. |

## Processing Summary

The dataset was built through these steps:

1. Combined the Jaguar dataset train/test manifests into one Jaguar manifest.
2. Converted Jaguar rows into the shared metadata schema:
   - `identity = ground_truth`
   - `path = images/Jaguar_data/relative_path`
   - `species = jaguar`
   - `dataset = Jaguar_data`
   - unknown fields such as date and orientation were left blank.
3. Appended the normalized Jaguar rows to the existing multi-dataset metadata.
4. Removed unused columns `split` and `cluster_id`.
5. Removed rows whose image path did not exist.
6. Removed rows whose image existed but could not be opened as a valid image.
   - This removed `661` zero-byte/unreadable Jaguar images.
7. Saved the final readable master metadata as `metadata.csv`.
8. Generated open-set ReID splits with `make_open_set_splits.py`.

The previous special identity `new_whale` from `humpback-whale-identification` is not treated as a real individual whale identity. It is excluded from supervised training and main validation, and is kept only in `unlabeled_ood.csv`.

## Legacy V1 Open-Set Split Policy

The split script uses deterministic seed `42` and dataset-scoped identity keys: `(dataset, identity)`.

Split behavior:

- `train.csv`: supervised identity training set.
- `val_known.csv`: held-out images from identities that also appear in train.
- `val_unknown.csv`: balanced unknown-identity validation set with no identity overlap with train.
- `val_unknown_extra.csv`: extra unknown rows that were capped out of balanced validation. This keeps useful data available without letting large identities dominate the main validation metric.
- `val.csv`: `val_known.csv + val_unknown.csv`.

Balancing rules:

- Singleton identities are excluded from supervised training and main validation.
- `new_whale` is excluded from supervised training and main validation.
- About `22.5%` of images from train-seen identities are held out into `val_known`, only when at least `2` train images and `2` validation images can be kept for that identity.
- About `12.5%` of eligible non-singleton identities per dataset are held out into `val_unknown`.
- Unknown identities are capped at `100` images each in `val_unknown`.
- Overflow rows from capped unknown validation identities are stored in `val_unknown_extra.csv`.
- Singleton identities and all `new_whale` rows are stored in `unlabeled_ood.csv`.

Do not use the following command expecting the old random policy: the script
now creates Split V2. V1 is retained as a frozen comparison baseline.

```bash
python data/make_open_set_splits.py
```

## Legacy V1 Split Integrity

The generated split files passed these checks:

| Check | Result |
|---|---:|
| Train and val_unknown identity overlap | `0` |
| Train and val_unknown_extra identity overlap | `0` |
| val_known identities missing from train | `0` |
| val_known and val_unknown identity overlap | `0` |
| val_unknown_extra identities missing from val_unknown | `0` |
| `val.csv == val_known.csv + val_unknown.csv` | `true` |
| `new_whale` rows in train | `0` |
| `new_whale` rows in val_known | `0` |
| `new_whale` rows in val_unknown | `0` |
| `new_whale` rows in val_unknown_extra | `0` |
| `new_whale` rows in unlabeled_ood | `9664` |
| Component rows accounted for | `158293` |

## Main Dataset Statistics

### `metadata.csv`

This is the final cleaned, readable master metadata file.

| Metric | Value |
|---|---:|
| Rows | `158293` |
| Columns | `identity,path,date,orientation,species,dataset` |
| File size | `18.14 MB` |
| Datasets | `19` |
| Dataset-scoped identities | `26468` |
| Unique image paths | `158293` |
| Missing image paths | `0` |
| Duplicate image paths | `0` |
| Zero-byte files | `0` |
| Total referenced image size | `15.398 GiB` |
| Median file size | `36351 bytes` |
| P90 file size | `201894 bytes` |
| P99 file size | `756891 bytes` |
| Max file size | `17235663 bytes` |

Missing values:

| Column | Missing |
|---|---:|
| `identity` | `0` |
| `path` | `0` |
| `date` | `107089` |
| `orientation` | `130035` |
| `species` | `77782` |
| `dataset` | `0` |

Identity distribution:

| Metric | Value |
|---|---:|
| Mean images per identity | `5.981` |
| Median images per identity | `2` |
| P90 images per identity | `8` |
| P99 images per identity | `68` |
| Max images per identity | `9664` |
| Singleton identities | `13119` |

Dataset counts:

| Dataset | Rows |
|---|---:|
| HappyWhale | `51033` |
| CzechLynx | `39760` |
| humpback-whale-identification | `25361` |
| WhaleSharkID | `6047` |
| IPanda50 | `5499` |
| LeopardID2022 | `5316` |
| GiraffeZebraID | `5273` |
| BalearicLizard | `4618` |
| CTai | `3707` |
| HyenaID2022 | `2428` |
| NDD20 | `2122` |
| SealID | `1668` |
| NyalaData | `1497` |
| SalamanderID2025 | `1388` |
| Jaguar_data | `1178` |
| Giraffes | `906` |
| ReunionTurtles | `237` |
| AmvrakikosTurtles | `141` |
| ZakynthosTurtles | `114` |

Species counts:

| Species | Rows |
|---|---:|
| blank | `77782` |
| lynx | `39760` |
| whaleshark | `6047` |
| panda | `5499` |
| leopard | `5316` |
| zebra | `4850` |
| lizard | `4618` |
| chimpanzee | `3707` |
| hyena | `2428` |
| dolphin | `2122` |
| seal | `1668` |
| nyala | `1497` |
| giraffe | `1329` |
| jaguar | `1178` |
| sea turtle | `492` |

Top identities:

| Dataset | Identity | Rows |
|---|---|---:|
| humpback-whale-identification | new_whale | `9664` |
| CzechLynx | lynx_278 | `3245` |
| CzechLynx | lynx_041 | `1663` |
| CzechLynx | lynx_295 | `1265` |
| CzechLynx | lynx_287 | `1062` |
| CzechLynx | lynx_246 | `1050` |
| CzechLynx | lynx_250 | `914` |
| CzechLynx | lynx_271 | `894` |
| CzechLynx | lynx_276 | `884` |
| CzechLynx | lynx_133 | `827` |

Date quality:

| Metric | Value |
|---|---:|
| Valid ISO dates | `11857` |
| Non-ISO non-empty dates | `39347` |
| Missing dates | `107089` |
| Valid ISO min date | `2007-02-24` |
| Valid ISO max date | `2024-09-19` |

Image extensions:

| Extension | Rows |
|---|---:|
| `.jpg` | `153394` |
| `.png` | `4885` |
| `.jpeg` | `14` |

## Open-Set Split Statistics

The main training split uses `train.csv` and `val.csv`, where `val.csv` is exactly `val_known.csv + val_unknown.csv`. The main split ratio is `70.25%` train and `29.75%` validation.

| File | Rows | Dataset-scoped identities | Purpose |
|---|---:|---:|---|
| `train.csv` | `93101` | `11677` | supervised training |
| `val_known.csv` | `25890` | `5342` | validation identities also present in train |
| `val_unknown.csv` | `13542` | `1671` | open-set validation identities absent from train |
| `val.csv` | `39432` | `7013` | main validation: `val_known + val_unknown` |
| `val_unknown_extra.csv` | `2977` | `23` | overflow unknown rows for stress testing |
| `unlabeled_ood.csv` | `22783` | `13120` | singleton identities plus all `new_whale` rows |

### `open_set_splits/train.csv`

| Metric | Value |
|---|---:|
| Rows | `93101` |
| Dataset-scoped identities | `11677` |
| Singleton identities | `0` |
| Mean images per identity | `7.973` |
| Median images per identity | `3.0` |
| P90 images per identity | `12.0` |
| P99 images per identity | `96.5` |
| Max images per identity | `2515` |
| Missing image paths | `0` |
| Duplicate image paths | `0` |
| Total image size | `9.631 GiB` |
| `new_whale` rows | `0` |

Dataset counts:

| Dataset | Rows |
|---|---:|
| HappyWhale | `28608` |
| CzechLynx | `27839` |
| humpback-whale-identification | `9617` |
| WhaleSharkID | `3889` |
| IPanda50 | `3604` |
| LeopardID2022 | `3528` |
| GiraffeZebraID | `2998` |
| BalearicLizard | `2952` |
| CTai | `2444` |
| HyenaID2022 | `1546` |
| NDD20 | `1488` |
| SealID | `1101` |
| NyalaData | `957` |
| Jaguar_data | `794` |
| SalamanderID2025 | `722` |
| Giraffes | `585` |
| ReunionTurtles | `207` |
| AmvrakikosTurtles | `123` |
| ZakynthosTurtles | `99` |

### `open_set_splits/val_known.csv`

| Metric | Value |
|---|---:|
| Rows | `25890` |
| Dataset-scoped identities | `5342` |
| Singleton identities | `0` |
| Mean images per identity | `4.846` |
| Median images per identity | `2.0` |
| P90 images per identity | `8.0` |
| P99 images per identity | `44.0` |
| Max images per identity | `730` |
| Missing image paths | `0` |
| Duplicate image paths | `0` |
| Total image size | `2.676 GiB` |
| `new_whale` rows | `0` |

Dataset counts:

| Dataset | Rows |
|---|---:|
| CzechLynx | `8109` |
| HappyWhale | `7544` |
| humpback-whale-identification | `2357` |
| WhaleSharkID | `1139` |
| IPanda50 | `1047` |
| LeopardID2022 | `1026` |
| BalearicLizard | `887` |
| GiraffeZebraID | `833` |
| CTai | `709` |
| HyenaID2022 | `536` |
| NDD20 | `430` |
| SealID | `321` |
| NyalaData | `291` |
| Jaguar_data | `229` |
| SalamanderID2025 | `219` |
| Giraffes | `213` |

### `open_set_splits/val_unknown.csv`

| Metric | Value |
|---|---:|
| Rows | `13542` |
| Dataset-scoped identities | `1671` |
| Singleton identities | `0` |
| Mean images per identity | `8.104` |
| Median images per identity | `3.0` |
| P90 images per identity | `15.0` |
| P99 images per identity | `100.0` |
| Max images per identity | `100` |
| Missing image paths | `0` |
| Duplicate image paths | `0` |
| Total image size | `1.581 GiB` |
| `new_whale` rows | `0` |

Dataset counts:

| Dataset | Rows |
|---|---:|
| HappyWhale | `5255` |
| CzechLynx | `1881` |
| humpback-whale-identification | `1650` |
| WhaleSharkID | `930` |
| BalearicLizard | `520` |
| IPanda50 | `516` |
| GiraffeZebraID | `502` |
| CTai | `456` |
| LeopardID2022 | `399` |
| HyenaID2022 | `340` |
| SealID | `246` |
| NDD20 | `204` |
| NyalaData | `194` |
| Jaguar_data | `155` |
| SalamanderID2025 | `137` |
| Giraffes | `94` |
| ReunionTurtles | `30` |
| AmvrakikosTurtles | `18` |
| ZakynthosTurtles | `15` |

### `open_set_splits/val_unknown_extra.csv`

This file contains overflow rows from unknown validation identities after the `100` image cap. It is useful as an imbalanced unknown stress test, but it is intentionally excluded from the main validation score.

| Metric | Value |
|---|---:|
| Rows | `2977` |
| Dataset-scoped identities | `23` |
| Singleton identities | `1` |
| Mean images per identity | `129.435` |
| Median images per identity | `37.0` |
| P90 images per identity | `392.0` |
| P99 images per identity | `679.7` |
| Max images per identity | `727` |
| Missing image paths | `0` |
| Duplicate image paths | `0` |
| Total image size | `0.151 GiB` |
| `new_whale` rows | `0` |

Dataset counts:

| Dataset | Rows |
|---|---:|
| CzechLynx | `1931` |
| HappyWhale | `368` |
| IPanda50 | `332` |
| LeopardID2022 | `237` |
| CTai | `95` |
| WhaleSharkID | `14` |

### `open_set_splits/unlabeled_ood.csv`

This file contains rows that should not be used as supervised identity labels: all true singleton identities and all `new_whale` rows.

| Metric | Value |
|---|---:|
| Rows | `22783` |
| Dataset-scoped identities | `13120` |
| Singleton identities | `13119` |
| Mean images per identity | `1.737` |
| Median images per identity | `1.0` |
| P90 images per identity | `1.0` |
| P99 images per identity | `1.0` |
| Max images per identity | `9664` |
| Missing image paths | `0` |
| Duplicate image paths | `0` |
| Total image size | `1.359 GiB` |
| `new_whale` rows | `9664` |

Dataset counts:

| Dataset | Rows |
|---|---:|
| humpback-whale-identification | `11737` |
| HappyWhale | `9258` |
| GiraffeZebraID | `940` |
| SalamanderID2025 | `310` |
| BalearicLizard | `259` |
| LeopardID2022 | `126` |
| WhaleSharkID | `75` |
| NyalaData | `55` |
| Giraffes | `14` |
| HyenaID2022 | `6` |
| CTai | `3` |

### `open_set_splits/val.csv`

This file is the union of `val_known.csv` and `val_unknown.csv`. It does not include `val_unknown_extra.csv` or `unlabeled_ood.csv`.

| Metric | Value |
|---|---:|
| Rows | `39432` |
| Dataset-scoped identities | `7013` |
| Singleton identities | `0` |
| Mean images per identity | `5.623` |
| Median images per identity | `2.0` |
| P90 images per identity | `9.0` |
| P99 images per identity | `64.0` |
| Max images per identity | `730` |
| Missing image paths | `0` |
| Duplicate image paths | `0` |
| Total image size | `4.257 GiB` |
| `new_whale` rows | `0` |

Dataset counts:

| Dataset | Rows |
|---|---:|
| HappyWhale | `12799` |
| CzechLynx | `9990` |
| humpback-whale-identification | `4007` |
| WhaleSharkID | `2069` |
| IPanda50 | `1563` |
| LeopardID2022 | `1425` |
| BalearicLizard | `1407` |
| GiraffeZebraID | `1335` |
| CTai | `1165` |
| HyenaID2022 | `876` |
| NDD20 | `634` |
| SealID | `567` |
| NyalaData | `485` |
| Jaguar_data | `384` |
| SalamanderID2025 | `356` |
| Giraffes | `307` |
| ReunionTurtles | `30` |
| AmvrakikosTurtles | `18` |
| ZakynthosTurtles | `15` |

## Jaguar Source CSV Statistics

### `images/Jaguar_data/train.csv`

| Metric | Value |
|---|---:|
| Rows | `1468` |
| Columns | `filename,ground_truth,split,relative_path` |
| Unique identities | `31` |
| Missing relative image paths | `0` |
| Duplicate relative paths | `0` |

Top Jaguar training identities:

| Identity | Rows |
|---|---:|
| Marcela | `154` |
| Ousado | `125` |
| Kwang | `113` |
| Medrosa | `104` |
| Jaju | `77` |
| Ti | `72` |
| Lua | `68` |
| Saseka | `66` |
| Benita | `63` |
| Overa | `62` |

### `images/Jaguar_data/test.csv`

| Metric | Value |
|---|---:|
| Rows | `371` |
| Columns | `filename,ground_truth,split,relative_path` |
| Unique identities | `31` |
| Missing relative image paths | `0` |
| Duplicate relative paths | `0` |

Top Jaguar test identities:

| Identity | Rows |
|---|---:|
| Medrosa | `36` |
| Ousado | `29` |
| Marcela | `28` |
| Jaju | `23` |
| Kwang | `23` |
| Kamaikua | `18` |
| Ti | `16` |
| Benita | `15` |
| Saseka | `15` |
| Lua | `14` |

### `images/Jaguar_data/identity_thresholds.csv`

| Metric | Value |
|---|---:|
| Rows | `31` |
| Columns | `identity,threshold,dedup_enabled` |
| Unique identities | `31` |
| Unique thresholds | `2` |
| Unique dedup flags | `2` |
| Missing values | `0` |

### `images/Jaguar_data/duplicate_removal_decisions.csv`

This file records near-duplicate removal decisions made during Jaguar cleanup.

| Metric | Value |
|---|---:|
| Rows | `60` |
| Columns | `ground_truth,kept_filename,removed_filename,cosine_similarity,reason` |
| Identities affected | `13` |
| Unique kept filenames | `59` |
| Unique removed filenames | `60` |
| Missing values | `0` |

Duplicate-removal rows by identity:

| Identity | Rows |
|---|---:|
| Lua | `9` |
| Medrosa | `9` |
| Ousado | `7` |
| Jaju | `6` |
| Marcela | `6` |
| Kamaikua | `4` |
| Tomas | `4` |
| Ti | `4` |
| Benita | `3` |
| Bagua | `3` |
| Guaraci | `2` |
| Katniss | `2` |
| Saseka | `1` |

## Recommended Usage

For supervised open-set animal re-identification:

```text
Train: open_set_splits/train.csv
Known validation: open_set_splits/val_known.csv
Unknown validation: open_set_splits/val_unknown.csv
Combined validation: open_set_splits/val.csv
Extra unknown stress test: open_set_splits/val_unknown_extra.csv
Unlabeled/OOD pool: open_set_splits/unlabeled_ood.csv
```

Use `val.csv` for training validation and early stopping. Use `val_unknown_extra.csv` only for secondary unknown stress testing. Use `unlabeled_ood.csv` for unlabeled/self-supervised pretraining, OOD testing, threshold experiments, or manual inspection.

Do not merge `val_unknown_extra.csv` into the main validation score unless you explicitly want an imbalanced stress test dominated by a small number of large unknown identities.

## Important Notes

- `new_whale` is not treated as a real individual identity. It is only stored in `unlabeled_ood.csv`.
- Singleton identities are only stored in `unlabeled_ood.csv`.
- `val_known.csv` identities are intentionally present in `train.csv`.
- `val_unknown.csv` identities are intentionally absent from `train.csv`.
- `val_unknown_extra.csv` identities are intentionally absent from `train.csv` and intentionally overlap with `val_unknown.csv` because it stores capped overflow rows.
- The final files contain only existing, readable images.
- Blank species values come from upstream datasets whose metadata did not provide species labels.
- Non-ISO CzechLynx dates were preserved instead of modified.
