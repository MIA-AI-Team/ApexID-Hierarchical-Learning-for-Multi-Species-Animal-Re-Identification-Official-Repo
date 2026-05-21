# ApexID Animal Re-Identification Dataset

This project contains a cleaned multi-dataset animal re-identification metadata table and an open-set train/validation split designed for animal ReID model training.

The final training-ready master metadata file is:

```text
metadata.csv
```

The final open-set split files are:

```text
open_set_splits/train.csv
open_set_splits/val_known.csv
open_set_splits/val_unknown.csv
open_set_splits/val_unknown_extra.csv
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

The previous special identity `new_whale` from `humpback-whale-identification` is treated as unknown data, not as a real individual whale identity.

## Open-Set Split Policy

The split script uses deterministic seed `42` and dataset-scoped identity keys: `(dataset, identity)`.

Split behavior:

- `train.csv`: supervised identity training set.
- `val_known.csv`: held-out images from identities that also appear in train.
- `val_unknown.csv`: balanced unknown-identity validation set with no identity overlap with train.
- `val_unknown_extra.csv`: extra unknown rows that were capped out of balanced validation. This keeps useful data available without letting large identities dominate the main validation metric.
- `val.csv`: `val_known.csv + val_unknown.csv`.

Balancing rules:

- About `20%` of images from train-seen identities are held out into `val_known`.
- About `10%` of eligible identities per dataset are held out into `val_unknown`.
- Normal unknown identities are capped at `50` images each in `val_unknown`.
- `new_whale` is excluded from train and val_known.
- `new_whale` contributes `1000` rows to `val_unknown`.
- The remaining `8664` `new_whale` rows are stored in `val_unknown_extra.csv`.

Recreate the split files:

```bash
python make_open_set_splits.py
```

## Split Integrity

The generated split files passed these checks:

| Check | Result |
|---|---:|
| Train and val_unknown identity overlap | `0` |
| Train and val_unknown_extra identity overlap | `0` |
| val_known identities missing from train | `0` |
| val_known and val_unknown identity overlap | `0` |
| `val.csv == val_known.csv + val_unknown.csv` | `true` |
| `new_whale` rows in train | `0` |
| `new_whale` rows in val_known | `0` |
| `new_whale` rows in val_unknown | `1000` |
| `new_whale` rows in val_unknown_extra | `8664` |

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

### `open_set_splits/train.csv`

| Metric | Value |
|---|---:|
| Rows | `107774` |
| Dataset-scoped identities | `25131` |
| Singleton identities | `17758` |
| Mean images per identity | `4.288` |
| Median images per identity | `1` |
| P90 images per identity | `6` |
| P99 images per identity | `50` |
| Max images per identity | `2596` |
| Missing image paths | `0` |
| Duplicate image paths | `0` |
| Total image size | `10.848 GiB` |

Dataset counts:

| Dataset | Rows |
|---|---:|
| HappyWhale | `37290` |
| CzechLynx | `29901` |
| humpback-whale-identification | `10974` |
| WhaleSharkID | `4219` |
| GiraffeZebraID | `3772` |
| LeopardID2022 | `3722` |
| IPanda50 | `3719` |
| BalearicLizard | `3228` |
| CTai | `2777` |
| HyenaID2022 | `1746` |
| NDD20 | `1500` |
| NyalaData | `1088` |
| SealID | `1054` |
| SalamanderID2025 | `1012` |
| Jaguar_data | `854` |
| Giraffes | `624` |
| ReunionTurtles | `142` |
| AmvrakikosTurtles | `84` |
| ZakynthosTurtles | `68` |

### `open_set_splits/val_known.csv`

| Metric | Value |
|---|---:|
| Rows | `28299` |
| Dataset-scoped identities | `12012` |
| Singleton identities | `9320` |
| Mean images per identity | `2.356` |
| Median images per identity | `1` |
| P90 images per identity | `3` |
| P99 images per identity | `25` |
| Max images per identity | `649` |
| Missing image paths | `0` |
| Duplicate image paths | `0` |
| Total image size | `2.915 GiB` |

Dataset counts:

| Dataset | Rows |
|---|---:|
| HappyWhale | `9597` |
| CzechLynx | `7476` |
| humpback-whale-identification | `3370` |
| WhaleSharkID | `1100` |
| GiraffeZebraID | `1063` |
| LeopardID2022 | `966` |
| BalearicLizard | `931` |
| IPanda50 | `930` |
| CTai | `697` |
| HyenaID2022 | `450` |
| NDD20 | `378` |
| NyalaData | `273` |
| SalamanderID2025 | `267` |
| SealID | `260` |
| Jaguar_data | `216` |
| Giraffes | `178` |
| ReunionTurtles | `71` |
| AmvrakikosTurtles | `42` |
| ZakynthosTurtles | `34` |

### `open_set_splits/val_unknown.csv`

| Metric | Value |
|---|---:|
| Rows | `10511` |
| Dataset-scoped identities | `1337` |
| Singleton identities | `0` |
| Mean images per identity | `7.862` |
| Median images per identity | `3` |
| P90 images per identity | `15` |
| P99 images per identity | `50` |
| Max images per identity | `1000` |
| Missing image paths | `0` |
| Duplicate image paths | `0` |
| Total image size | `1.195 GiB` |
| `new_whale` rows | `1000` |

Dataset counts:

| Dataset | Rows |
|---|---:|
| HappyWhale | `3670` |
| humpback-whale-identification | `2353` |
| CzechLynx | `1044` |
| WhaleSharkID | `538` |
| BalearicLizard | `459` |
| GiraffeZebraID | `438` |
| LeopardID2022 | `360` |
| IPanda50 | `250` |
| SealID | `249` |
| HyenaID2022 | `232` |
| NDD20 | `228` |
| CTai | `187` |
| NyalaData | `136` |
| SalamanderID2025 | `109` |
| Giraffes | `104` |
| Jaguar_data | `103` |
| ReunionTurtles | `24` |
| AmvrakikosTurtles | `15` |
| ZakynthosTurtles | `12` |

### `open_set_splits/val_unknown_extra.csv`

This file contains the unknown rows that were excluded from the balanced validation set due to caps. It is useful as a larger, imbalanced unknown stress test or as unlabeled data.

| Metric | Value |
|---|---:|
| Rows | `11709` |
| Dataset-scoped identities | `48` |
| Singleton identities | `0` |
| Mean images per identity | `243.938` |
| Median images per identity | `38.5` |
| P90 images per identity | `176` |
| P99 images per identity | `8664` |
| Max images per identity | `8664` |
| Missing image paths | `0` |
| Duplicate image paths | `0` |
| Total image size | `0.440 GiB` |
| `new_whale` rows | `8664` |

Dataset counts:

| Dataset | Rows |
|---|---:|
| humpback-whale-identification | `8664` |
| CzechLynx | `1339` |
| IPanda50 | `600` |
| HappyWhale | `476` |
| LeopardID2022 | `268` |
| WhaleSharkID | `190` |
| SealID | `105` |
| CTai | `46` |
| NDD20 | `16` |
| Jaguar_data | `5` |

### `open_set_splits/val.csv`

This file is the union of `val_known.csv` and `val_unknown.csv`. It does not include `val_unknown_extra.csv`.

| Metric | Value |
|---|---:|
| Rows | `38810` |
| Dataset-scoped identities | `13349` |
| Singleton identities | `9320` |
| Mean images per identity | `2.907` |
| Median images per identity | `1` |
| P90 images per identity | `4` |
| P99 images per identity | `34` |
| Max images per identity | `1000` |
| Missing image paths | `0` |
| Duplicate image paths | `0` |
| Total image size | `4.110 GiB` |
| `new_whale` rows | `1000` |

Dataset counts:

| Dataset | Rows |
|---|---:|
| HappyWhale | `13267` |
| CzechLynx | `8520` |
| humpback-whale-identification | `5723` |
| WhaleSharkID | `1638` |
| GiraffeZebraID | `1501` |
| BalearicLizard | `1390` |
| LeopardID2022 | `1326` |
| IPanda50 | `1180` |
| CTai | `884` |
| HyenaID2022 | `682` |
| NDD20 | `606` |
| SealID | `509` |
| NyalaData | `409` |
| SalamanderID2025 | `376` |
| Jaguar_data | `319` |
| Giraffes | `282` |
| ReunionTurtles | `95` |
| AmvrakikosTurtles | `57` |
| ZakynthosTurtles | `46` |

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
```

Use `val_unknown_extra.csv` for stress testing or unlabeled/self-supervised use. Do not combine it into the main validation score unless you explicitly want an imbalanced validation set dominated by `new_whale` and other large unknown identities.

## Important Notes

- `new_whale` is not treated as a real individual identity. It is only used as unknown validation data.
- `val_known.csv` identities are intentionally present in `train.csv`.
- `val_unknown.csv` and `val_unknown_extra.csv` identities are intentionally absent from `train.csv`.
- The final files contain only existing, readable images.
- Blank species values come from upstream datasets whose metadata did not provide species labels.
- Non-ISO CzechLynx dates were preserved instead of modified.
