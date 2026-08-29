# ApexID Observation-Aware Split V2 Report

Status: implementation and local synthetic validation complete; real-corpus
DINOv2 generation, perceptual review, normalization, and training-pipeline smoke
test require the documented H100/Slurm handoff.

## Reproducibility anchor

| Item | Audited value |
|---|---|
| Authoritative input | repository-root `metadata.csv` |
| Rows | 158,293 |
| Columns | `identity,path,date,orientation,species,dataset` |
| SHA-256 | `fd8a921f7ab5566b02fa166bfbbfc948375ee68226b15c29caab8b7fc3ebacf0` |
| Dataset-scoped identities | 26,468 |
| Datasets | 19 |
| Root vs `ApexID_Dataset/metadata.csv` | byte-identical at audit time |
| Split seed | 42 |
| Unknown identity ratio | 0.125, selected per dataset |
| Known validation image target | 0.225 per identity |

The implementation changes only the known image-level split. Dataset-wise
unknown identity selection, the unknown cap/overflow interface, singleton OOD
routing, `new_whale` OOD routing, and all six public CSV schemas are retained.
`data/build_apexid_benchmark.py`, model architecture, losses, evaluator,
checkpointing, and training objectives are unchanged.

## Corpus preflight results

Strict date parsing produced:

| Date state | Rows |
|---|---:|
| `%Y-%m-%d` | 11,857 |
| `%d-%m-%Y` | 39,347 |
| blank/missing | 107,089 |
| malformed nonblank | 0 |

The byte-level scan found 12 same-identity duplicate groups and 12 extra rows.
No identical-content group carried conflicting canonical `(dataset, identity)`
labels. V1 contained two exact-content groups crossing train and `val_known`.
It also contained 4,081 shared dataset/identity/date groups across that boundary;
these are precisely the leakage modes Split V2 makes impossible by construction
and validates again after writing.

Verified explicit parsers:

| Dataset | Parser evidence | Rows structurally covered at audit |
|---|---|---:|
| CTai | numeric `img-id` | 3,707 |
| Giraffes | original basename before crop/hash suffix | 906 |
| IPanda50 | `vNNN` video or explicit timestamp session | 5,454 |
| NDD20 | original numeric filename prefix | 2,122 |
| NyalaData | numeric capture shared by sides/copies | 1,497 |

The IPanda50 total includes structurally valid timestamp sessions even when a
session has only one retained prepared image; a single group alone does not make
an identity eligible for group-subset validation.

## Scientific method

Known identities are routed in this order:

1. Fully dated identities with at least two distinct dates use a chronological
   suffix closest to the target, with at least one earlier training group.
2. Identities completely covered by at least two explicit source groups use a
   deterministic whole-component subset closest to the target.
3. All other identities use the similarity fallback. Exact duplicates,
   explicit sources, and known same-date observations are hard must-links.
   DINOv2 threshold edges form connected components within each identity. Every
   multi-image component remains in training; only singleton components can be
   selected for `val_known`. If none is independently available, the identity
   is train-only.

This follows WildlifeReID-10k Section 3.1/Algorithm 1 for similarity-based
splitting: detected similar clusters are retained in training and only
unclustered observations are eligible for validation. Similarity thresholds are
frozen independently of ApexID validation metrics.

## Frozen feature environment

| Item | Value |
|---|---|
| Model | `timm/vit_large_patch14_dinov2.lvd142m` |
| Hugging Face revision | `4741e1cafbf45415e77074bb0cb42dba76c8684a` |
| Weight SHA-256 | `0424a5d1b515278cba3c6640ccbeaacc41de59d3a93df0dd5e494285eea2b355` |
| Feature shape | 1,024-dimensional float32 |
| Input | RGB, 518×518 bilinear, `[0,1]` |
| Normalization | ImageNet mean/std, then embedding L2 normalization |
| `timm` | 1.0.27 |
| WildlifeDatasets reference | v1.0.9 / commit `67380d7951b89f02e48c84a7d3546188b9dbec8a` |
| WildlifeDatasets license | AGPL-3.0-or-later |

The complete extraction environment is frozen in `requirements-split-v2.txt`;
the main training environment remains in `requirements.txt`.

Tagged WildlifeDatasets numeric thresholds are CTai 0.95, HyenaID2022 0.98,
IPanda50 0.97, LeopardID2022 0.96, NDD20 0.97, NyalaData 0.999, SealID 0.96,
and WhaleSharkID 0.98. Every other dataset uses the frozen 0.98 default.

The splitter does not import WildlifeDatasets code. Its version/commit/license
are recorded as scientific provenance. A release limitation remains: the DINO
Hugging Face model card declares Apache-2.0 while its pretrained configuration
declares CC-BY-NC-4.0. Distribution must resolve that metadata discrepancy.

## V1 versus V2

The V1 values below are actual corpus outputs. V2 columns are intentionally not
fabricated; `split_summary.json` fills this comparison during the H100 run.

| Output | V1 rows | V1 identities | V2 rows | V2 identities |
|---|---:|---:|---:|---:|
| train | 93,101 | 11,677 | H100 pending | H100 pending |
| val_known | 25,890 | 5,342 | H100 pending | H100 pending |
| val_unknown | 13,542 | 1,671 | H100 pending | H100 pending |
| val | 39,432 | 7,013 | H100 pending | H100 pending |
| val_unknown_extra | 2,977 | 23 | H100 pending | H100 pending |
| unlabeled_ood | 22,783 | 13,120 | H100 pending | H100 pending |

| Leakage/audit measure | V1 actual | V2 completion requirement |
|---|---:|---:|
| Exact-content groups crossing train↔`val_known` | 2 | 0 |
| Dataset/identity/date groups crossing train↔`val_known` | 4,081 | 0 |
| Conservative perceptual candidate pairs | 1,443 preflight audit | list and explain all remaining |

## Interfaces and artifacts

The canonical command is:

```bash
python data/make_open_set_splits.py \
  --metadata metadata.csv \
  --data-root ApexID_Dataset \
  --output-dir ApexID_Dataset/open_set_splits_v2 \
  --feature-cache ApexID_Dataset/split_v2_feature_cache \
  --near-duplicate-audit
```

The cache contains `features.npy`, `feature_rows.csv`, and
`cache_metadata.json`. It is accepted only when ordered paths/hashes, metadata
hash, model/checkpoint, preprocessing, dtype/shape, and both artifact hashes
match. A stale or incomplete cache aborts instead of being silently reused.

Output creation is staged and atomically renamed. An occupied destination is
refused unless `--overwrite-v2` is explicit; even then only the exact named V2
directory is replaced. V1 is never cleared or overwritten.

## Validation completed locally

Synthetic coverage includes:

- chronological suffixes, same-day protection, and two-date identities;
- every explicit parser and whole-group assignment;
- similarity connected components and train-only multi-image policy;
- exact duplicates and cross-label conflict failure;
- identity-disjoint unknown selection and group-aware cap overflow;
- singleton and `new_whale` routing;
- partial, missing, and malformed dates;
- deterministic byte-identical outputs and complete accounting;
- cache incompleteness/staleness rejection;
- train-only normalization provenance.

The repository test commands are:

```bash
pytest -q tests/data/test_make_open_set_splits.py \
  tests/utils/test_calculate_mean_std.py tests/data/test_mean_std.py
pytest -q
```

Local results at this implementation revision are 27 focused split,
normalization, and data-pipeline tests passed; the complete repository suite is
175 passed. The data-pipeline smoke test constructs and loads an
`open_set_splits_v2` directory through an actual transformed P×K DataLoader.
Repeat both checks on the H100 checkout after real artifacts exist.

## H100 completion gate

Use `slurm_make_splits.sh` and `slurm_mean_std.sh` as described in
`CLUSTER_RUN_README.md`. Completion requires all of the following actual
artifacts/results:

- V2 `split_summary.json` and `split_manifest.csv` hashes;
- actual V2 global/per-dataset counts, ratios, and V1 comparison;
- zero exact-content and zero temporal-observation crossings;
- both V1 and V2 perceptual candidate CSVs with candidate disposition notes;
- train-only `normalization_stats.json` and its manifest/train CSV hashes;
- identical updated mean/std in all four 440×440 RGB training YAMLs;
- validation-only success, full test-suite success, and a V2 data-pipeline smoke
  test.

Until those returned cluster artifacts are inserted above, the scientific split
implementation is ready but the real-corpus release is not declared complete.
