# Online Robustness, Reproducibility, and Bootstrap Fixes

## Scientific corrections

### Online discovered prototypes

Before this change, a normalized centroid was multiplied by its count during an
incremental update. After the second observation that quantity was no longer
the true vector sum, so later assignments depended on distorted prototypes.

Version `unnormalized_vector_sum_v2` stores a float64 unnormalized vector sum
and count. Incoming embeddings remain L2-normalized, scoring normalizes only a
view of the sum, and accepted observations add their original normalized vector
to the sum. Arrival order, threshold comparison (`>=`), `argmax` tie behavior,
causality, stream resets, and `NEW_000001...` naming are unchanged. Known
gallery prototypes remain static; the compatibility flag
`update_known_prototypes=true` now raises before inference.

### Identity-block bootstrap

The previous implementation concatenated resampled identity blocks without
relabeling duplicate occurrences. Multiple copies of one identity could merge
with themselves and understate uncertainty.

Version `identity_block_occurrence_layers_v2` samples true identities with
replacement, gives each occurrence a unique true label, and relabels every
original predicted cluster by its occurrence-rank layer. Copies of the same
identity cannot interact. Different identities that share a predicted cluster
retain the maximum possible overlap at matching ranks, and the original
fragmentation inside every copied block is preserved.

The evaluator exposes existing per-query Rank-1 and AP contributions without
changing their formulas. Identity-block percentile intervals cover Overall
ARI, Overall B³ F1, Known R1, Known mAP, Known E2E Accuracy, and Novelty AUROC.
Defaults are 1,000 replicates, evaluator seed, 0.95 confidence, percentile
method, and 0.90 minimum valid fraction. Output records requested, valid, and
invalid counts and reliability. Unreliable bounds are `NaN` with a warning.
Online stream permutation summaries remain descriptive order sensitivity, not
bootstrap or training-seed intervals.

## Portable environment and provenance

- Active YAMLs recursively expand `${VAR}` and `${VAR:-fallback}`. Unresolved
  required variables fail with their names. `APEXID_DATA_ROOT`,
  `APEXID_SPLIT_ROOT`, and `APEXID_OUTPUT_ROOT` replace committed site paths.
- The unset split fallback remains V1-relative. Final-paper commands must set
  `APEXID_SPLIT_ROOT` to frozen Split V2 explicitly.
- Python 3.13.5 CPU validation uses PyTorch 2.10.0+cpu, TorchVision 0.25.0+cpu,
  timm 1.0.27, Transformers 5.5.0, and Hugging Face Hub 1.13.0. The exact graph
  is in `requirements-lock.txt`; Split V2 retains its dedicated lock.
- Moving `wildlife-tools` and direct `gdown` requirements were removed. The
  immutable WildlifeDatasets commit remains scientific provenance but its
  unused downloader dependency graph is excluded from the training lock.
- DINOv3 Small and every OSNet variant use immutable Hugging Face revisions and
  verified SHA-256 values. Moving Google Drive OSNet downloads are gone.
- `apexid.utils.provenance` supplies atomic JSON/YAML, SHA-256, environment and
  Git capture, artifact records, strict clean-Git validation, and normalization
  hash validation.
- Training and evaluation write `config_resolved.yaml`, `environment.json`, and
  `run_manifest.json`. Evaluation records evaluator, B³, online-prototype, and
  bootstrap versions/configuration. Public inputs are hashed before inference;
  private labels are hashed only after prediction persistence.
- `scripts/repro_smoke.py` exercises config expansion, a scratch model forward,
  checkpoint round-trip, causal online assignment, and evaluator metrics fully
  offline with synthetic tensors.

## Compatibility and historical impact

- Representation retraining required: **no**.
- Online inference and all online metrics must be rerun.
- Historical bootstrap intervals must be regenerated.
- Batch predictions remain valid for the online prototype defect.
- Existing thresholds, batch inference, model/loss definitions, Split V2, and
  benchmark membership are unchanged.
- The CI CSV is additive; existing result keys and `paper_main_results.csv`
  remain.
- This change does not resolve unrelated `AUDIT.md` findings.

## Changed surfaces

Core changes are in `apexid/evaluation/benchmark_evaluator.py`,
`apexid/evaluation/open_set_metrics.py`, pretrained backbone wrappers,
`apexid/utils/config.py`, new `apexid/utils/provenance.py`, `scripts/train.py`,
and new `scripts/repro_smoke.py`. All four active YAMLs, the three Slurm jobs
and base64 copies, dependency inputs/lock, focused tests, root/package/runbook
documentation, and `THIRD_PARTY_PROVENANCE.md` were updated. Split V2 code,
benchmark construction/membership, heads, losses, configured selection metrics,
and historical outputs were not changed.

## Validation evidence

Baseline before this task: 201 tests. After implementation:

- final focused evaluator/model/config/provenance/smoke suites: **73 passed**;
- fresh Python 3.13 CPU lock: `pip check` reported no broken requirements;
- local full suite: **229 passed**, **0 failed**, **0 skipped**;
- fresh-environment full suite: **229 passed**, **0 failed**, **0 skipped**, 14
  Matplotlib/PyParsing third-party deprecation warnings;
- offline reproduction smoke: passed with exact checkpoint output equality and
  deterministic `NEW_000001, NEW_000001, NEW_000002` assignment;
- evaluator and training CLI help: passed;
- compile/AST, `bash -n`, all three base64 decode comparisons, and
  `git diff --check`: passed.

## Remaining boundary

CUDA/H100 validation is pending by design. The target job must install the same
lock with a site-supported PyTorch backend and return `environment.json`,
`run_manifest.json`, `nvidia-smi`, smoke/test logs, and any deterministic
comparison evidence. Persistent worker-internal RNG and hardware-dependent CUDA
kernels may prevent bitwise equivalence even with restored explicit generator
state.
