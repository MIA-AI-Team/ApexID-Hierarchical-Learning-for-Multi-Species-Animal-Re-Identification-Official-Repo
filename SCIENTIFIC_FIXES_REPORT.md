# ApexID Scientific Fixes Report: B³ and Resume Safety

Date: 2026-08-30  
Scope: confirmed B³ correctness and early-stopping resume defects only

## Executive Result

The benchmark evaluator now uses standard item-averaged B³, identified as
`standard_item_averaged_v2`. Resumable training checkpoints now use format
version 2 and preserve early-stopping decisions plus explicit DataLoader
generators. The expanded suite passes all 201 tests.

From the perspective of these two defects, ApexID may begin a fresh final
training run. This statement does not resolve or waive other findings in
`AUDIT.md`.

## A. B³ Correction

### Defect and correction

The old implementation divided the global sum of squared contingency counts by
the global sum of squared cluster sizes. That is not item-averaged B³ and can
produce incorrect scientific results.

The corrected implementation uses, for contingency cell count `n_ij`, predicted
cluster size `|C_j|`, true cluster size `|T_i|`, and `N` images:

```text
precision = (1/N) * sum_ij(n_ij^2 / |C_j|)
recall    = (1/N) * sum_ij(n_ij^2 / |T_i|)
F1        = 2 * precision * recall / (precision + recall)
```

Audit regression:

```text
true:      [A, A, B]
predicted: [X, Y, Y]

before: precision=0.600000, recall=0.600000, F1=0.600000
after:  precision=0.666667, recall=0.666667, F1=0.666667
```

Inputs now explicitly support arbitrary hashable labels and reject unequal
lengths, multidimensional arrays, and unhashable labels. A single item returns
`1/1/1`. Empty inputs retain the benchmark convention of three `NaN` values for
an unavailable scenario subset.

### Consumers and provenance

B³ flows through `clustering_metrics` and `discovery_metrics` into overall,
scenario, dataset, species, macro-dataset, and paper-table benchmark outputs. It
can also select both online discovery and batch clustering thresholds when
`discovery_threshold_objective=b3_f1`.

Corrected runs record the version in:

- `run_config.json` as `metric_versions.b3`;
- `evaluation_results.json` as `metric_versions.b3`;
- `paper_main_results.csv` as `B3_Metric_Version`.

The online discovery threshold curve now reports separate `mean_ari` and
`mean_b3_f1` columns. Previously, a B³ objective value was mislabeled as
`mean_ari`.

### Historical impact

- All B³ values produced before `standard_item_averaged_v2` are invalid.
- If B³ was reporting-only, predictions remain usable for this defect, but all
  B³-bearing metrics and tables must be regenerated.
- If `b3_f1` selected a development threshold, retune the threshold and
  regenerate final predictions, diagnostics, and metrics.
- This correction does not invalidate ARI, mAP, or unrelated metrics.

## B. Resume-Safety Correction

### Checkpoint V2

`checkpoint_format_version` is now `2`. The existing `extra_state` field stores:

- `early_stopping`: state version, monitor, mode, patience, minimum delta, best
  metric/score, best epoch, patience counter, stop flag, and `val_loss_min`;
- `data_loader_generators`: recursively collected explicit `torch.Generator`
  states for train and validation loaders.

The model, optimizer, scheduler, epoch, next epoch, history, metrics, and Python,
NumPy, PyTorch CPU, and PyTorch CUDA RNG states retain their existing checkpoint
locations.

Resume validates monitor, direction, patience, and minimum delta before training
continues. A mismatch fails clearly instead of silently changing future
decisions. A restored `early_stop=True` returns the saved history without
running another epoch.

### Best artifacts and legacy checkpoints

A worse first post-resume validation no longer becomes a false new best. Tests
confirm that the existing `best_full_model.pt` bytes remain unchanged, the
historical best metric/epoch survive, and patience continues from its saved
counter. A genuinely better post-resume value still replaces the best artifact
and resets patience.

Historical checkpoints without early-stopping state remain loadable under
Policy A. They emit `LegacyCheckpointWarning`, state that exact resume semantics
are unavailable, and do not fabricate best history. `scripts/train.py` gives
this category an explicit visible filter even though unrelated library warnings
are normally suppressed.

### Reproducibility boundary

The epoch-derived P x K sampler remains reproducible from seed plus epoch. The
random-sampler path and validation loaders restore explicit generator state.
The deterministic integration test uses CPU, zero workers, shuffled loading,
AdamW, and CyclicLR and obtains equal model weights, optimizer state, scheduler
state/LR sequence, history, best epoch/metric, counter, and stopping epoch.

Bitwise equivalence is not promised for persistent worker-internal RNG,
third-party augmentation state retained inside workers, or hardware-dependent
CUDA kernels. Logical early-stopping/model-selection equivalence is preserved.

## Files Changed and Compatibility

| Area | Files | Scientific effect | Compatibility |
|---|---|---|---|
| B³ implementation | `apexid/evaluation/benchmark_metrics.py`, package exports | Correct item-averaged metric | Public metric keys unchanged; corrected values intentionally differ |
| Benchmark evaluator | `apexid/evaluation/benchmark_evaluator.py` | Correct threshold objective and versioned outputs | Existing objectives retained; output schemas gain provenance fields |
| Early stopping | `apexid/engine/early_stopping.py`, package exports | Complete serializable decision state | Existing constructor/call behavior retained |
| Checkpoints | `apexid/engine/checkpoint.py` | Format V2, early-stop and loader-generator restoration | Legacy files load with a warning |
| Training lifecycle | `apexid/engine/trainer.py`, `scripts/train.py` | Saves/restores state before the next decision | Configured model-selection metric remains unchanged |
| Tests | engine, evaluator, and script tests | Analytical, randomized, round-trip, and interruption regressions | 26 tests added |
| Documentation | root, engine, evaluation, config, and test READMEs | Historical and operational guidance | No experiment YAML values changed |

No Split V2 file, benchmark manifest, model, loss, dataset-preparation code, or
training YAML value was changed.

## Validation Evidence

Baseline before changes:

- 175 tests collected.
- 56 focused evaluator/engine/script tests passed.
- Audit B³ regression returned `0.6/0.6/0.6`.

After changes:

- Focused regression suite: `75 passed`, `0 failed`, `0 skipped`.
- Full repository suite: `201 passed`, `0 failed`, `0 skipped` in 23.82 seconds.
- Python compilation: passed for `apexid`, `scripts`, and `tests`.
- `git diff --check`: passed.
- Training CLI help: passed.
- Sealed evaluator CLI help: passed.
- Checkpoint V2 save/load smoke test: passed.
- B³ smoke test: exact `2/3` precision, recall, and F1; passed.

Observed environment warnings:

- `pytest-asyncio` reports an unset future loop-scope default during startup.
- Albumentations cannot perform its online version check in the restricted
  environment.
- The evaluator CLI created temporary Matplotlib/font caches because the user
  cache directory is read-only in the sandbox.

These warnings did not fail tests and are unrelated to either scientific fix.

## Required Follow-Up for Existing Results

Do not relabel historical artifacts as corrected. Regenerate B³ metrics for
reporting-only runs, and retune/reinfer any run that selected thresholds with
`b3_f1`. Start final training from a V2 checkpoint lineage; treat any continued
legacy-checkpoint run as non-equivalent unless restarted from the beginning.
