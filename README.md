# ApexID

ApexID is a PyTorch project for animal re-identification. It trains an
embedding model with configurable backbones, including OSNet, ConvNeXt,
MiewID, and MegaDescriptor, applies
metric/classification heads such as ArcFace and SphereFace2, validates with
retrieval metrics, and saves checkpoints, experiment history, and optional
embedding-space plots.

Run commands from the repository root:

```bash
cd /path/to/ApexID
```

## Project Map

- `apexid/`: Python package with datasets, transforms, models, losses, training
  engine, evaluation metrics, and utilities.
- `configs/`: YAML experiment configuration. `configs/default.yaml` is the
  default entry point used by scripts and tests.
- `scripts/`: command-line entry points for training and data-pipeline checks.
- `ApexID_Dataset/`: ignored local dataset root used by the default config.
- `data/`: local dataset preparation notes and scripts.
- `notebooks/`: exploratory notebook examples.
- `tests/`: pytest suite covering data, model, engine, metric, script, and
  utility behavior.
- `outputs/` and `wandb/`: generated runtime artifacts. These are not source
  modules and are not documented per run folder.

Each important folder has its own `README.md` explaining what it owns, how it
connects to the rest of the project, and which commands are useful for that
folder.

## Setup

The canonical environment is Python 3.13.5 with an exact CPU-validated lock:

```bash
python3.13 -m venv .venv
. .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu \
  torch==2.10.0 torchvision==0.25.0
pip install --no-deps -r requirements-lock.txt
pip check
```

`requirements.txt` is the readable dependency input. `requirements-lock.txt`
is the exact training/evaluation environment; `requirements-split-v2.txt`
remains the dedicated splitter environment. The CPU combination is tested.
CUDA/H100 validation is a separate handoff and is not claimed here.

## Train

All active configs expand `${VAR}` and `${VAR:-fallback}` recursively. Set the
three portable roots for a final-paper Split V2 run:

```bash
export APEXID_DATA_ROOT=/mounted/ApexID_Dataset
export APEXID_SPLIT_ROOT="$APEXID_DATA_ROOT/open_set_splits_v2"
export APEXID_OUTPUT_ROOT=/mounted/apexid-results
```

When unset, historical configs deliberately retain the V1-relative fallback
`ApexID_Dataset/open_set_splits`; selecting V2 is therefore explicit and does
not silently change historical membership. The new MiewID and MegaDescriptor
recipes below explicitly default to the repository's `open_set_splits_v2/`.

The train and validation CSV files must contain at least:

```text
identity,path
```

For the default ApexID dataset, CSV image paths such as `images/...` are
resolved relative to `root_dir: ApexID_Dataset`.

Start training:

```bash
python scripts/train.py --config configs/default.yaml
```

Training builds datasets, transforms, a P x K sampler, an ApexID model, the
configured head/loss, optimizer, scheduler, validator, checkpoint writer, early
stopping, optional W&B logging, and optional UMAP embedding plots.

`last_checkpoint.pt` is the resumable checkpoint. Format version 2 restores the
model, optimizer, scheduler, epoch/history, RNG state, early-stopping decisions,
and explicit DataLoader generators. `best_full_model.pt` and
`best_backbone.pt` remain the best-model artifacts selected by the configured
metric. Historical checkpoints without early-stopping state load with a visible
warning and cannot provide exact patience or best-model continuity.

To run a ConvNeXt experiment, keep the same ApexID model wrapper and choose a
registered TorchVision ConvNeXt backbone in the config:

```yaml
model:
  name: apexid
  embedding_dim: 512
  input_channels: 3
  backbone:
    name: convnext_tiny_pt
    kwargs:
      pretrained: true
```

Registered ConvNeXt backbones include `convnext_tiny`, `convnext_small`,
`convnext_base`, `convnext_large`, and pretrained-default aliases with `_pt`.
The default config remains OSNet-based.

## Fine-tune MiewID and MegaDescriptor

Two standalone recipes use the existing trainer, SphereFace2 head, open-set
validation, and checkpoint resume support:

| Config | Pretrained wildlife model | RGB size | Native embedding |
|---|---|---|---|
| `configs/finetune_miewid_msv3.yaml` | [conservationxlabs/miewid-msv3](https://huggingface.co/conservationxlabs/miewid-msv3) | 440 x 440 | 2152 |
| `configs/finetune_megadescriptor_l_384.yaml` | [BVRA/MegaDescriptor-L-384](https://huggingface.co/BVRA/MegaDescriptor-L-384) | 384 x 384 | 1536 |

These recipes default to the repository's `open_set_splits_v2/` CSVs. Set
`APEXID_DATA_ROOT` to the folder containing the CSVs' `images/...` paths.
For example, in PowerShell:

```powershell
$env:APEXID_DATA_ROOT = "D:/datasets/ApexID_Dataset"
$env:APEXID_SPLIT_ROOT = "D:/ApexID/open_set_splits_v2"
$env:APEXID_OUTPUT_ROOT = "D:/ApexID/outputs"
python scripts/debug_data_pipeline.py --config configs/finetune_miewid_msv3.yaml
python scripts/train.py --config configs/finetune_miewid_msv3.yaml
```

Run MegaDescriptor as a separate experiment:

```bash
python scripts/train.py --config configs/finetune_megadescriptor_l_384.yaml
```

On Linux, use `export APEXID_DATA_ROOT=/path/to/dataset` and equivalent exports
for the other roots. Both models use dependencies already in `requirements.txt`.
The first pretrained load downloads weights from Hugging Face; later runs reuse
the cache. The recipes pin model revisions and write to separate output folders.

Both recipes train the full embedding model at `1e-5` and the new head at
`1e-4`, with cosine decay and warmup. They use RGB without wavelets and retain
native embeddings without a new projection. A different `model.embedding_dim`
adds a trainable linear projection. To use ArcFace instead, set `head.name:
arcface`, replace its kwargs with `{s: 40.0, m: 0.3}`, and set `loss.name:
cross_entropy`.

MiewID retains its pretrained GeM pooling and batch normalization and replaces
the original unused classifier with ApexID's head. Its weights are verified by
SHA-256; loading does not execute Hugging Face remote code. MegaDescriptor uses
timm's Swin checkpoint conversion and requires exactly 384 x 384 inputs. Its
recipe follows the model-card example's mean/std `[0.5, 0.5, 0.5]` (the Hub
config instead lists ImageNet statistics). Keep this choice consistent between
training and evaluation. MiewID uses ImageNet mean/std.

The starting batch size is 8 (2 identities x 4 images). MegaDescriptor enables
gradient checkpointing to reduce activation memory. Adjust `num_identities`
and `batch_size` together, keeping `batch_size = num_identities * num_instances`;
adjust `val_batch_size` separately. W&B is disabled by default. Re-running a
recipe resumes its `last_checkpoint.pt`; use a new output directory for a new
experiment. For offline restoration from your own ApexID checkpoint, set
`model.backbone.kwargs.pretrained: false` to skip fetching the provider weights.

## Open-Set Validation

The default training config evaluates the full open-set validation splits every
epoch:

- `val_known.csv`: known identities held out from training images.
- `val_unknown.csv`: identities that do not appear in `train.csv`.
- `val.csv`: the mixed open-set split, normally
  `val_known.csv + val_unknown.csv`.

When `val.csv` is exactly the known split followed by the unknown split, the
validator reuses the known and unknown embeddings instead of running the mixed
loader again. It also reuses gallery-match scores for mixed metrics. This keeps
full validation exact while avoiding duplicated work.

Exact metrics can still be expensive on large galleries because `known/mAP`,
CMC/top-k retrieval, rejection metrics, threshold selection, and clustering all
depend on gallery or validation-set size. For faster approximate validation, set
positive `evaluation.sample_limits` values in `configs/default.yaml`; leave them
as `null` to keep full-split validation.

Known retrieval logs `known/top1`, `known/top5`, and `known/top10` for the common
small ranks. CMC logging is reserved for additional larger ranks by default
(`known/cmc_rank20` and `known/cmc_rank50`) to avoid duplicate top-k columns.

## Validate The Data Pipeline

Before a long run, inspect the CSV-to-label flow and one DataLoader batch:

```bash
python scripts/debug_data_pipeline.py --config configs/default.yaml
```

Skip batch loading when you only want metadata and label-map checks:

```bash
python scripts/debug_data_pipeline.py --config configs/default.yaml --skip-batch
```

## Calculate Mean And Std

Calculate normalization statistics for the configured training CSV:

```bash
python -m apexid.utils.calculate_mean_std --config configs/default.yaml --split train
```

Calculate statistics for validation:

```bash
python -m apexid.utils.calculate_mean_std --config configs/default.yaml --split val
```

Write training statistics back into `configs/default.yaml`:

```bash
python -m apexid.utils.calculate_mean_std --config configs/default.yaml --split train --write-config
```

Use an explicit CSV instead of the configured split:

```bash
python -m apexid.utils.calculate_mean_std \
  --config configs/default.yaml \
  --csv animals/fine_tune_train.csv \
  --root-dir animals
```

## Track Experiments With Weights & Biases

Log in once on the machine:

```bash
wandb login
```

Enable tracking in `configs/default.yaml`:

```yaml
tracking:
  enabled: true
  provider: wandb
  project: ApexID
  entity: MIA-Robotics
  run_name: null
  group: null
  tags: []
  mode: online
  log_artifacts: metadata
  block_on_artifact_upload: false
  artifact_finish_timeout: 300
```

Tips:

- Set `run_name` for a human-readable run name. Otherwise the experiment name is
  used.
- Add strings to `tags`, for example `["osnet", "sphereface2"]`.
- Use `mode: offline` to test logging locally without uploading. Local W&B files
  are written under `wandb/`, which is ignored by Git.
- `log_artifacts: none` logs metrics only and skips all W&B file uploads. This is
  the recommended setting on slow networks such as Kaggle.
- Use `log_artifacts: metadata` to upload small files such as best-run JSON
  metadata and UMAP plots when embedding plotting is enabled.
- Use `log_artifacts: best` when you also want the best `.pt` weights uploaded to
  W&B.
- `block_on_artifact_upload: false` keeps training moving while W&B uploads run
  in the background. W&B may still wait for pending uploads during `finish()`.
- If logging appears stuck for a long time after validation, the usual cause is
  uploading large checkpoint artifacts over a slow connection, not scalar metric
  logging.

The trainer logs epoch-level metrics such as `train_loss`, `val_loss`,
`val_top1`, `val_top5`, `val_mAP`, `val_micro_f1`, `lr`,
open-set metrics such as `known/mAP`, `unknown/ARI`, `rejection/AUROC`,
`mixed/NormAcc`, diagnostics cadence state, validation reuse flags, validation
sample counts when sampling is enabled, and early-stopping state.

## Test

Run the full suite:

```bash
pytest
```

Run focused suites:

```bash
pytest tests/data
pytest tests/engine
pytest tests/models
pytest tests/scripts
pytest tests/utils
pytest tests/test_default_config.py
```

## Sealed Open-World Benchmark Evaluation

This workflow creates an evaluation-only benchmark from `merged.csv` and
evaluates an already trained ApexID checkpoint. Run every command from the
repository root. The benchmark builder and the evaluator are deliberately
separate from training: building a benchmark does not train a model, and the
evaluator never changes the benchmark files.

Keep the source CSV, benchmark directory, image data, model checkpoints, and
evaluation results out of Git. In particular, the benchmark's private ground
truth is for the evaluator only and must not be published.

The benchmark evaluator now identifies B³ as
`standard_item_averaged_v2`. All B³ values from older evaluator code are invalid.
Reporting-only runs need metric regeneration; runs that used `b3_f1` for
development threshold selection need threshold retuning plus regenerated final
predictions and metrics. ARI, mAP, and unrelated metrics are unaffected by this
specific correction.

### 1. Check the benchmark and evaluator code

Run the focused tests before creating a benchmark or starting an expensive
inference run:

```bash
pytest -q tests/data/test_build_apexid_benchmark.py
pytest -q tests/evaluation/test_benchmark_metrics.py
pytest -q tests/evaluation/test_evaluate_apexid_model.py
```

Run the complete project test suite when preparing a reproducible result:

```bash
pytest -q
```

### 2. Build the evaluation benchmark

`merged.csv` must contain the columns `path`, `identity`, and `dataset`. Image
paths should be relative to the image root that will later be passed as
`--data-root`. The command below deterministically writes `open20`, `open40`,
and `open60` protocols below the output directory.

```bash
python data/build_apexid_benchmark.py \
  --input /path/to/merged.csv \
  --output-dir /path/to/apexid_benchmark \
  --seed 42 \
  --dev-novel-fraction 0.30 \
  --gallery-encounter-fraction 0.60 \
  --num-stream-orders 10
```

Parameter meanings:

- `--input`: source metadata CSV; defaults to `merged.csv` in the repository
  root.
- `--output-dir`: destination for all three benchmark protocols; defaults to
  `apexid_benchmark`.
- `--seed`: deterministic split and stream-order seed; use the same value to
  reproduce a build.
- `--dev-novel-fraction`: fraction of novel identities reserved for development
  tuning; the remainder is reserved for final testing.
- `--gallery-encounter-fraction`: target fraction of a known identity's whole
  encounters used as its database/gallery. Encounters, rather than individual
  images, are kept together to reduce leakage.
- `--num-stream-orders`: number of deterministic online final-test streams to
  produce for each protocol.

For an additional, slower image-content leakage audit, supply the root that
contains the relative `path` values:

```bash
python data/build_apexid_benchmark.py \
  --input /path/to/merged.csv \
  --output-dir /path/to/apexid_benchmark \
  --data-root /path/to/dataset-root \
  --audit-image-hashes
```

`--data-root` allows the builder to open images; `--audit-image-hashes` adds an
exact-content hash check and therefore requires every referenced image to be
mounted and readable. `--species-map-json /path/to/species_map.json` can supply
dataset-to-species metadata when it is not present in the input. Use
`--verbose` for detailed encounter-parsing and split diagnostics.

Each protocol includes public image manifests, private development and final
labels, validation reports, reproducibility metadata, and online stream files.
The public inference manifests do not contain identity or ground-truth columns.
Inspect the generated validation report before continuing, for example:

```bash
python -m json.tool /path/to/apexid_benchmark/open40/validation_report.json
```

### Optional: Create training splits and train a checkpoint

This is independent of the sealed evaluation benchmark. The observation-aware training
split script consumes the dataset metadata described in `data/README.md`
(`metadata.csv`, not the three-column benchmark `merged.csv`) and writes the
CSV paths used by the default training configuration:

```bash
python data/make_open_set_splits.py \
  --metadata metadata.csv \
  --data-root ApexID_Dataset \
  --output-dir ApexID_Dataset/open_set_splits_v2
python scripts/debug_data_pipeline.py --config /path/to/config_v2.yaml
python scripts/train.py --config /path/to/config_v2.yaml
```

- `data/make_open_set_splits.py` builds observation-aware `train.csv`, known
  validation, and identity-disjoint unknown validation files without modifying
  V1. See `data/README.md` and `SPLIT_V2_REPORT.md` for provenance and the H100
  completion gate.
- `--config configs/default.yaml` selects the model, data paths, transforms,
  optimizer, and validation settings. Copy and edit this file for a new
  experiment instead of changing a checkpoint's architecture at evaluation
  time.
- `debug_data_pipeline.py` verifies CSV paths, labels, and a DataLoader batch
  before training.

### 3. Run sealed batch and online evaluation

Use the original model configuration that matches the checkpoint. The example
evaluates the `open40` protocol, tunes all thresholds using development labels,
then evaluates final test labels only after final predictions have been saved.

```bash
python -m apexid.evaluation.evaluate_apexid_model \
  --weights /path/to/checkpoint.pt \
  --model-config /path/to/original_training_config.yaml \
  --benchmark-dir /path/to/apexid_benchmark/open40 \
  --data-root /path/to/dataset-root \
  --output-dir /path/to/results/open40 \
  --device cuda \
  --batch-size 64 \
  --num-workers 8 \
  --seed 42 \
  --evaluation-mode both \
  --gallery-scoring centroid \
  --threshold-objective f1 \
  --discovery-threshold-objective ari \
  --retrieval-chunk-size 1024 \
  --bootstrap-samples 1000 \
  --bootstrap-seed 42 \
  --bootstrap-confidence-level 0.95 \
  --bootstrap-method percentile \
  --bootstrap-min-valid-fraction 0.90
```

Parameter meanings:

- `--weights`: checkpoint to evaluate. Raw model weights, `state_dict`, and
  `model_state_dict` checkpoints are accepted, but incompatible embedding
  tensors fail rather than being silently ignored.
- `--model-config`: YAML that defines the exact ApexID embedding model and
  evaluation preprocessing used for those weights. It defaults to
  `configs/default.yaml`; specify the original experiment config whenever the
  checkpoint differs from that default.
- `--benchmark-dir`: one built protocol directory, such as `open20`, `open40`,
  or `open60`.
- `--data-root`: directory under which paths in the benchmark manifests resolve.
- `--output-dir`: new directory for caches, predictions, rankings, metrics,
  plots, and error diagnostics. It never writes into `--benchmark-dir`.
- `--device`: PyTorch device, usually `cuda` or `cpu`.
- `--batch-size` and `--num-workers`: image-extraction throughput controls;
  reduce them if GPU memory or host-memory pressure is high.
- `--seed`: makes evaluator sampling and development permutations reproducible.
- `--evaluation-mode`: `batch`, `online`, or `both` (the default). `online`
  evaluates every supplied stream independently and resets discovery state for
  each stream.
- `--gallery-scoring`: known-gallery aggregation: `centroid` (default),
  `max_image`, or `mean_top_k`.
- `--threshold-objective`: development-set objective for the known/novel
  decision: `f1` (default), `balanced_accuracy`, or `youden`.
- `--discovery-threshold-objective`: development-set objective for choosing the
  novel re-association threshold: `ari` (default) or `b3_f1`.
- `--retrieval-chunk-size`: number of queries scored per retrieval chunk;
  lower it to reduce peak memory on large galleries.
- `--bootstrap-samples`: number of deterministic bootstrap resamples for
  reported uncertainty; set `0` to skip bootstrapping.
- `--bootstrap-seed`, `--bootstrap-confidence-level`, `--bootstrap-method`, and
  `--bootstrap-min-valid-fraction`: configure identity-block percentile CIs.

Useful optional evaluator switches:

- `--image-size H W` overrides evaluation resize dimensions. One value creates
  a square size. It should agree with the checkpoint's training preprocessing.
- `--force-recompute` ignores otherwise valid embedding cache files after an
  intentional input or code change.
- `--no-save-embeddings` disables persistent embedding caches; saving is enabled
  by default under `OUTPUT_DIR/cache/`.
- `--no-amp` disables CUDA automatic mixed precision, which can help investigate
  numerical issues or run a strict CPU comparison.
- `--update-known-prototypes` remains parseable for compatibility but fails
  clearly because the protocol keeps known gallery prototypes static.
- `--require-clean-git` rejects dirty or unverifiable Git state. Without it,
  dirty paths are recorded in `run_manifest.json`.
- `--verbose` enables detailed progress and diagnostic logging.

Results are written only under `--output-dir`, including cached embeddings,
development-selected thresholds, final predictions, retrieval rankings, batch
metrics, individual online-stream metrics, aggregate reports, curves, and
plots. Do not use final-test ground-truth files to pick a checkpoint, threshold,
or clustering policy.

Training and evaluation both write `config_resolved.yaml`, `environment.json`,
and `run_manifest.json`. Online prototype updates are versioned as
`unnormalized_vector_sum_v2`; bootstrap intervals use
`identity_block_occurrence_layers_v2`. Online metrics and historical bootstrap
intervals must be regenerated. Batch predictions are unaffected by the online
prototype correction.

## Contributor Notes

- Keep configuration-driven behavior in sync with `configs/default.yaml`.
- Keep README command examples runnable from the repository root.
- Do not document generated folders per run or per image identity; document the
  source folder that owns the behavior instead.
- When adding a module, add or update the nearest folder README and include a
  targeted pytest command if tests exist.
