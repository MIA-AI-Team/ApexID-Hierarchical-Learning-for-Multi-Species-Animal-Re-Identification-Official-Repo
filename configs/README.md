# configs

This folder stores YAML experiment configuration. The default project entry
point is `default.yaml`.

## Main Files

- `finetune_miewid_msv3.yaml`: pinned wildlife-pretrained MiewID, native
  2152-d embeddings, 440px RGB and ImageNet normalization.
- `finetune_megadescriptor_l_384.yaml`: pinned wildlife-pretrained Swin-L,
  native 1536-d embeddings, 384px RGB, model-card 0.5 mean/std and gradient
  checkpointing. Both new recipes use SphereFace2, full fine-tuning, separate
  output folders, and default explicitly to repository `open_set_splits_v2/`.

- `default.yaml`: complete training configuration for the default ApexID
  experiment. It defines dataset paths, transforms, sampler settings, model
  shape, backbone, head, loss, optimizer, scheduler, LR range test, finetune
  profile, training behavior, validation metrics, embedding plots, checkpoints,
  early stopping, and experiment tracking.
- `cluster_home.yaml`: current cluster ConvNeXt Small fine-tuning recipe.
- `cluster_dinov3_vitb16.yaml`: DINOv3 ViT-B/16 native 768-d CLS embeddings,
  512px RGB, ImageNet processor mean/std, pinned Hugging Face revision and
  weight SHA-256, SphereFace2, and `head_first` then full fine-tune.
- `cluster_dinov3_vitb16_subcenter.yaml`: same DINOv3 ViT-B/16 recipe with
  Sub-center ArcFace (`k=3`, `s=40`) and CrossEntropy instead of SphereFace2.
- `cluster_convnextv2_base.yaml`: ConvNeXt V2 Base ImageNet-22k weights, native
  1024-d pooled embeddings, 440px gray3, SphereFace2, and full fine-tune
  matching the best ConvNeXt Small recipe.
- `cluster_bhca_arcface.yaml`, `cluster_bhca_subcenter_arcface.yaml`, and
  `cluster_bhca_subcenter_sphereface2.yaml`: controlled ConvNeXt Small Split V2
  BHCA comparison with identical data, sampling, optimizer, schedule, and
  validation settings. These runs automatically fill approved dataset-level
  species metadata and stop on unresolved nulls or conflicts. The frozen CSVs
  are not rewritten. See `docs/BHCA_LOSS.md`.
- `cluster_pretrained.yaml`: cluster paths for ImageNet-pretrained fine-tuning
  (e.g. `osnet_x1_0_pt`). Auto-switches to cosine + warmup when
  `finetune.enabled: auto`.

## How It Connects

Most runnable commands load this file through `apexid.utils.config.load_config`.
`scripts/train.py` builds the full training stack from it, and tests assert that
the default values remain internally consistent.

Important sections:

- `experiment`: run name and output directory.
- `data`: train, known-validation, unknown-validation, mixed-validation CSVs,
  image root, image size, wavelet mode, optional `grayscale` (gray→3ch), mean/std.
- `dataloader`: batch size, workers, and P x K sampler parameters.
- `model`: ApexID embedding dimensions, input channels, and backbone selection.
- `head` and `loss`: margin head and external objective configuration.
- `optimizer`, `scheduler`, `lr_range_test`, and `finetune`: learning-rate behavior.
- `training`, `validation`, and `evaluation`: epoch loop, label ignore indices,
  metric cadence, open-set metric toggles, thresholds, sample limits, and
  validation chunk size.
- `checkpoint` and `early_stopping`: resumable latest checkpoint and best-model
  artifacts.
- `tracking`: W&B logging settings.

`checkpoint.resume_path` accepts an explicit last-checkpoint path or `auto`.
Checkpoint format version 2 validates that `early_stopping.monitor`, `mode`,
`patience`, and `min_delta` still match before resuming. Changing those values
requires a new run rather than silently reinterpreting saved patience state.
Legacy checkpoints remain loadable with a visible reproducibility warning.

## Portable roots

`load_config` recursively expands `${VAR}` and `${VAR:-fallback}` in mappings
and lists and raises on unresolved variables. Every active YAML uses:

```bash
export APEXID_DATA_ROOT=/mounted/ApexID_Dataset
export APEXID_SPLIT_ROOT="$APEXID_DATA_ROOT/open_set_splits_v2"
export APEXID_OUTPUT_ROOT=/mounted/apexid-results
```

The historical unset split fallback remains `ApexID_Dataset/open_set_splits`
(V1) for compatibility. The new wildlife fine-tuning recipes explicitly default
to repository `open_set_splits_v2/`. Set `APEXID_SPLIT_ROOT` when the frozen
V2 directory is mounted elsewhere.

## Common Commands

Train with the default config:

```bash
python scripts/train.py --config configs/default.yaml
```

Audit BHCA species metadata without changing a split:

```bash
python scripts/prepare_bhca_species_metadata.py \
  --input open_set_splits_v2/train.csv \
  --check-only
```

## Validation Settings

`evaluation.sample_limits` controls whether open-set metric computation uses the
full validation splits or deterministic identity-balanced samples. The default
values are `null`, which means full validation:

```yaml
evaluation:
  sample_limits:
    gallery: null
    known: null
    unknown: null
    mixed: null
```

Set positive integers only when you intentionally want faster approximate
metrics during development. The default open-set loader setup expects `val.csv`
to be `val_known.csv` followed by `val_unknown.csv`; when that is true, the
validator reuses known+unknown embeddings and avoids a separate mixed forward
pass.

`evaluation.top_k` logs `known/top1`, `known/top5`, and `known/top10`.
`evaluation.cmc_ranks` is kept to larger CMC ranks only by default:

```yaml
evaluation:
  top_k: [1, 5, 10]
  cmc_ranks: [20, 50]
```

Ranks that appear in both lists are de-duplicated in code, so duplicate CMC keys
are not logged when the matching `known/top*` metrics already exist.

Debug the configured dataset and label flow:

```bash
python scripts/debug_data_pipeline.py --config configs/default.yaml
```

Calculate configured training mean/std:

```bash
python -m apexid.utils.calculate_mean_std --config configs/default.yaml --split train
```

Run the config test:

```bash
pytest tests/test_default_config.py
```

## Backbone Selection

The default config keeps the OSNet experiment:

```yaml
model:
  backbone:
    name: osnet_x1_0
```

To run TorchVision ConvNeXt instead, change only the backbone name and optional
kwargs:

```yaml
model:
  backbone:
    name: convnext_tiny_pt
    kwargs:
      pretrained: true
```

Available ConvNeXt aliases are `convnext_tiny`, `convnext_small`,
`convnext_base`, `convnext_large`, and pretrained-default `_pt` variants.
All ConvNeXt backbones return the configured embedding dimension and are
normalized by `ApexID.forward`, matching the OSNet path.

DINOv3 ViT aliases are `dinov3_vits16`, `dinov3_vits16plus`, `dinov3_vitb16`,
`dinov3_vitl16`, and pretrained-default `_pt` variants. Keep `embedding_dim`
equal to the ViT hidden size (768 for ViT-B) so the backbone returns native CLS
features. Image size must be a multiple of 16; the cluster recipe uses 512x512.

ConvNeXt V2 aliases are `convnextv2_tiny`, `convnextv2_base`, `convnextv2_large`,
and `_pt` variants. Defaults load ImageNet-22k @ 384 weights. Keep
`embedding_dim` at 1024 for Base so pooled features stay native. The cluster
recipe trains at 440x440 gray3 like the best ConvNeXt Small run.

## Pretrained Fine-Tuning

When the backbone is ImageNet-pretrained (`*_pt` aliases or `kwargs.pretrained:
true`), `scripts/train.py` calls `resolve_training_profile()` and applies the
`finetune` section automatically when `finetune.enabled: auto`.

### Scratch vs pretrained profiles

| Profile | Backbone examples | LR range test | Scheduler | Optimizer LRs |
|---------|-------------------|---------------|-----------|---------------|
| **Scratch** | `osnet_x1_0`, `convnext_large` + `pretrained: false` | **On** by default (`lr_range_test.enabled: true`) | **CyclicLR** (finder sets `base_lr` / `max_lr`) | Single `optimizer.kwargs.lr` |
| **Pretrained** | `osnet_x1_0_pt`, `convnext_base_pt`, `pretrained: true` | **Off** by default (`finetune.lr_range_test.enabled: false`) | **Cosine annealing + warmup** | Separate backbone/head LR and optional weight decay |

The top-level `scheduler` and `lr_range_test` blocks in YAML are kept for scratch
runs. When the finetune profile activates, `resolve_training_profile()` overrides
the effective scheduler and disables the LR finder unless you explicitly re-enable
it under `finetune.lr_range_test`.

### Finetune strategies

Choose with `finetune.strategy`:

| Strategy | Behavior | When to use |
|----------|----------|-------------|
| `full` | Unfreeze backbone + head from epoch 0; differential LRs; cosine decay | MiewID-style full fine-tune from ImageNet weights |
| `head_first` | Run `head_only_epochs` extra head-only warmup epochs with the backbone frozen; then unfreeze all for `training.epochs` real epochs | WildFusion-style stable head warmup before backbone updates |

`head_first` options:

- `head_only_epochs`: extra warmup epochs that train only the head before real
  training starts (default `3`).
- `training.epochs`: real training epochs after the backbone is unfrozen.
- The scheduler, validation, checkpointing, early stopping, and epoch-level
  experiment logging start only after the head-only warmup.
- `unfreeze_top_fraction` is kept in configs for compatibility, but head-first
  warmup freezes the full backbone; leave it at `0.0`.

### Full `finetune` block

```yaml
finetune:
  enabled: auto          # auto | true | false
  strategy: full         # full | head_first
  head_only_epochs: 3    # extra head-only warmup epochs before training.epochs
  unfreeze_top_fraction: 0.0
  optimizer:
    backbone_lr: 1.0e-5
    head_lr: 1.0e-4
    head_lr_multiplier: null   # optional: head_lr = backbone_lr * multiplier
    backbone_weight_decay: null # null falls back to optimizer.kwargs.weight_decay
    head_weight_decay: null     # set 0.0 or another value for a separate head decay
  scheduler:
    name: cosine_annealing_lr
    kwargs:
      T_max: null              # defaults to training.epochs
      eta_min: 1.0e-6
    warmup:
      enabled: true
      epochs: 5
      start_lr: 1.0e-6
  lr_range_test:
    enabled: false             # default for pretrained
    mode: suggest_lr           # optional: run finder, apply LRs to param groups
    apply_to: head
    backbone_lr_divisor: 10.0  # backbone_lr = suggested_max / divisor
```

Set `finetune.enabled: false` to force scratch-style CyclicLR even with a `_pt`
backbone. Set `finetune.enabled: true` to force the finetune profile even on a
random-init backbone.

### Optional LR range test for pretrained

By default, pretrained runs **do not** run the LR finder (fixed finetune LRs are
used instead). To experiment:

```yaml
finetune:
  lr_range_test:
    enabled: true
    mode: suggest_lr
```

This runs the finder once, writes artifacts under `outputs/.../lr_range_test/`,
sets backbone/head LRs from the suggestion, and still uses cosine scheduling —
not CyclicLR.

### OSNet pretrained example

Use `embedding_dim: 512` so ImageNet fc weights load. ArcFace `s` / `m` must be
under `head.kwargs`:

```yaml
model:
  embedding_dim: 512
  input_channels: 3
  backbone:
    name: osnet_x1_0_pt
    kwargs:
      pretrained: true
head:
  name: arcface
  kwargs:
    s: 40.0
    m: 0.3
finetune:
  enabled: auto
  strategy: head_first
  head_only_epochs: 3
```

### Cluster usage

Scratch on cluster (ConvNeXt-Large, no wavelet):

```bash
# configs/cluster_home.yaml — current ConvNeXt Small cluster recipe
sbatch slurm_train_h100.sh
```

DINOv3 ViT-B/16 native CLS (RGB, 512px, head-first):

```bash
CONFIG=configs/cluster_dinov3_vitb16.yaml sbatch slurm_train_h100.sh
```

The DINOv3 weights are gated. Accept the DINOv3 license on Hugging Face and
export `HF_TOKEN` on the cluster before launching.

DINOv3 ViT-B/16 with Sub-center ArcFace (same backbone/data, different head):

```bash
CONFIG=configs/cluster_dinov3_vitb16_subcenter.yaml sbatch slurm_train_h100.sh
```

ConvNeXt V2 Base ImageNet-22k (gray3, 440px, full fine-tune):

```bash
CONFIG=configs/cluster_convnextv2_base.yaml sbatch slurm_train_h100.sh
```

Pretrained on cluster:

```bash
CONFIG=configs/cluster_pretrained.yaml sbatch slurm_train_h100.sh
```

Sub-center ArcFace dynamic A/B candidate:

```bash
CONFIG=configs/cluster_subcenter_dynamic.yaml sbatch slurm_train_h100.sh
```

Quick config check on the login node:

```bash
grep -E "backbone|pretrained|strategy|wavelet|resume_path" configs/cluster_pretrained.yaml
grep -A 6 "^head:" configs/cluster_pretrained.yaml
grep -A 12 "^finetune:" configs/cluster_pretrained.yaml
```

Ready-to-use templates: [`cluster_pretrained.yaml`](cluster_pretrained.yaml),
[`cluster_subcenter_dynamic.yaml`](cluster_subcenter_dynamic.yaml),
[`cluster_dinov3_vitb16.yaml`](cluster_dinov3_vitb16.yaml),
[`cluster_dinov3_vitb16_subcenter.yaml`](cluster_dinov3_vitb16_subcenter.yaml),
[`cluster_convnextv2_base.yaml`](cluster_convnextv2_base.yaml).

### Sub-center margin modes

`arcface_subcenter_dynamic` now supports two margin-source modes:

- `margins_source: uniform` (default): all classes use `margin_default`.
- `margins_source: frequency_bins`: `scripts/train.py` computes per-class margins
  from train-set class counts (MiewID-style), then injects them into
  `head.kwargs.margins` before building the head.

Example:

```yaml
head:
  name: arcface_subcenter_dynamic
  kwargs:
    s: 40.0
    k: 3
    margins_source: frequency_bins
    margin_min: 0.2
    margin_max: 0.3
```

In this mode, rarer classes receive larger margins; frequent classes receive
smaller margins. If `head.kwargs.margins` is provided explicitly, it overrides
the computed values.

## Contributor Notes

- Keep paths relative to the repository root unless the code explicitly expects
  absolute paths.
- When `data.wavelet: true`, set `model.input_channels` to `6` or leave it on an
  auto/null value handled by the model factory, and provide six mean/std values.
- When `data.grayscale: true`, train/val convert RGB to grayscale and stack to
  three identical channels. Keep `model.input_channels: 3` and recompute
  mean/std with the same flag (channels become nearly identical).
- SphereFace2 heads compute their own loss; set `loss.name` to `null` when using
  them.
- Keep output directories under `outputs/` so generated artifacts stay separate
  from source files.
