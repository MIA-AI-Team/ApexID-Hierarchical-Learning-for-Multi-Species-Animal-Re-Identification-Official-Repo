# configs

This folder stores YAML experiment configuration. The default project entry
point is `default.yaml`.

## Main Files

- `default.yaml`: complete training configuration for the default ApexID
  experiment. It defines dataset paths, transforms, sampler settings, model
  shape, backbone, head, loss, optimizer, scheduler, LR range test, training
  behavior, validation metrics, embedding plots, checkpoints, early stopping,
  and experiment tracking.

## How It Connects

Most runnable commands load this file through `apexid.utils.config.load_config`.
`scripts/train.py` builds the full training stack from it, and tests assert that
the default values remain internally consistent.

Important sections:

- `experiment`: run name and output directory.
- `data`: train/validation CSVs, image root, image size, wavelet mode, mean/std.
- `dataloader`: batch size, workers, and P x K sampler parameters.
- `model`: ApexID embedding dimensions, input channels, and backbone selection.
- `head` and `loss`: margin head and external objective configuration.
- `optimizer`, `scheduler`, and `lr_range_test`: learning-rate behavior.
- `training` and `validation`: epoch loop, label ignore indices, and metrics.
- `checkpoint` and `early_stopping`: resumable latest checkpoint and best-model
  artifacts.
- `tracking`: W&B logging settings.

## Common Commands

Train with the default config:

```bash
python scripts/train.py --config configs/default.yaml
```

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

## Contributor Notes

- Keep paths relative to the repository root unless the code explicitly expects
  absolute paths.
- When `data.wavelet: true`, set `model.input_channels` to `6` or leave it on an
  auto/null value handled by the model factory, and provide six mean/std values.
- SphereFace2 heads compute their own loss; set `loss.name` to `null` when using
  them.
- Keep output directories under `outputs/` so generated artifacts stay separate
  from source files.
