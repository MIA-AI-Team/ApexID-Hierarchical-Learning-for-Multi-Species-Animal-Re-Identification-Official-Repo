# tests/models

This folder tests model construction and head behavior.

## Main Files

- `test_wildlife_backbones.py`: MiewID and MegaDescriptor loading contracts,
  strict embedding-weight restoration, gradient flow, normalized outputs,
  offline checkpoint restoration, and fine-tuning recipe preprocessing.

- `test_input_channels.py`: model/input-channel behavior, especially RGB versus
  wavelet six-channel inputs.
- `test_convnext.py`: TorchVision ConvNeXt registration, embedding shape,
  pretrained aliases, and six-channel input handling.
- `test_sphereface2_focal.py`: SphereFace2 focal head behavior.

## How It Connects

These tests protect `apexid.models`, including model factories, OSNet and
ConvNeXt backbone input channels, and trainable heads used by `scripts/train.py`.

## Common Commands

Run all model tests:

```bash
pytest tests/models
```

Run input-channel tests:

```bash
pytest tests/models/test_input_channels.py
```

Run SphereFace2 focal tests:

```bash
pytest tests/models/test_sphereface2_focal.py
```

## Contributor Notes

- Add tests when registering new backbones or heads.
- Keep config-driven input-channel behavior covered when changing wavelet mode.
- Test both training-time forward behavior and validation-safe logits when a
  head exposes `predict_logits`.
