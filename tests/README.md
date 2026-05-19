# tests

This folder contains the pytest suite for ApexID. Tests are organized by the
package or script area they validate.

## Main Files

- `conftest.py`: shared pytest setup and repository import-path handling.
- `test_default_config.py`: validates assumptions in `configs/default.yaml`.
- `data/`: dataset, transform, DWT, and mean/std tests.
- `engine/`: training, validation, checkpoint, early-stopping, label-contract,
  and W&B logger tests.
- `evaluation/`: re-identification metric tests.
- `models/`: model input-channel and head behavior tests.
- `scripts/`: CLI/script behavior tests.
- `utils/`: utility tests for statistics, UMAP plotting, and LR range testing.

## How It Connects

The suite protects the config-driven training path. When changing source code,
run the nearest targeted test folder first, then run broader tests if the change
touches shared behavior.

## Common Commands

Run all tests:

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

## Contributor Notes

- Keep tests close to the behavior they protect.
- Prefer small deterministic fixtures over relying on full training runs.
- When changing config keys, update `test_default_config.py` and relevant folder
  READMEs.
