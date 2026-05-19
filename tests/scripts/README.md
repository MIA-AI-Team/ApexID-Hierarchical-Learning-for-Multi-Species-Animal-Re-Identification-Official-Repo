# tests/scripts

This folder tests command-line script behavior without requiring full production
training runs.

## Main Files

- `test_debug_data_pipeline.py`: data-pipeline debug script behavior.
- `test_train_script.py`: train script construction and orchestration behavior.

## How It Connects

These tests protect `scripts/train.py` and `scripts/debug_data_pipeline.py`.
They make sure the scripts keep working as thin wrappers around tested package
logic.

## Common Commands

Run script tests:

```bash
pytest tests/scripts
```

Run train script tests:

```bash
pytest tests/scripts/test_train_script.py
```

Run debug script tests:

```bash
pytest tests/scripts/test_debug_data_pipeline.py
```

## Contributor Notes

- Add tests when a placeholder script becomes a real CLI.
- Keep tests fast by mocking heavy training/data-loading behavior where possible.
- Make sure new CLI options are covered by parsing or integration-style tests.
