# tests/utils

This folder tests shared utility modules.

## Main Files

- `test_calculate_mean_std.py`: CLI/helper behavior for normalization statistics.
- `test_embedding_umap.py`: UMAP plot and metadata generation behavior.
- `test_lr_range_test.py`: LR range test result and artifact behavior.

## How It Connects

These tests protect `apexid.utils`, which supports config loading, statistics,
training diagnostics, embedding visualization, and reproducibility helpers used
throughout the project.

## Common Commands

Run utility tests:

```bash
pytest tests/utils
```

Run mean/std CLI tests:

```bash
pytest tests/utils/test_calculate_mean_std.py
```

Run UMAP tests:

```bash
pytest tests/utils/test_embedding_umap.py
```

Run LR range test coverage:

```bash
pytest tests/utils/test_lr_range_test.py
```

## Contributor Notes

- Keep generated test artifacts in temporary directories.
- Add tests when utility command-line options or output artifact schemas change.
- Keep optional dependency behavior explicit when utilities require UMAP,
  matplotlib, or similar packages.
