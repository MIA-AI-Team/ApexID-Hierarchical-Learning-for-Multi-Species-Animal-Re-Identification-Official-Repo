# tests/data

This folder tests data loading, preprocessing, and data statistics behavior.

## Main Files

- `test_dataset.py`: CSV/DataFrame loading, column standardization, label maps,
  unknown validation identities, and sample return contracts.
- `test_dwt_timing.py`: wavelet channel generation behavior and timing-related
  expectations.
- `test_make_open_set_splits.py`: clean open-set split routing, identity
  disjointness, output cleanup, and row-accounting invariants.
- `test_mean_std.py`: dataset mean/std calculation behavior.

## How It Connects

These tests protect `apexid.data`, especially `ApexID_Dataset`, transform
helpers, wavelet channels, and normalization-statistics logic used by training
and the mean/std and split-generation CLIs.

## Common Commands

Run all data tests:

```bash
pytest tests/data
```

Run a specific dataset test file:

```bash
pytest tests/data/test_dataset.py
```

Run the mean/std utility coverage from the data side:

```bash
pytest tests/data/test_mean_std.py
```

## Contributor Notes

- Update these tests when accepted CSV aliases, label-map rules, transform shape,
  or wavelet-channel behavior changes.
- Keep fixtures small; tests should not require a full real dataset.
