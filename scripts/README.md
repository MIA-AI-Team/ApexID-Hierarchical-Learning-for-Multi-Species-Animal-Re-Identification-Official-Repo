# Command-line workflows

Run from the repository root.

```bash
python scripts/debug_data_pipeline.py --config configs/default.yaml --skip-batch
python scripts/train.py --config configs/default.yaml
```

The first command audits identity labels and split membership. Omit `--skip-batch` to load a batch with the configured image preprocessing and sampler. See [experiment presets](../configs/README.md) for other models and the category-weight sweep.

Use `extract_embeddings.py` for trained and pretrained descriptors in the [external evaluation workflow](../docs/evaluation.md). The [data guide](../data/README.md) covers category metadata preparation for new split CSVs.
