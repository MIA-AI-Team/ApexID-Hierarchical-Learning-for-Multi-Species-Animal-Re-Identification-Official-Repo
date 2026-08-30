# scripts

This folder contains command-line entry points for project workflows. Scripts
are thin orchestration layers around the `apexid/` package.

## Main Files

- `train.py`: trains ApexID from a YAML config. It builds datasets, transforms,
  samplers, model components, optimizer, scheduler, validator, checkpointing,
  early stopping, optional W&B logging, and optional embedding UMAP plots.
- `debug_data_pipeline.py`: audits CSV paths, identity mappings, known/unknown
  validation labels, sampler shape, and optionally one transformed DataLoader
  batch.
- `repro_smoke.py`: fully offline synthetic config, model-forward, checkpoint,
  online-assignment, and evaluator-metric smoke test.
- `evaluate.py`: placeholder file; no CLI behavior is implemented yet.
- `extract_embeddings.py`: placeholder file; no CLI behavior is implemented yet.
- `infer.py`: placeholder file; no CLI behavior is implemented yet.

## How It Connects

Scripts load `configs/default.yaml` by default and call package-level builders in
`apexid.data`, `apexid.models`, `apexid.engine`, and `apexid.utils`. Keep shared
logic in the package so tests can import it directly.

## Common Commands

Train:

```bash
python scripts/train.py --config configs/default.yaml
```

Run the offline reproduction smoke without private data or downloads:

```bash
python scripts/repro_smoke.py --output /tmp/apexid-repro-smoke.json
```

Debug the data pipeline:

```bash
python scripts/debug_data_pipeline.py --config configs/default.yaml
```

Print more row examples during debugging:

```bash
python scripts/debug_data_pipeline.py --config configs/default.yaml --max-examples 50
```

Skip loading a transformed batch:

```bash
python scripts/debug_data_pipeline.py --config configs/default.yaml --skip-batch
```

Run script tests:

```bash
pytest tests/scripts
```

## Contributor Notes

- Keep CLIs small: parse arguments, load config, call tested package functions.
- Add tests under `tests/scripts/` when a script gains behavior.
- Do not document placeholder scripts as runnable until they have argument
  parsing and implemented behavior.
