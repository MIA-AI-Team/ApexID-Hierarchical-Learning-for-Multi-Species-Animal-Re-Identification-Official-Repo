# Utilities

`config.py` loads YAML presets, merges a relative `extends` file, and expands `${VAR}` or `${VAR:-fallback}` in resolved values. Child mappings merge recursively. Child lists and scalar values replace their parent values.

`device.py` and `seed.py` configure PyTorch devices, random seeds, and data-loader generators. `provenance.py` records the Git revision, environment, and hashes of run inputs.
