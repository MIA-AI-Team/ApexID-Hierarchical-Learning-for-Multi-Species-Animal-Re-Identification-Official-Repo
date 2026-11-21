# Embedding models

`ApexID` wraps the paper's three image encoders and returns L2-normalized embeddings. The ConvNeXt-Small encoder has a learned 2048-dimensional projection. MIEW-ID v3 and MegaDescriptor-L retain their native 2152- and 1536-dimensional outputs.

`model_factory.py` builds an encoder and the requested training head from an experiment config. Checkpoint evaluation builds the encoder alone. See the [paper presets](../../configs/README.md) for preprocessing and checkpoint settings.
