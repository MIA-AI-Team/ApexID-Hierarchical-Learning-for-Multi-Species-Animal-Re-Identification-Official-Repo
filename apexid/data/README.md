# Data loading and partitioning

| Module | Role |
| --- | --- |
| [`datasets/`](datasets) | Loads segmented RGB images and maps dataset-scoped identities into classification and retrieval labels. |
| [`samplers/`](samplers) | Draws identity-uniform P × K training batches, with replacement for identities with fewer than K samples. |
| [`transforms.py`](transforms.py) | Applies checkpoint-specific resizing and normalization, plus the training augmentations. |
| [`species_metadata.py`](species_metadata.py) | Harmonizes training categories and builds the identity-to-category mapping. |
| [`split_v2.py`](split_v2.py) | Groups observations, generates the development partitions, and validates frozen manifests. |
| [`encounters.py`](encounters.py) | Parses Bristol gorilla sessions and ELPephants encounter dates for external evaluation. |

See the [dataset guide](../../data/README.md) for inputs and commands.
