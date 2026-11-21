# ApexID package

- `data` loads paper split manifests, transforms RGB foregrounds, and samples identities in P × K batches.
- `models` builds the three paper encoders and their training heads.
- `losses` implements the ApexID hierarchical and baseline objectives.
- `engine` runs training and validation and saves resumable checkpoints.
- `evaluation` provides feature extraction and retrieval and discovery metrics.
- `utils` resolves experiment configurations, configures devices and seeds, and records run provenance.

Run the workflows from the repository root. See the [project README](../README.md) for commands.
