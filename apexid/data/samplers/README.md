# apexid/data/samplers

This folder contains DataLoader samplers used to form training batches.

## Main Files

- `pk_sampler.py`: defines `PKBatchSampler`.
- `__init__.py`: exports `PKBatchSampler`.

## How It Connects

`PKBatchSampler` builds P x K batches for re-identification training. Each batch
contains `P` identities and `K` samples per identity, so the configured batch
size is `P * K`.

In `configs/default.yaml`, the default sampler configuration is:

```yaml
dataloader:
  sampler:
    name: pk
    num_identities: 16
    num_instances: 4
```

The sampler reads `dataset.identity_to_indices` from `ApexID_Dataset`, samples
identities without replacement inside a batch, samples images with replacement
only when an identity has fewer than `K` images, and uses `seed + epoch` for
reproducible epoch-specific batches.

## Common Commands

Run sampler-related data tests:

```bash
pytest tests/data
```

Use the sampler directly:

```python
from torch.utils.data import DataLoader
from apexid.data.samplers import PKBatchSampler

sampler = PKBatchSampler(
    dataset=train_dataset,
    num_identities=8,
    num_instances=4,
    seed=42,
)
loader = DataLoader(train_dataset, batch_sampler=sampler)
```

## Contributor Notes

- Keep `set_epoch(epoch)` deterministic; the trainer calls it before each epoch.
- Validate sampler parameters early so configuration mistakes fail before
  training starts.
- Keep replacement behavior for low-sample identities covered by tests.
