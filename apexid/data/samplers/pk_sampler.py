from collections.abc import Mapping

import numpy as np
from torch.utils.data import Sampler

from apexid.utils.seed import DEFAULT_SEED, seed_everything


seed_everything(DEFAULT_SEED)


class PKBatchSampler(Sampler):
    """Deterministic P x K batch sampler for animal re-identification.

    Args:
        dataset (Dataset or Mapping): Dataset exposing ``identity_to_indices`` or
            a mapping ``{identity: sequence_of_sample_indices}``.
        num_identities (int): Number of unique identities, ``P``, sampled per
            batch.
        num_instances (int): Number of images, ``K``, sampled for each selected
            identity.
        batches_per_epoch (int, optional): Number of batches yielded by one
            iterator pass. If omitted, it is inferred from dataset size and
            ``drop_last``.
        seed (int): Base random seed. The effective RNG seed is ``seed + epoch``.
        drop_last (bool): Whether inferred epoch length should drop incomplete
            batches.
        shuffle (bool): Whether to randomly sample identities. When ``False``,
            identities are traversed deterministically.

    Returns:
        PKBatchSampler: An iterable sampler that yields lists of dataset row
        indices. Each yielded list has shape ``(P * K,)`` as Python integers.

    Functionality:
        Builds deterministic batches for metric learning. Each yielded batch
        contains ``num_identities`` identities and ``num_instances`` samples per
        identity. Identities are sampled without replacement inside a batch.
        Samples are drawn without replacement when an identity has enough images,
        otherwise with replacement.
    """

    def __init__(
        self,
        dataset,
        num_identities,
        num_instances,
        batches_per_epoch=None,
        seed=DEFAULT_SEED,
        drop_last=True,
        shuffle=True,
    ):
        """Create a deterministic P x K batch sampler.

        Args:
            dataset (Dataset or Mapping): Dataset exposing ``identity_to_indices``
                or a direct mapping from identity to sample indices. Each index
                points to one row in the dataset.
            num_identities (int): Number of identities per batch, ``P``.
            num_instances (int): Number of samples per identity, ``K``.
            batches_per_epoch (int, optional): Number of batches returned by
                ``__iter__``. If ``None``, the value is derived from available
                samples and ``batch_size = P * K``.
            seed (int): Base seed used for deterministic NumPy sampling.
            drop_last (bool): Used only when inferring ``batches_per_epoch``.
            shuffle (bool): If ``True``, sample identities randomly. If
                ``False``, walk identities in deterministic order.

        Returns:
            None: Stores precomputed identity arrays, replacement flags, batch
            size, and epoch length.

        Functionality:
            Validates sampler parameters, converts identity index lists to NumPy
            arrays for efficient sampling, precomputes whether each identity needs
            replacement sampling, and prepares deterministic epoch state.
        """
        if num_identities <= 0:
            raise ValueError("num_identities must be > 0.")
        if num_instances <= 0:
            raise ValueError("num_instances must be > 0.")

        identity_to_indices = self._get_identity_to_indices(dataset)
        identity_to_indices = {
            identity: np.asarray(indices, dtype=np.int64)
            for identity, indices in identity_to_indices.items()
            if len(indices) > 0
        }
        if len(identity_to_indices) == 0:
            raise ValueError("PKBatchSampler requires at least one identity with samples.")
        if len(identity_to_indices) < num_identities:
            raise ValueError(
                "Not enough identities for PK sampling. "
                f"Need at least {num_identities}, found {len(identity_to_indices)}."
            )

        self.dataset = dataset
        self.num_identities = int(num_identities)
        self.num_instances = int(num_instances)
        self.batch_size = self.num_identities * self.num_instances
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.epoch = 0

        self.identity_to_indices = {
            identity: indices.copy()
            for identity, indices in sorted(identity_to_indices.items(), key=lambda item: str(item[0]))
        }
        self.identities = tuple(self.identity_to_indices.keys())
        self._identity_indices = tuple(self.identity_to_indices[identity] for identity in self.identities)
        self._replace_flags = tuple(indices.size < self.num_instances for indices in self._identity_indices)
        self.num_samples = int(sum(indices.size for indices in self._identity_indices))

        if batches_per_epoch is None:
            batches_per_epoch = self.num_samples // self.batch_size
            if not self.drop_last and self.num_samples % self.batch_size:
                batches_per_epoch += 1
            batches_per_epoch = max(int(batches_per_epoch), 1)
        elif batches_per_epoch <= 0:
            raise ValueError("batches_per_epoch must be > 0 when provided.")

        self.batches_per_epoch = int(batches_per_epoch)

    @staticmethod
    def _get_identity_to_indices(dataset):
        """Extract the identity-to-sample-index mapping used by the sampler.

        Args:
            dataset (Dataset or Mapping): Either a mapping shaped like
                ``{identity: [sample_index, ...]}`` or a dataset object with an
                ``identity_to_indices`` attribute of that same shape.

        Returns:
            Mapping: Identity-to-indices mapping. Each value is expected to be a
            non-empty sequence of integer dataset indices before later filtering.

        Functionality:
            Allows the sampler to work with the ApexID dataset class and with
            plain mappings in tests.

        Raises:
            ValueError: If the dataset does not expose ``identity_to_indices``.
            TypeError: If ``identity_to_indices`` exists but is not a mapping.
        """
        if isinstance(dataset, Mapping):
            return dataset
        if not hasattr(dataset, "identity_to_indices"):
            raise ValueError(
                "Dataset must expose identity_to_indices or be an identity-to-indices mapping."
            )
        identity_to_indices = getattr(dataset, "identity_to_indices")
        if not isinstance(identity_to_indices, Mapping):
            raise TypeError("dataset.identity_to_indices must be a mapping.")
        return identity_to_indices

    def set_epoch(self, epoch):
        """Set the epoch used to derive the deterministic RNG state.

        Args:
            epoch (int): Epoch index. The sampler RNG is seeded with
                ``self.seed + epoch``.

        Returns:
            None

        Functionality:
            Lets training loops or distributed wrappers request repeatable but
            different P x K batches for each epoch.
        """
        self.epoch = int(epoch)

    def _rng(self):
        """Create the NumPy random number generator for the current epoch.

        Args:
            None

        Returns:
            numpy.random.Generator: RNG seeded with ``self.seed + self.epoch``.

        Functionality:
            Centralizes random-state creation so the sampler is deterministic for
            a given seed/epoch pair.
        """
        return np.random.default_rng(self.seed + self.epoch)

    def __iter__(self):
        """Yield one epoch of P x K batches.

        Args:
            None

        Yields:
            list[int]: Dataset indices with length ``num_identities *
            num_instances``. The list layout is grouped by selected identity:
            ``K`` sample indices for identity 1, then ``K`` for identity 2, and
            so on.

        Functionality:
            Selects identities, samples ``K`` image indices per identity, yields
            batch index lists for DataLoader, and increments ``self.epoch`` after
            the iterator completes.
        """
        rng = self._rng()
        identity_count = len(self._identity_indices)

        for batch_idx in range(self.batches_per_epoch):
            if self.shuffle:
                selected_positions = rng.choice(
                    identity_count,
                    size=self.num_identities,
                    replace=False,
                )
            else:
                start = (
                    (self.epoch * self.batches_per_epoch + batch_idx)
                    * self.num_identities
                ) % identity_count
                selected_positions = (start + np.arange(self.num_identities)) % identity_count

            batch_indices = []
            for pos in selected_positions:
                pos = int(pos)
                candidate_indices = self._identity_indices[pos]
                sampled = rng.choice(
                    candidate_indices,
                    size=self.num_instances,
                    replace=self._replace_flags[pos],
                )
                batch_indices.extend(int(index) for index in sampled)

            yield batch_indices
        self.epoch += 1

    def __len__(self):
        """Return the number of batches in one sampler epoch.

        Args:
            None

        Returns:
            int: Number of batch index lists yielded by ``__iter__``.
        """
        return self.batches_per_epoch
