from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
import random
from torchvision import transforms
import pandas as pd
import os
from collections import defaultdict


class ApexID_Dataset(Dataset):
    def __init__(
        self,
        data_csv,
        transform=None,
        label_to_idx=None,
        root_dir=None,
        allow_unknown_identity=False,
        unknown_label=-100,
        metrics_label_map=None,
        image_mode='RGB',
        path_column='path',
        identity_column='identity',
    ):
        """Initialize the dataset and build classification and metric label maps.

        Args:
            data_csv (pandas.DataFrame or str): DataFrame or CSV path containing the
                dataset metadata.
            transform (callable, optional): Transform applied to each loaded image.
            label_to_idx (dict, optional): Mapping from known identity labels to
                classification indices.
            root_dir (str, optional): Root directory used to resolve relative image
                paths.
            allow_unknown_identity (bool, optional): Whether identities missing from
                ``label_to_idx`` are allowed and mapped to ``unknown_label``.
            unknown_label (int, optional): Classification label assigned to unknown
                identities during open-set evaluation.
            metrics_label_map (dict, optional): Mapping used for retrieval and
                verification metrics.
            image_mode (str, optional): PIL image mode used when opening images.
            path_column (str, optional): Column name containing image paths after
                column standardization.
            identity_column (str, optional): Column name containing identity labels
                after column standardization.

        Returns:
            None
        """
        super().__init__()
        self.data_csv = self._standardize_columns(data_csv).reset_index(drop=True)
        self.transform = transform
        self.root_dir = root_dir
        self.allow_unknown_identity = allow_unknown_identity
        self.unknown_label = unknown_label
        self.image_mode = image_mode
        self.path_column = path_column
        self.identity_column = identity_column
        self.identity_to_indices = {}

        if self.path_column not in self.data_csv.columns:
            raise KeyError(f"Missing required column: '{self.path_column}'")
        if self.identity_column not in self.data_csv.columns:
            raise KeyError(f"Missing required column: '{self.identity_column}'")

        # In open-set training, this should come from TRAIN only.
        if label_to_idx is None:
            unique_identities = sorted(self.data_csv[self.identity_column].dropna().unique())
            label_to_idx = {label: idx for idx, label in enumerate(unique_identities)}

        self.label_to_idx = dict(label_to_idx)
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        # Used for retrieval / verification metrics.
        # This should usually be a union map across train + val so unknown val IDs stay distinct.
        if metrics_label_map is None:
            metrics_label_map = self.build_label_mapping(self.data_csv, identity_column=self.identity_column)
        self.metrics_label_map = dict(metrics_label_map)
        self.metric_idx_to_label = {idx: label for label, idx in self.metrics_label_map.items()}

        self._build_identity_index()

    @staticmethod
    def _standardize_columns(data_csv):
        """Normalize common dataset column names to the expected schema.

        Args:
            data_csv (pandas.DataFrame or str): DataFrame or CSV path to normalize.

        Returns:
            pandas.DataFrame: Copy of the input data with recognized column names
            renamed to the project's canonical names.
        """
        if not isinstance(data_csv, pd.DataFrame):
            data_csv = pd.read_csv(data_csv)

        df = data_csv.copy()
        rename_map = {}
        for col in df.columns:
            col_norm = str(col).strip().lower()
            if col_norm in {'path', 'image_path', 'img_path', 'filepath', 'file_path'}:
                rename_map[col] = 'path'
            elif col_norm in {'identity', 'id', 'individual_id', 'label'}:
                rename_map[col] = 'identity'
            elif col_norm in {'species', 'animal', 'class_name'}:
                rename_map[col] = 'species'
        df = df.rename(columns=rename_map)
        return df

    @staticmethod
    def build_label_mapping(data_csv, identity_column='identity'):
        """Build a label-to-index mapping from a single dataset split.

        Args:
            data_csv (pandas.DataFrame or str): DataFrame or CSV path containing
                identity labels.
            identity_column (str, optional): Column name holding the identity values.

        Returns:
            dict: Mapping from each unique identity label to a contiguous integer
            index.
        """
        df = ApexID_Dataset._standardize_columns(data_csv)
        unique_identities = sorted(df[identity_column].dropna().unique())
        return {label: idx for idx, label in enumerate(unique_identities)}

    @staticmethod
    def build_label_mapping_union(train_df, val_df, identity_column='identity'):
        """Build a shared label mapping across train and validation splits.

        Args:
            train_df (pandas.DataFrame or str): Training split metadata.
            val_df (pandas.DataFrame or str): Validation split metadata.
            identity_column (str, optional): Column name holding the identity values.

        Returns:
            dict: Mapping from every identity seen in either split to a contiguous
            integer index.
        """
        train_df = ApexID_Dataset._standardize_columns(train_df)
        val_df = ApexID_Dataset._standardize_columns(val_df)
        train_ids = set(train_df[identity_column].dropna().unique().tolist())
        val_ids = set(val_df[identity_column].dropna().unique().tolist())
        all_ids = sorted(train_ids | val_ids)
        return {label: idx for idx, label in enumerate(all_ids)}
    
    def _build_identity_index(self):
        """Build the sampling index for identities known to the classifier.

        Args:
            None

        Returns:
            None
        """
        identity_to_indices = defaultdict(list)

        for idx in range(len(self.data_csv)):
            identity = self.data_csv.iloc[idx][self.identity_column]

            # Only include identities that are valid training classes
            if identity in self.label_to_idx:
                identity_to_indices[identity].append(idx)

        self.identity_to_indices = dict(identity_to_indices)
        self.train_identities = sorted(self.identity_to_indices.keys())

    def __len__(self):
        """Return the number of samples in the dataset.

        Args:
            None

        Returns:
            int: Number of rows available in ``self.data_csv``.
        """
        return len(self.data_csv)

    def _resolve_img_path(self, img_path):
        """Resolve an image path against the dataset root when needed.

        Args:
            img_path (str): Raw image path value stored in the dataset metadata.

        Returns:
            str: Existing absolute or resolved path when found, otherwise the
            original path string.
        """
        img_path = str(img_path)

        if os.path.exists(img_path):
            return img_path

        if self.root_dir is not None and not os.path.isabs(img_path):
            candidate = os.path.join(self.root_dir, img_path)
            if os.path.exists(candidate):
                return candidate

        if self.root_dir is not None:
            candidate = os.path.join(self.root_dir, os.path.basename(img_path))
            if os.path.exists(candidate):
                return candidate

        return img_path

    def __getitem__(self, idx):
        """Load one dataset sample and return image and label tensors.

        Args:
            idx (int): Row index of the sample to retrieve.

        Returns:
            tuple: Three-item tuple containing the transformed image, the
            classification label tensor, and the metric label tensor.
        """
        row = self.data_csv.iloc[idx]

        img_path = self._resolve_img_path(row[self.path_column])
        identity = row[self.identity_column]

        # Classification label: known IDs only. Unknown val IDs get unknown_label.
        if identity in self.label_to_idx:
            cls_label = self.label_to_idx[identity]
        else:
            if not self.allow_unknown_identity:
                raise KeyError(
                    f"Identity '{identity}' is missing from label_to_idx. "
                    f"Pass the TRAIN mapping, and set allow_unknown_identity=True for open-set validation."
                )
            cls_label = self.unknown_label

        # Metric label: every identity must remain distinct for retrieval / verification.
        if identity not in self.metrics_label_map:
            raise KeyError(
                f"Identity '{identity}' is missing from metrics_label_map. "
                f"Use a union mapping across train + val identities."
            )
        metric_label = self.metrics_label_map[identity]

        img = Image.open(img_path).convert(self.image_mode)

        if self.transform is not None:
            img = self.transform(img)

        cls_label = torch.tensor(cls_label, dtype=torch.long)
        metric_label = torch.tensor(metric_label, dtype=torch.long)

        return img, cls_label, metric_label
    

class PKBatchSampler(Sampler):
    """
    P x K sampler:
        - selects P identities
        - selects K samples per identity
        - batch size = P * K

    Works best for train datasets where all sampled identities are known classes.
    """

    def __init__(
        self,
        dataset,
        num_identities,
        num_instances,
        batches_per_epoch= None,
        seed= 42,
        drop_last = True,
    ):
        """Initialize a P x K batch sampler over identity-grouped samples.

        Args:
            dataset (ApexID_Dataset): Dataset exposing ``identity_to_indices`` for
                identity-aware sampling.
            num_identities (int): Number of identities sampled per batch.
            num_instances (int): Number of samples drawn per identity.
            batches_per_epoch (int, optional): Fixed number of batches to yield per
                epoch. If ``None``, an estimate is derived from dataset size.
            seed (int, optional): Seed controlling deterministic sampling order.
            drop_last (bool, optional): Whether to drop incomplete batches.

        Returns:
            None
        """
        if not hasattr(dataset, "identity_to_indices")or len(dataset.identity_to_indices) == 0:
            raise ValueError("Dataset must have identity_to_indices. Use build_identity_index=True.")

        self.dataset = dataset
        self.num_identities = num_identities   # P
        self.num_instances = num_instances     # K
        self.batch_size = num_identities * num_instances
        self.seed = seed
        self.drop_last = drop_last

        # Keep only identities with at least 1 sample
        self.identity_to_indices = {
            identity: indices
            for identity, indices in dataset.identity_to_indices.items()
            if len(indices) > 0
        }

        self.identities = list(self.identity_to_indices.keys())

        if len(self.identities) < self.num_identities:
            raise ValueError(
                f"Not enough identities for PK sampling. "
                f"Need at least {self.num_identities}, found {len(self.identities)}."
            )

        if batches_per_epoch is None:
            est = len(dataset) // self.batch_size
            self.batches_per_epoch = max(est, 1)
        else:
            self.batches_per_epoch = batches_per_epoch

    def __iter__(self):
        """Yield batches of dataset indices following the P x K sampling scheme.

        Args:
            None

        Returns:
            iterator: Iterator yielding lists of dataset indices for each batch.
        """
        rng = random.Random(self.seed)

        for _ in range(self.batches_per_epoch):
            selected_identities = rng.sample(self.identities, self.num_identities)

            batch_indices = []

            for identity in selected_identities:
                candidate_indices = self.identity_to_indices[identity]

                if len(candidate_indices) >= self.num_instances:
                    sampled = rng.sample(candidate_indices, self.num_instances)
                else:
                    # sample with replacement if identity has fewer than K images
                    sampled = [rng.choice(candidate_indices) for _ in range(self.num_instances)]

                batch_indices.extend(sampled)

            if len(batch_indices) == self.batch_size:
                yield batch_indices
            elif not self.drop_last and len(batch_indices) > 0:
                yield batch_indices

    def __len__(self):
        """Return the number of batches produced in one epoch.

        Args:
            None

        Returns:
            int: Number of batches yielded per epoch.
        """
        return self.batches_per_epoch


if __name__ == "__main__":
    train_csv = pd.read_csv('data_splits/train.csv')
    val_csv = pd.read_csv('data_splits/val.csv')

    transform = transforms.Compose([
        transforms.Resize((440, 440)),
        transforms.ToTensor(),
    ])

    train_label_to_idx = ApexID_Dataset.build_label_mapping(train_csv)
    union_label_map = ApexID_Dataset.build_label_mapping_union(train_csv, val_csv)

    train_dataset = ApexID_Dataset(
        train_csv,
        transform=transform,
        label_to_idx=train_label_to_idx,
        metrics_label_map=union_label_map,
        allow_unknown_identity=False,
    )

    val_dataset = ApexID_Dataset(
        val_csv,
        transform=transform,
        label_to_idx=train_label_to_idx,
        metrics_label_map=union_label_map,
        allow_unknown_identity=True,
        unknown_label=-100,
    )

    print('Train length:', len(train_dataset))
    print('Val length:', len(val_dataset))

    img, cls_label, metric_label = val_dataset[0]
    print('Image shape:', img.shape)
    print('Classification label:', cls_label.item())
    print('Metric label:', metric_label.item())
