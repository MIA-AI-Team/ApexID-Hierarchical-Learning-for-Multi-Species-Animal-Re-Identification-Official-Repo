import torch
from torch.utils.data import Dataset
import random
import pandas as pd
import os
from collections import defaultdict
import cv2
import numpy as np

try:
    from apexid.utils.seed import DEFAULT_SEED, seed_everything
except ImportError:
    DEFAULT_SEED = 42

    def seed_everything(seed=DEFAULT_SEED, deterministic=True):
        """Fallback seed initializer used when ``apexid.utils.seed`` is unavailable.

        Args:
            seed (int): Seed value applied to Python ``random``, NumPy, and
                PyTorch RNGs.
            deterministic (bool): Kept for API compatibility with the shared
                seed helper. This fallback does not use the flag directly.

        Returns:
            int: The seed value that was applied.

        Functionality:
            Sets CPU and CUDA RNG state so dataset-side random behavior is
            reproducible even if the package-level seed utility cannot be
            imported.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        return seed

seed_everything(DEFAULT_SEED)

class ApexID_Dataset(Dataset):
    """CSV-backed animal re-identification dataset.

    Each row describes one image and one animal identity. Images are loaded with
    OpenCV as RGB arrays with shape ``(height, width, 3)`` before optional
    Albumentations transforms. ``__getitem__`` returns the image plus two label
    tensors: one for classification heads and one for retrieval/verification
    metrics.
    """

    def __init__(
        self,
        data_csv,
        transform=None,
        label_to_idx=None,
        root_dir=None,
        allow_unknown_identity=False,
        unknown_label=-100,
        metrics_label_map=None,
        path_column='path',
        identity_column='identity',
    ):
        """Initialize the dataset and build label maps.

        Args:
            data_csv (pandas.DataFrame or str): Dataset metadata with shape
                ``(num_samples, num_columns)`` or a CSV path. Required columns
                after standardization are ``path`` and ``identity``.
            transform (callable, optional): Transform applied to each loaded RGB
                image. It must accept ``image`` with shape ``(height, width, 3)``
                and return a mapping containing transformed ``image``. The
                expected tensor output shape is usually
                ``(channels, configured_height, configured_width)``.
            label_to_idx (dict, optional): Mapping from known identity labels to
                contiguous classification indices. If omitted, the mapping is
                built from identities in ``data_csv``.
            root_dir (str, optional): Root directory used to resolve relative
                image paths.
            allow_unknown_identity (bool, optional): Whether identities missing from
                ``label_to_idx`` are allowed and mapped to ``unknown_label``.
            unknown_label (int, optional): Classification label assigned to unknown
                identities during open-set evaluation.
            metrics_label_map (dict, optional): Mapping used for retrieval and
                verification metrics.
            path_column (str, optional): Column name containing image paths after
                column standardization.
            identity_column (str, optional): Column name containing identity labels
                after column standardization.

        Returns:
            None: The dataset stores metadata, label maps, and
            ``identity_to_indices`` for P x K sampling.

        Functionality:
            Standardizes CSV columns, validates required fields, builds
            classification labels for trainable heads, builds metric labels for
            retrieval evaluation, and indexes sample rows by identity.
        """
        super().__init__()
        self.data_csv = self._standardize_columns(data_csv).reset_index(drop=True)
        self.transform = transform
        self.root_dir = root_dir
        self.allow_unknown_identity = allow_unknown_identity
        self.unknown_label = unknown_label
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
            data_csv (pandas.DataFrame or str): DataFrame with shape
                ``(num_samples, num_columns)`` or a CSV path. Common aliases such
                as ``image_path`` and ``individual_id`` are accepted.

        Returns:
            pandas.DataFrame: Copy of the input with the same rows and normalized
            column names: ``path`` for image paths, ``identity`` for animal IDs,
            and ``species`` when a species-like column is present.

        Functionality:
            Keeps all original data values unchanged and only renames recognized
            metadata columns to the names used by the dataset code.
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
            data_csv (pandas.DataFrame or str): Metadata table with shape
                ``(num_samples, num_columns)`` or a CSV path containing identity
                labels.
            identity_column (str, optional): Column name holding the identity values.

        Returns:
            dict: Mapping from each unique identity label to a contiguous integer
            index.

        Functionality:
            Sorts non-null identity labels and assigns deterministic integer class
            IDs, which are used by classification heads during training.
        """
        df = ApexID_Dataset._standardize_columns(data_csv)
        unique_identities = sorted(df[identity_column].dropna().unique())
        return {label: idx for idx, label in enumerate(unique_identities)}

    @staticmethod
    def build_label_mapping_union(train_df, val_df, identity_column='identity'):
        """Build a shared label mapping across train and validation splits.

        Args:
            train_df (pandas.DataFrame or str): Training split metadata with shape
                ``(num_train_samples, num_columns)`` or a CSV path.
            val_df (pandas.DataFrame or str): Validation split metadata with shape
                ``(num_val_samples, num_columns)`` or a CSV path.
            identity_column (str, optional): Column name holding the identity values.

        Returns:
            dict: Mapping from every identity seen in either split to a contiguous
            integer index.

        Functionality:
            Creates a deterministic union identity map so metric labels remain
            distinct for open-set validation identities that are not classification
            classes during training.
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
            None: Populates ``self.identity_to_indices`` as
            ``{identity: [row_index, ...]}`` and ``self.train_identities`` as a
            sorted identity list.

        Functionality:
            Filters out unknown/open-set identities and keeps only rows whose
            identity exists in ``label_to_idx``. The resulting mapping is consumed
            by P x K batch samplers.
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
            int: Number of rows available in ``self.data_csv``. This is the
            first dimension of the metadata table, ``num_samples``.
        """
        return len(self.data_csv)

    def _resolve_img_path(self, img_path):
        """Resolve an image path against the dataset root when needed.

        Args:
            img_path (str): Raw image path value stored in the dataset metadata.
                It may be absolute, relative to ``root_dir``, or a filename.

        Returns:
            str: Existing absolute or resolved path when found, otherwise the
            original path string.

        Functionality:
            Checks the path directly, then ``root_dir / img_path``, then
            ``root_dir / basename(img_path)``. Returning the unresolved path lets
            OpenCV produce a clear missing-file error later.
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

    @staticmethod
    def _load_image_rgb(img_path):
        """Load a single image from disk in RGB channel order.

        Args:
            img_path (str): Path to an image readable by OpenCV.

        Returns:
            numpy.ndarray: RGB image with shape ``(height, width, 3)`` and dtype
            usually ``uint8``. Pixel values are in ``[0, 255]`` before transform
            normalization.

        Functionality:
            Uses ``cv2.imread`` to load BGR data from disk and converts it to RGB
            so the rest of the pipeline sees a consistent channel order.

        Raises:
            FileNotFoundError: If OpenCV cannot read the image path.
        """
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image with OpenCV: {img_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx):
        """Load one dataset sample and return image and label tensors.

        Args:
            idx (int): Row index of the sample to retrieve, in
                ``[0, len(self) - 1]``.

        Returns:
            tuple: ``(image, cls_label, metric_label)`` where ``image`` is either
            an RGB NumPy array with shape ``(height, width, 3)`` when no transform
            is set, or a transformed tensor/array such as
            ``(channels, configured_height, configured_width)``. ``cls_label`` and
            ``metric_label`` are scalar ``torch.long`` tensors with shape ``()``.

        Functionality:
            Resolves and loads the image, maps the identity to a trainable
            classification label and a metric label, applies the optional
            transform, and returns all three items for training or evaluation.
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

        img = self._load_image_rgb(img_path)

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        cls_label = torch.tensor(cls_label, dtype=torch.long)
        metric_label = torch.tensor(metric_label, dtype=torch.long)

        return img, cls_label, metric_label
    
