from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import os


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
        super().__init__()
        self.data_csv = self._standardize_columns(data_csv).reset_index(drop=True)
        self.transform = transform
        self.root_dir = root_dir
        self.allow_unknown_identity = allow_unknown_identity
        self.unknown_label = unknown_label
        self.image_mode = image_mode
        self.path_column = path_column
        self.identity_column = identity_column

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

    @staticmethod
    def _standardize_columns(data_csv):
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
        df = ApexID_Dataset._standardize_columns(data_csv)
        unique_identities = sorted(df[identity_column].dropna().unique())
        return {label: idx for idx, label in enumerate(unique_identities)}

    @staticmethod
    def build_label_mapping_union(train_df, val_df, identity_column='identity'):
        train_df = ApexID_Dataset._standardize_columns(train_df)
        val_df = ApexID_Dataset._standardize_columns(val_df)
        train_ids = set(train_df[identity_column].dropna().unique().tolist())
        val_ids = set(val_df[identity_column].dropna().unique().tolist())
        all_ids = sorted(train_ids | val_ids)
        return {label: idx for idx, label in enumerate(all_ids)}

    def __len__(self):
        return len(self.data_csv)

    def _resolve_img_path(self, img_path):
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