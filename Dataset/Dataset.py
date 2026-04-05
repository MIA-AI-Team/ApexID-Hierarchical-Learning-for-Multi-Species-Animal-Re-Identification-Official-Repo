from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import os


class ApexID_Dataset(Dataset):
    def __init__(self, data_csv, transform=None, label_to_idx=None):
        super().__init__()
        self.data_csv = data_csv.reset_index(drop=True)
        self.transform = transform

        if label_to_idx is None:
            unique_identities = sorted(self.data_csv['identity'].dropna().unique())
            label_to_idx = {label: idx for idx, label in enumerate(unique_identities)}

        self.label_to_idx = dict(label_to_idx)
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

    @staticmethod
    def build_label_mapping(data_csv):
        unique_identities = sorted(data_csv['identity'].dropna().unique())
        return {label: idx for idx, label in enumerate(unique_identities)}

    def __len__(self):
        return len(self.data_csv)

    def __getitem__(self, idx):
        img_path = self.data_csv.iloc[idx]['path']
        img_path = os.path.join('animalCLEF26', img_path)

        label_str = self.data_csv.iloc[idx]['identity']
        if label_str not in self.label_to_idx:
            raise KeyError(
                f"Identity '{label_str}' is missing from the provided label_to_idx mapping."
            )

        label = self.label_to_idx[label_str]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(label, dtype=torch.long)

        return img, label


if __name__ == "__main__":
    data_csv = pd.read_csv('animalCLEF26/metadata.csv')

    data_csv.dropna(inplace=True)

    transform = transforms.Compose([
        transforms.Resize((440, 440)),
        transforms.ToTensor(),
    ])

    label_to_idx = ApexID_Dataset.build_label_mapping(data_csv)
    dataset = ApexID_Dataset(data_csv, transform, label_to_idx=label_to_idx)
    print("Dataset length:", len(dataset))

    img, label = dataset[0]
    print("Image shape:", img.shape)
    print("Encoded label:", label)
    print("Original identity:", dataset.idx_to_label[label.item()])
