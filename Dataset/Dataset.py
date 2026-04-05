from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import os


class ApexID_Dataset(Dataset):
    def __init__(self, data_csv, transform=None):
        super().__init__()
        self.data_csv = data_csv.reset_index(drop=True)
        self.transform = transform

        unique_identities = sorted(self.data_csv['identity'].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_identities)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

    def __len__(self):
        return len(self.data_csv)

    def __getitem__(self, idx):
        img_path = self.data_csv.iloc[idx]['path']
        img_path = os.path.join('animalCLEF26', img_path)

        label_str = self.data_csv.iloc[idx]['identity']
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

    dataset = ApexID_Dataset(data_csv, transform)
    print("Dataset length:", len(dataset))

    img, label = dataset[0]
    print("Image shape:", img.shape)
    print("Encoded label:", label)
    print("Original identity:", dataset.idx_to_label[label.item()])