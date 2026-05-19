import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from apexid.engine.trainer import Trainer


class TwoClassHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_features = 2
        self.linear = nn.Linear(2, 2)

    def forward(self, embeddings, labels):
        return self.linear(embeddings)

    def predict_logits(self, embeddings):
        return self.linear(embeddings)


def test_trainer_rejects_out_of_range_labels_before_loss(tmp_path):
    images = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 2], dtype=torch.long)
    loader = DataLoader(TensorDataset(images, labels, labels), batch_size=2)

    trainer = Trainer(
        nn.Identity(),
        TwoClassHead(),
        loss=nn.CrossEntropyLoss(),
        device="cpu",
        progress=False,
        checkpoint_dir=tmp_path,
        early_stopping_enabled=False,
    )

    with pytest.raises(ValueError, match=r"labels must be in range \[0, 1\]"):
        trainer.train_one_epoch(loader, epoch=0, epochs=1)
