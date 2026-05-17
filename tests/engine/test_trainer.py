from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from apexid.engine.trainer import Trainer


class TrainHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, embeddings, labels):
        return self.linear(embeddings)

    def predict_logits(self, embeddings):
        return self.linear(embeddings)


class DummyValidator:
    def __init__(self):
        self.calls = 0

    def validate(self, val_loader, epoch=None, epochs=None):
        self.calls += 1
        return {
            "loss": 1.0 / self.calls,
            "acc": 0.5,
            "top1": 0.25,
            "top5": 0.75,
            "mAP": 0.4 + (0.1 * self.calls),
            "micro_f1": 0.6,
            "num_known": 4,
            "num_samples": 4,
        }


def _loader():
    images = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
    )
    labels = torch.tensor([0, 0, 1, 1])
    return DataLoader(TensorDataset(images, labels, labels), batch_size=2, shuffle=False)


def test_trainer_uses_validator_history_checkpoint_and_val_map_early_stopping(tmp_path):
    torch.manual_seed(42)
    backbone = nn.Linear(2, 2)
    head = TrainHead()
    optimizer = torch.optim.SGD(list(backbone.parameters()) + list(head.parameters()), lr=0.01)
    validator = DummyValidator()

    trainer = Trainer(
        backbone,
        head,
        loss=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        validator=validator,
        device="cpu",
        progress=False,
        checkpoint_dir=tmp_path,
        checkpoint_every_n_epochs=1,
        checkpoint_filename="checkpoint_epoch_{epoch_number:03d}.pt",
        checkpoint_delete_previous=True,
        early_stopping_enabled=True,
        early_stopping_monitor="val_mAP",
        early_stopping_mode="max",
        early_stopping_patience=3,
        early_stopping_dir=tmp_path,
    )

    history = trainer.train(_loader(), _loader(), epochs=2)

    assert validator.calls == 2
    assert trainer.early_stopping.monitor == "val_mAP"
    assert trainer.early_stopping.best_metric == history["val_mAP"][-1]
    assert set(history.keys()) == {
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "val_top1",
        "val_top5",
        "val_mAP",
        "val_micro_f1",
    }
    assert len(history["train_loss"]) == 2
    assert len(history["val_mAP"]) == 2
    assert sorted(path.name for path in Path(tmp_path).glob("checkpoint_epoch_*.pt")) == [
        "checkpoint_epoch_002.pt"
    ]
    assert (tmp_path / "best_full_model.pt").exists()
    assert (tmp_path / "best_backbone.pt").exists()


def test_trainer_rejects_unknown_labels_in_training(tmp_path):
    images = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    cls_labels = torch.tensor([0, -100])
    loader = DataLoader(TensorDataset(images, cls_labels, cls_labels), batch_size=2)

    trainer = Trainer(
        nn.Linear(2, 2),
        TrainHead(),
        loss=nn.CrossEntropyLoss(),
        device="cpu",
        progress=False,
        checkpoint_dir=tmp_path,
        early_stopping_enabled=False,
    )

    try:
        trainer.train_one_epoch(loader, epoch=0, epochs=1)
    except ValueError as exc:
        assert "Unknown/ignored labels" in str(exc)
    else:
        raise AssertionError("Expected training with unknown labels to fail.")
