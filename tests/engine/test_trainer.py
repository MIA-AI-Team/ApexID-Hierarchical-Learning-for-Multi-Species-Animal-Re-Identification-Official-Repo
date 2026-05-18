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


def _trainer(tmp_path, progress=False, verbose=True, validator=None):
    torch.manual_seed(42)
    backbone = nn.Linear(2, 2)
    head = TrainHead()
    optimizer = torch.optim.SGD(list(backbone.parameters()) + list(head.parameters()), lr=0.01)
    return Trainer(
        backbone,
        head,
        loss=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        validator=validator or DummyValidator(),
        device="cpu",
        progress=progress,
        verbose=verbose,
        checkpoint_dir=tmp_path,
        early_stopping_enabled=False,
    )


class TrackingEpochSampler:
    def __init__(self):
        self.epochs = []

    def set_epoch(self, epoch):
        self.epochs.append(epoch)


class TrackingLoader:
    def __init__(self, sampler=None, batch_sampler=None):
        self.sampler = sampler
        self.batch_sampler = batch_sampler

    def __iter__(self):
        images = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ]
        )
        labels = torch.tensor([0, 0, 1, 1])
        yield images, labels, labels

    def __len__(self):
        return 1


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


def test_trainer_prints_stable_epoch_summaries(tmp_path, capsys):
    trainer = _trainer(tmp_path, progress=False, verbose=True)

    trainer.train(_loader(), _loader(), epochs=2)

    captured = capsys.readouterr()
    assert "Epoch 1/2" in captured.out
    assert "Epoch 2/2" in captured.out
    assert "train_loss=" in captured.out
    assert "val_loss=" in captured.out
    assert "mAP=" in captured.out
    assert "lr=1.00e-02" in captured.out


def test_trainer_verbose_false_suppresses_epoch_summaries(tmp_path, capsys):
    trainer = _trainer(tmp_path, progress=False, verbose=False)

    trainer.train(_loader(), _loader(), epochs=1)

    captured = capsys.readouterr()
    assert "Epoch 1/1" not in captured.out
    assert "train_loss=" not in captured.out


def test_trainer_resume_summary_uses_actual_epoch_number(tmp_path, capsys):
    trainer = _trainer(tmp_path, progress=False, verbose=True)

    trainer.train(_loader(), _loader(), epochs=4, start_epoch=2)

    captured = capsys.readouterr()
    assert "Epoch 3/4" in captured.out
    assert "Epoch 4/4" in captured.out
    assert "Epoch 1/4" not in captured.out


def test_trainer_sets_epoch_on_train_loader_sampler_and_batch_sampler(tmp_path):
    sampler = TrackingEpochSampler()
    batch_sampler = TrackingEpochSampler()
    sampler_loader = TrackingLoader(sampler=sampler)
    batch_sampler_loader = TrackingLoader(batch_sampler=batch_sampler)

    _trainer(tmp_path / "sampler", progress=False, verbose=False).train(
        sampler_loader,
        _loader(),
        epochs=3,
    )
    _trainer(tmp_path / "batch_sampler", progress=False, verbose=False).train(
        batch_sampler_loader,
        _loader(),
        epochs=3,
    )

    assert sampler.epochs == [0, 1, 2]
    assert batch_sampler.epochs == [0, 1, 2]
