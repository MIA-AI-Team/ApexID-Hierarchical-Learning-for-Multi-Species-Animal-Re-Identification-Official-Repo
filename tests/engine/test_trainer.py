from pathlib import Path
import json

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from apexid.engine.trainer import Trainer
from apexid.models.heads import SphereFace2Head


class RecordingLogger:
    def __init__(self):
        self.metrics = []
        self.artifacts = []
        self.finished = False

    def log_metrics(self, metrics, step=None):
        self.metrics.append((dict(metrics), step))

    def log_artifact(self, path, name=None, artifact_type="model", aliases=None, metadata=None):
        self.artifacts.append(
            {
                "path": Path(path),
                "name": name,
                "artifact_type": artifact_type,
                "aliases": list(aliases or []),
                "metadata": dict(metadata or {}),
            }
        )

    def finish(self):
        self.finished = True


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


class EmbeddingValidator:
    def __init__(self):
        self.return_embeddings_flags = []

    def validate(self, val_loader, epoch=None, epochs=None, return_embeddings=False):
        self.return_embeddings_flags.append(bool(return_embeddings))
        result = {
            "loss": 1.0,
            "acc": 0.5,
            "top1": 0.25,
            "top5": 0.75,
            "mAP": 0.5,
            "micro_f1": 0.6,
            "num_known": 4,
            "num_samples": 4,
        }
        if return_embeddings:
            result["embeddings"] = torch.randn(4, 2)
            result["metric_labels"] = torch.tensor([0, 0, 1, 1])
        return result


class OpenSetValidator:
    def __init__(self):
        self.calls = []

    def validate(self, *args, **kwargs):
        raise AssertionError("Expected open-set validation path.")

    def validate_open_set(
        self,
        gallery_loader,
        known_loader,
        unknown_loader,
        mixed_loader,
        evaluation_config=None,
        epoch=None,
        epochs=None,
        return_embeddings=False,
    ):
        self.calls.append(
            {
                "gallery": gallery_loader,
                "known": known_loader,
                "unknown": unknown_loader,
                "mixed": mixed_loader,
                "evaluation_config": dict(evaluation_config or {}),
                "epoch": epoch,
                "return_embeddings": bool(return_embeddings),
            }
        )
        result = {
            "loss": 0.25,
            "acc": 1.0,
            "top1": 1.0,
            "top5": 1.0,
            "mAP": 0.9,
            "micro_f1": 0.8,
            "num_known": 2,
            "num_samples": 4,
            "known/mAP": 0.9,
            "known/top1": 1.0,
            "known/top5": 1.0,
            "known/top10": 1.0,
            "unknown/ARI": 0.4,
            "mixed/F1": 0.8,
            "mixed/NormAcc": 0.7 + (0.01 * int(epoch or 0)),
            "rejection/AUROC": 0.6,
            "validation/diagnostics_ran": 1 if int(epoch or 0) == 4 else 0,
        }
        if int(epoch or 0) == 4:
            result["diagnostics/embedding_norm_mean"] = 1.0
        if return_embeddings:
            result["embeddings"] = torch.randn(4, 2)
            result["metric_labels"] = torch.tensor([0, 1, 2, 3])
        return result


class RecordingEmbeddingPlotter:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.plot_calls = []
        self.best_calls = []

    def plot_epoch(self, embeddings, labels, epoch):
        self.plot_calls.append(
            {
                "epoch": int(epoch),
                "embedding_shape": tuple(embeddings.shape),
                "labels": labels.tolist(),
            }
        )
        png = self.output_dir / f"epoch_{int(epoch) + 1:03d}_umap.png"
        meta = png.with_suffix(".json")
        png.parent.mkdir(parents=True, exist_ok=True)
        png.write_bytes(b"png")
        meta.write_text(json.dumps({"epoch": int(epoch)}), encoding="utf-8")
        return {"png": png, "json": meta, "metadata": {"epoch": int(epoch)}}

    def save_best(self, epoch_artifacts):
        self.best_calls.append(epoch_artifacts)
        png = self.output_dir / "best_embedding_space.png"
        meta = png.with_suffix(".json")
        png.write_bytes(epoch_artifacts["png"].read_bytes())
        meta.write_text(epoch_artifacts["json"].read_text(encoding="utf-8"), encoding="utf-8")
        return {"png": png, "json": meta, "metadata": {"best": True}}


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
        "val_loss",
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


def test_trainer_logs_epoch_metrics_and_best_artifacts(tmp_path):
    torch.manual_seed(42)
    backbone = nn.Linear(2, 2)
    head = TrainHead()
    optimizer = torch.optim.SGD(list(backbone.parameters()) + list(head.parameters()), lr=0.01)
    logger = RecordingLogger()

    trainer = Trainer(
        backbone,
        head,
        loss=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        validator=DummyValidator(),
        device="cpu",
        progress=False,
        verbose=False,
        checkpoint_dir=tmp_path,
        checkpoint_every_n_epochs=1,
        checkpoint_filename="last_checkpoint.pt",
        early_stopping_enabled=True,
        early_stopping_monitor="val_mAP",
        early_stopping_mode="max",
        early_stopping_dir=tmp_path,
        experiment_logger=logger,
    )

    trainer.train(_loader(), _loader(), epochs=1)

    assert logger.finished is True
    assert len(logger.metrics) == 1
    metrics, step = logger.metrics[0]
    assert step == 1
    assert metrics["epoch"] == 1
    assert metrics["epoch_index"] == 0
    assert metrics["epochs"] == 1
    assert metrics["lr"] == 0.01
    assert metrics["train_loss"] > 0.0
    assert metrics["val_loss"] == 1.0
    assert metrics["val_top1"] == 0.25
    assert metrics["val_top5"] == 0.75
    assert metrics["val_mAP"] == 0.5
    assert metrics["val_micro_f1"] == 0.6
    assert metrics["val_num_known"] == 4
    assert metrics["val_num_samples"] == 4
    assert metrics["early_stopping_improved"] == 1
    assert metrics["early_stopping_best_metric"] == 0.5
    assert metrics["early_stopping_counter"] == 0
    assert metrics["early_stopping_early_stop"] == 0

    artifact_names = {artifact["name"] for artifact in logger.artifacts}
    assert artifact_names == {"best_full_model", "best_backbone", "best_metadata"}
    assert all("best" in artifact["aliases"] for artifact in logger.artifacts)
    assert "last_checkpoint" not in artifact_names


def test_trainer_plots_validation_embeddings_and_logs_best_artifacts(tmp_path):
    torch.manual_seed(42)
    backbone = nn.Linear(2, 2)
    head = TrainHead()
    optimizer = torch.optim.SGD(list(backbone.parameters()) + list(head.parameters()), lr=0.01)
    validator = EmbeddingValidator()
    plotter = RecordingEmbeddingPlotter(tmp_path / "embedding_space")
    logger = RecordingLogger()

    trainer = Trainer(
        backbone,
        head,
        loss=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        validator=validator,
        device="cpu",
        progress=False,
        verbose=False,
        checkpoint_dir=tmp_path,
        early_stopping_enabled=True,
        early_stopping_monitor="val_mAP",
        early_stopping_mode="max",
        early_stopping_dir=tmp_path,
        experiment_logger=logger,
        embedding_plotter=plotter,
    )

    trainer.train(_loader(), _loader(), epochs=1)

    assert validator.return_embeddings_flags == [True]
    assert len(plotter.plot_calls) == 1
    assert plotter.plot_calls[0]["embedding_shape"] == (4, 2)
    assert len(plotter.best_calls) == 1
    assert (tmp_path / "embedding_space" / "epoch_001_umap.png").exists()
    assert (tmp_path / "embedding_space" / "best_embedding_space.png").exists()

    artifact_names = {artifact["name"] for artifact in logger.artifacts}
    assert "best_embedding_space" in artifact_names
    assert "best_embedding_space_metadata" in artifact_names


def test_trainer_does_not_request_embeddings_when_plotting_disabled(tmp_path):
    validator = EmbeddingValidator()
    trainer = _trainer(tmp_path, progress=False, verbose=False, validator=validator)

    trainer.train(_loader(), _loader(), epochs=1)

    assert validator.return_embeddings_flags == [False]


def test_trainer_dispatches_open_set_loaders_and_logs_slash_metrics(tmp_path):
    torch.manual_seed(42)
    backbone = nn.Linear(2, 2)
    head = TrainHead()
    optimizer = torch.optim.SGD(list(backbone.parameters()) + list(head.parameters()), lr=0.01)
    validator = OpenSetValidator()
    logger = RecordingLogger()
    val_loaders = {
        "gallery": object(),
        "known": object(),
        "unknown": object(),
        "mixed": object(),
    }

    trainer = Trainer(
        backbone,
        head,
        loss=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        validator=validator,
        device="cpu",
        progress=False,
        verbose=False,
        checkpoint_dir=tmp_path,
        early_stopping_enabled=True,
        early_stopping_dir=tmp_path,
        experiment_logger=logger,
        evaluation_config={"diagnostics_every_n_epochs": 5},
    )

    trainer.train(_loader(), val_loaders, epochs=5)

    assert trainer.early_stopping.monitor == "mixed/NormAcc"
    assert len(validator.calls) == 5
    assert validator.calls[0]["evaluation_config"] == {"diagnostics_every_n_epochs": 5}
    assert len(logger.metrics) == 5
    first_metrics, first_step = logger.metrics[0]
    last_metrics, last_step = logger.metrics[-1]
    assert first_step == 1
    assert last_step == 5
    assert "train_acc" not in first_metrics
    assert "val_acc" not in first_metrics
    assert first_metrics["mixed/NormAcc"] == 0.7
    assert first_metrics["validation/diagnostics_ran"] == 0
    assert "diagnostics/embedding_norm_mean" not in first_metrics
    assert last_metrics["mixed/NormAcc"] == 0.74
    assert last_metrics["validation/diagnostics_ran"] == 1
    assert last_metrics["diagnostics/embedding_norm_mean"] == 1.0
    assert last_metrics["early_stopping_best_metric"] == 0.74


def test_trainer_prints_open_set_metrics_without_classifier_accuracy(tmp_path, capsys):
    torch.manual_seed(42)
    backbone = nn.Linear(2, 2)
    head = TrainHead()
    optimizer = torch.optim.SGD(list(backbone.parameters()) + list(head.parameters()), lr=0.01)
    val_loaders = {
        "gallery": object(),
        "known": object(),
        "unknown": object(),
        "mixed": object(),
    }

    trainer = Trainer(
        backbone,
        head,
        loss=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        validator=OpenSetValidator(),
        device="cpu",
        progress=False,
        verbose=True,
        checkpoint_dir=tmp_path,
        early_stopping_enabled=False,
    )

    trainer.train(_loader(), val_loaders, epochs=1)

    captured = capsys.readouterr()
    assert "train_acc=" not in captured.out
    assert "val_acc=" not in captured.out
    assert "  known: mAP=0.9000" in captured.out
    assert "top10=1.0000" in captured.out
    assert "  unknown: ARI=0.4000" in captured.out
    assert "  mixed: F1=0.8000 | NormAcc=0.7000" in captured.out
    assert "  rejection: AUROC=0.6000" in captured.out
    assert "  validation: diagnostics_ran=0.0000" in captured.out


def test_trainer_steps_cyclic_lr_once_per_optimizer_update(tmp_path):
    torch.manual_seed(42)
    backbone = nn.Linear(2, 2)
    head = TrainHead()
    optimizer = torch.optim.SGD(list(backbone.parameters()) + list(head.parameters()), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=0.001,
        max_lr=0.01,
        step_size_up=1,
        cycle_momentum=False,
    )
    initial_last_epoch = scheduler.last_epoch
    train_loader = _loader()

    trainer = Trainer(
        backbone,
        head,
        loss=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        validator=DummyValidator(),
        device="cpu",
        progress=False,
        verbose=False,
        checkpoint_dir=tmp_path,
        early_stopping_enabled=False,
    )

    trainer.train(train_loader, _loader(), epochs=1)

    assert scheduler.last_epoch == initial_last_epoch + len(train_loader)


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


def test_trainer_trains_one_epoch_with_sphereface2_internal_loss(tmp_path):
    torch.manual_seed(42)
    head = SphereFace2Head(
        in_features=2,
        out_features=2,
        alpha=0.7,
        r=16.0,
        m=0.2,
        t=2.0,
        lw=20.0,
    )
    trainer = Trainer(
        nn.Identity(),
        head,
        loss=None,
        device="cpu",
        progress=False,
        checkpoint_dir=tmp_path,
        early_stopping_enabled=False,
    )

    metrics = trainer.train_one_epoch(_loader(), epoch=0, epochs=1)

    assert metrics["loss"] > 0.0
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert "acc" not in metrics


def test_trainer_prints_stable_epoch_summaries(tmp_path, capsys):
    trainer = _trainer(tmp_path, progress=False, verbose=True)

    trainer.train(_loader(), _loader(), epochs=2)

    captured = capsys.readouterr()
    assert "Epoch 1/2" in captured.out
    assert "Epoch 2/2" in captured.out
    assert "train_loss=" in captured.out
    assert "val_loss=" in captured.out
    assert "train_acc=" not in captured.out
    assert "val_acc=" not in captured.out
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
