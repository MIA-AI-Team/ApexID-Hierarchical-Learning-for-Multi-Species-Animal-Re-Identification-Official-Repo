from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from apexid.engine.checkpoint import load_training_checkpoint
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
    assert step is None
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
    assert first_step is None
    assert last_step is None
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


class EpochMetricValidator:
    def __init__(self, values):
        self.values = list(values)
        self.calls = []

    def validate(self, val_loader, epoch=None, epochs=None):
        epoch = int(epoch)
        self.calls.append(epoch)
        value = float(self.values[epoch])
        return {
            "loss": 1.0 - value,
            "acc": value,
            "top1": value,
            "top5": value,
            "mAP": value,
            "micro_f1": value,
            "num_known": 4,
            "num_samples": 4,
        }


def _shuffled_loader(seed):
    images = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1])
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(images, labels, labels),
        batch_size=2,
        shuffle=True,
        generator=generator,
    )


def _resume_trainer(output_dir, metric_values, logger=None, patience=3):
    torch.manual_seed(2026)
    backbone = nn.Linear(2, 2)
    head = TrainHead()
    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(head.parameters()),
        lr=1e-3,
        weight_decay=1e-4,
        foreach=False,
    )
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=1e-4,
        max_lr=1e-3,
        step_size_up=2,
        cycle_momentum=False,
    )
    return Trainer(
        backbone,
        head,
        loss=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        validator=EpochMetricValidator(metric_values),
        device="cpu",
        progress=False,
        verbose=False,
        checkpoint_dir=output_dir,
        checkpoint_every_n_epochs=1,
        checkpoint_filename="last_checkpoint.pt",
        early_stopping_enabled=True,
        early_stopping_monitor="val_mAP",
        early_stopping_mode="max",
        early_stopping_patience=patience,
        early_stopping_dir=output_dir,
        experiment_logger=logger,
    )


def _assert_optimizer_state_equal(actual, expected):
    assert actual["param_groups"] == expected["param_groups"]
    assert actual["state"].keys() == expected["state"].keys()
    for parameter_id in actual["state"]:
        assert actual["state"][parameter_id].keys() == expected["state"][parameter_id].keys()
        for key, actual_value in actual["state"][parameter_id].items():
            expected_value = expected["state"][parameter_id][key]
            if torch.is_tensor(actual_value):
                torch.testing.assert_close(actual_value, expected_value)
            else:
                assert actual_value == expected_value


def _load_resume_checkpoint(trainer, checkpoint_path, train_loader, val_loader):
    return load_training_checkpoint(
        checkpoint_path,
        model=trainer.model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        early_stopping=trainer.early_stopping,
        data_loaders={"train": train_loader, "validation": val_loader},
    )


def test_continuous_and_interrupted_training_have_identical_resume_state(tmp_path):
    metric_values = [0.60, 0.59, 0.58, 0.57, 0.56]

    continuous_logger = RecordingLogger()
    continuous = _resume_trainer(tmp_path / "continuous", metric_values, continuous_logger)
    continuous_train_loader = _shuffled_loader(77)
    continuous_val_loader = _loader()
    continuous_history = continuous.train(
        continuous_train_loader,
        continuous_val_loader,
        epochs=5,
    )

    first_logger = RecordingLogger()
    interrupted = _resume_trainer(tmp_path / "interrupted", metric_values, first_logger)
    interrupted_train_loader = _shuffled_loader(77)
    interrupted_val_loader = _loader()
    interrupted.train(interrupted_train_loader, interrupted_val_loader, epochs=2)
    checkpoint_path = tmp_path / "interrupted" / "last_checkpoint.pt"
    best_before_resume = (tmp_path / "interrupted" / "best_full_model.pt").read_bytes()

    resumed_logger = RecordingLogger()
    resumed = _resume_trainer(tmp_path / "interrupted", metric_values, resumed_logger)
    resumed_train_loader = _shuffled_loader(999)
    resumed_val_loader = _loader()
    checkpoint = _load_resume_checkpoint(
        resumed,
        checkpoint_path,
        resumed_train_loader,
        resumed_val_loader,
    )
    resumed_history = resumed.train(
        resumed_train_loader,
        resumed_val_loader,
        epochs=5,
        start_epoch=checkpoint["next_epoch"],
        resume_history=checkpoint["history"],
    )

    assert resumed_history == continuous_history
    assert len(resumed_history["train_loss"]) == 4
    assert resumed.early_stopping.state_dict() == continuous.early_stopping.state_dict()
    assert resumed.early_stopping.best_metric == 0.60
    assert resumed.early_stopping.best_epoch == 0
    assert resumed.early_stopping.counter == 3
    assert resumed.early_stopping.early_stop is True
    assert (tmp_path / "interrupted" / "best_full_model.pt").read_bytes() == best_before_resume

    for key, expected_value in continuous.model.state_dict().items():
        torch.testing.assert_close(resumed.model.state_dict()[key], expected_value)
    _assert_optimizer_state_equal(resumed.optimizer.state_dict(), continuous.optimizer.state_dict())
    assert resumed.scheduler.state_dict() == continuous.scheduler.state_dict()

    continuous_lrs = [metrics["lr"] for metrics, _ in continuous_logger.metrics]
    resumed_lrs = [metrics["lr"] for metrics, _ in first_logger.metrics + resumed_logger.metrics]
    assert resumed_lrs == continuous_lrs
    assert [metrics["epoch"] for metrics, _ in first_logger.metrics + resumed_logger.metrics] == [1, 2, 3, 4]


def test_better_metric_after_resume_replaces_historical_best(tmp_path):
    metric_values = [0.60, 0.59, 0.65]
    trainer = _resume_trainer(tmp_path, metric_values, patience=4)
    train_loader = _shuffled_loader(12)
    val_loader = _loader()
    trainer.train(train_loader, val_loader, epochs=2)
    best_before_resume = (tmp_path / "best_full_model.pt").read_bytes()

    resumed = _resume_trainer(tmp_path, metric_values, patience=4)
    resumed_train_loader = _shuffled_loader(999)
    resumed_val_loader = _loader()
    checkpoint = _load_resume_checkpoint(
        resumed,
        tmp_path / "last_checkpoint.pt",
        resumed_train_loader,
        resumed_val_loader,
    )
    resumed.train(
        resumed_train_loader,
        resumed_val_loader,
        epochs=3,
        start_epoch=checkpoint["next_epoch"],
        resume_history=checkpoint["history"],
    )

    assert resumed.early_stopping.best_metric == 0.65
    assert resumed.early_stopping.best_epoch == 2
    assert resumed.early_stopping.counter == 0
    assert (tmp_path / "best_full_model.pt").read_bytes() != best_before_resume


def test_resumed_already_stopped_checkpoint_runs_no_additional_epoch(tmp_path):
    validator = EpochMetricValidator([0.6, 0.5, 0.4])
    trainer = _resume_trainer(tmp_path, [0.6, 0.5, 0.4], patience=2)
    trainer.validator = validator
    trainer.early_stopping(0.6, model=trainer.model, epoch=0)
    trainer.early_stopping(0.5, model=trainer.model, epoch=1)
    trainer.early_stopping(0.4, model=trainer.model, epoch=2)
    history = {key: [1.0] for key in trainer.empty_history()}

    result = trainer.train(
        _shuffled_loader(2),
        _loader(),
        epochs=5,
        start_epoch=3,
        resume_history=history,
    )

    assert result is history
    assert validator.calls == []
