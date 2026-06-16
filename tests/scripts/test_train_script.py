import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from apexid.models.heads import SphereFace2Head
from apexid.engine.loggers import NullLogger


def _load_train_script():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("apexid_train_script", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_rgb_png(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def _make_csv(path, image_dir, identities):
    rows = []
    for idx, identity in enumerate(identities):
        image_path = image_dir / f"{path.stem}_{idx}.png"
        image = np.full((24, 24, 3), [idx * 20 + 10, idx * 20 + 20, idx * 20 + 30], dtype=np.uint8)
        _write_rgb_png(image_path, image)
        rows.append({"identity": identity, "path": image_path.name})
    pd.DataFrame(rows).to_csv(path, index=False)


def _base_cfg(tmp_path):
    image_dir = tmp_path / "images"
    train_csv = tmp_path / "train.csv"
    val_known_csv = tmp_path / "val_known.csv"
    val_unknown_csv = tmp_path / "val_unknown.csv"
    val_csv = tmp_path / "val.csv"
    _make_csv(train_csv, image_dir, ["a", "a", "b", "b"])
    _make_csv(val_known_csv, image_dir, ["a", "b"])
    _make_csv(val_unknown_csv, image_dir, ["c", "c"])
    _make_csv(val_csv, image_dir, ["a", "b", "c", "c"])
    return {
        "seed": 42,
        "data": {
            "train_path": str(train_csv),
            "val_known_path": str(val_known_csv),
            "val_unknown_path": str(val_unknown_csv),
            "val_path": str(val_csv),
            "root_dir": str(image_dir),
            "image_size": [32, 32],
            "wavelet": False,
            "transforms": {
                "mean": [0.5, 0.5, 0.5],
                "std": [0.25, 0.25, 0.25],
            },
        },
        "dataloader": {
            "batch_size": 2,
            "val_batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "drop_last": False,
            "sampler": {
                "name": "none",
            },
        },
        "model": {
            "name": "apexid",
            "embedding_dim": 8,
            "backbone": {"name": "osnet_x0_25", "kwargs": {}},
        },
        "head": {
            "name": "arcface",
            "kwargs": {"s": 16.0, "m": 0.2},
        },
        "loss": {"name": "cross_entropy", "kwargs": {}},
        "optimizer": {"name": "sgd", "kwargs": {"lr": 0.01}},
        "scheduler": {"name": None, "kwargs": {}},
        "training": {
            "epochs": 1,
            "progress": False,
            "gradient_clip_norm": None,
            "classification_ignore_index": -100,
            "metric_ignore_index": None,
        },
        "validation": {"every_n_epochs": 1, "metrics": ["top1", "top5", "mAP", "micro_f1"]},
        "evaluation": {
            "diagnostics_every_n_epochs": 5,
            "diagnostics": {"enabled": True, "max_pairs": 100},
        },
        "checkpoint": {
            "output_dir": str(tmp_path / "checkpoints"),
            "every_n_epochs": 1,
            "filename": "last_checkpoint.pt",
            "delete_previous": True,
            "resume_path": None,
        },
        "early_stopping": {
            "enabled": False,
            "monitor": "mixed/NormAcc",
            "mode": "max",
            "patience": 2,
            "min_delta": 0.0,
            "output_dir": str(tmp_path / "checkpoints"),
            "full_model_name": "best_full_model.pt",
            "backbone_name": "best_backbone.pt",
            "metadata_name": "best_metadata.json",
        },
        "experiment": {"name": "test", "output_dir": str(tmp_path / "outputs")},
    }


class TrainHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, embeddings, labels):
        return self.linear(embeddings)

    def predict_logits(self, embeddings):
        return self.linear(embeddings)


def _tensor_loader():
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


def test_train_script_builds_datasets_and_loaders_from_train_val_paths(tmp_path):
    train_script = _load_train_script()
    cfg = _base_cfg(tmp_path)

    train_dataset, validation_datasets, num_classes = train_script.build_datasets(cfg)
    train_loader, validation_loaders = train_script.build_dataloaders(
        cfg,
        train_dataset,
        validation_datasets,
        torch.device("cpu"),
    )

    assert num_classes == 2
    assert len(train_dataset) == 4
    assert len(validation_datasets["gallery"]) == 4
    assert len(validation_datasets["known"]) == 2
    assert len(validation_datasets["unknown"]) == 2
    assert len(validation_datasets["mixed"]) == 4
    assert train_dataset.label_to_idx == {"a": 0, "b": 1}
    assert set(validation_datasets["mixed"].metrics_label_map.keys()) == {"a", "b", "c"}
    train_batch = next(iter(train_loader))
    val_batch = next(iter(validation_loaders["mixed"]))
    unknown_batch = next(iter(validation_loaders["unknown"]))
    assert tuple(train_batch[0].shape[1:]) == (3, 32, 32)
    assert tuple(val_batch[0].shape[1:]) == (3, 32, 32)
    assert set(validation_loaders) == {"gallery", "known", "unknown", "mixed"}
    assert unknown_batch[1].tolist() == [-100, -100]


def test_train_script_builds_finetune_optimizer_param_groups(tmp_path):
    train_script = _load_train_script()
    cfg = _base_cfg(tmp_path)
    cfg["model"]["backbone"] = {"name": "osnet_x0_25_pt", "kwargs": {"pretrained": False}}
    cfg.setdefault("finetune", {})["optimizer"] = {
        "backbone_weight_decay": 1e-5,
        "head_weight_decay": 5e-4,
    }
    cfg = train_script.resolve_training_profile(cfg)
    _, _, num_classes = train_script.build_datasets(cfg)
    model, head, _ = train_script.build_components(cfg, num_classes=num_classes)
    optimizer = train_script.build_optimizer(cfg, model, head)

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["param_group_name"] == "backbone"
    assert optimizer.param_groups[1]["param_group_name"] == "head"
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(1e-4)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(1e-5)
    assert optimizer.param_groups[1]["weight_decay"] == pytest.approx(5e-4)


def test_train_script_builds_cosine_warmup_scheduler(tmp_path):
    train_script = _load_train_script()
    cfg = _base_cfg(tmp_path)
    cfg["scheduler"] = {
        "name": "cosine_annealing_lr",
        "kwargs": {"T_max": 10, "eta_min": 1e-6},
        "warmup": {"enabled": True, "epochs": 2, "start_lr": 1e-6},
    }
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=1e-3)
    scheduler = train_script.build_scheduler(cfg, optimizer, total_epochs=10)

    assert isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR)


def test_train_script_builds_optimizer_and_scheduler(tmp_path):
    train_script = _load_train_script()
    cfg = _base_cfg(tmp_path)
    train_dataset, _, num_classes = train_script.build_datasets(cfg)
    model, head, criterion = train_script.build_components(cfg, num_classes=num_classes)
    optimizer = train_script.build_optimizer(cfg, model, head)
    scheduler = train_script.build_scheduler(cfg, optimizer)

    assert len(train_dataset.label_to_idx) == 2
    assert criterion is not None
    assert type(optimizer).__name__ == "SGD"
    assert scheduler is None


def test_train_script_builds_cyclic_lr_scheduler_from_loader_length(tmp_path):
    train_script = _load_train_script()
    cfg = _base_cfg(tmp_path)
    cfg["scheduler"] = {
        "name": "cyclic_lr",
        "kwargs": {
            "base_lr": 1e-4,
            "max_lr": 1e-3,
            "step_size_epochs": 3,
            "mode": "triangular2",
            "cycle_momentum": False,
        },
    }
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=1e-4)

    scheduler = train_script.build_scheduler(cfg, optimizer, train_loader=_tensor_loader())

    assert isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR)
    assert scheduler.total_size == 2 * 3 * len(_tensor_loader())
    assert scheduler.step_ratio == 0.5


def test_train_script_auto_applies_lr_range_test_overrides(tmp_path):
    train_script = _load_train_script()
    cfg = _base_cfg(tmp_path)
    cfg["lr_range_test"] = {
        "enabled": True,
        "mode": "auto_apply",
        "start_lr": 1e-5,
        "end_lr": 1e-2,
        "num_iter": 4,
        "smooth_beta": 0.98,
        "stop_factor": 4.0,
        "base_lr_divisor": 4.0,
        "output_dir": str(tmp_path / "lr_auto"),
    }
    backbone = nn.Linear(2, 2)
    head = TrainHead()
    optimizer = torch.optim.SGD(list(backbone.parameters()) + list(head.parameters()), lr=0.01)

    overrides = train_script.maybe_run_lr_range_test(
        cfg,
        backbone,
        head,
        nn.CrossEntropyLoss(),
        optimizer,
        _tensor_loader(),
        torch.device("cpu"),
        resume_path=None,
    )

    assert set(overrides) == {"base_lr", "max_lr"}
    assert optimizer.param_groups[0]["lr"] == overrides["base_lr"]
    assert overrides["max_lr"] > overrides["base_lr"]
    assert (tmp_path / "lr_auto" / "lr_range_test.json").exists()
    assert (tmp_path / "lr_auto" / "lr_range_test.csv").exists()


def test_train_script_report_only_lr_range_test_keeps_optimizer_lr(tmp_path):
    train_script = _load_train_script()
    cfg = _base_cfg(tmp_path)
    cfg["lr_range_test"] = {
        "enabled": True,
        "mode": "report_only",
        "start_lr": 1e-5,
        "end_lr": 1e-2,
        "num_iter": 4,
        "output_dir": str(tmp_path / "lr_report"),
    }
    backbone = nn.Linear(2, 2)
    head = TrainHead()
    optimizer = torch.optim.SGD(list(backbone.parameters()) + list(head.parameters()), lr=0.01)

    overrides = train_script.maybe_run_lr_range_test(
        cfg,
        backbone,
        head,
        nn.CrossEntropyLoss(),
        optimizer,
        _tensor_loader(),
        torch.device("cpu"),
        resume_path=None,
    )

    assert overrides is None
    assert optimizer.param_groups[0]["lr"] == 0.01
    assert (tmp_path / "lr_report" / "lr_range_test.json").exists()


def test_train_script_skips_lr_range_test_when_resuming(tmp_path):
    train_script = _load_train_script()
    cfg = _base_cfg(tmp_path)
    cfg["lr_range_test"] = {
        "enabled": True,
        "mode": "auto_apply",
        "output_dir": str(tmp_path / "lr_resume"),
    }
    backbone = nn.Linear(2, 2)
    head = TrainHead()
    optimizer = torch.optim.SGD(list(backbone.parameters()) + list(head.parameters()), lr=0.01)

    overrides = train_script.maybe_run_lr_range_test(
        cfg,
        backbone,
        head,
        nn.CrossEntropyLoss(),
        optimizer,
        _tensor_loader(),
        torch.device("cpu"),
        resume_path=str(tmp_path / "checkpoint.pt"),
    )

    assert overrides is None
    assert optimizer.param_groups[0]["lr"] == 0.01
    assert not (tmp_path / "lr_resume").exists()


def test_train_script_builds_embedding_plotter_with_default_output_dirs(tmp_path):
    train_script = _load_train_script()
    cfg = _base_cfg(tmp_path)
    cfg["embedding_plot"] = {
        "enabled": True,
        "split": "val",
        "output_dir": None,
        "best_output_dir": None,
        "n_neighbors": 5,
        "min_dist": 0.2,
        "metric": "cosine",
        "random_state": None,
        "point_size": 9,
        "alpha": 0.7,
        "dpi": 150,
        "max_points": 100,
        "filename_template": "epoch_{epoch_number:03d}_umap.png",
        "best_filename": "best_embedding_space.png",
    }

    plotter = train_script.build_embedding_plotter(cfg)

    assert plotter.output_dir == tmp_path / "outputs" / "embedding_space"
    assert plotter.best_output_dir == tmp_path / "checkpoints"
    assert plotter.n_neighbors == 5
    assert plotter.min_dist == 0.2
    assert plotter.metric == "cosine"
    assert plotter.random_state == 42
    assert plotter.max_points == 100


def test_train_script_rejects_non_validation_embedding_plot_split(tmp_path):
    train_script = _load_train_script()
    cfg = _base_cfg(tmp_path)
    cfg["embedding_plot"] = {"enabled": True, "split": "train"}

    with pytest.raises(ValueError, match="supports only 'val'"):
        train_script.build_embedding_plotter(cfg)


def test_train_script_builds_sphereface2_internal_loss_components(tmp_path):
    train_script = _load_train_script()
    cfg = _base_cfg(tmp_path)
    cfg["head"] = {
        "name": "sphereface2",
        "kwargs": {
            "magn_type": "C",
            "alpha": 0.7,
            "r": 16.0,
            "m": 0.2,
            "t": 2.0,
            "lw": 20.0,
        },
    }
    cfg["loss"] = {"name": None, "kwargs": {}}
    _, _, num_classes = train_script.build_datasets(cfg)

    model, head, criterion = train_script.build_components(cfg, num_classes=num_classes)
    optimizer = train_script.build_optimizer(cfg, model, head)

    assert isinstance(head, SphereFace2Head)
    assert criterion is None
    assert len(optimizer.param_groups[0]["params"]) > 0


def test_train_script_rejects_external_loss_for_sphereface2(tmp_path):
    train_script = _load_train_script()
    cfg = _base_cfg(tmp_path)
    cfg["head"] = {
        "name": "sphereface2_focal",
        "kwargs": {"alpha": 0.7, "focal_gamma": 2.0},
    }
    _, _, num_classes = train_script.build_datasets(cfg)

    with pytest.raises(ValueError, match="Set loss.name to null"):
        train_script.build_components(cfg, num_classes=num_classes)


def test_train_script_builds_trainer_verbose_from_config(tmp_path):
    train_script = _load_train_script()
    cfg = _base_cfg(tmp_path)
    cfg["training"]["progress"] = False
    cfg["training"]["verbose"] = True
    _, _, num_classes = train_script.build_datasets(cfg)
    model, head, criterion = train_script.build_components(cfg, num_classes=num_classes)
    optimizer = train_script.build_optimizer(cfg, model, head)
    scheduler = train_script.build_scheduler(cfg, optimizer)

    trainer = train_script.build_trainer(
        cfg,
        model,
        head,
        criterion,
        optimizer,
        scheduler,
        torch.device("cpu"),
    )

    assert trainer.progress is False
    assert trainer.verbose is True

    cfg["training"].pop("verbose")
    fallback_trainer = train_script.build_trainer(
        cfg,
        model,
        head,
        criterion,
        optimizer,
        scheduler,
        torch.device("cpu"),
    )

    assert fallback_trainer.verbose is False


def test_train_script_tracking_disabled_builds_null_logger(tmp_path):
    train_script = _load_train_script()
    cfg = _base_cfg(tmp_path)
    cfg["tracking"] = {
        "enabled": False,
        "provider": "wandb",
        "project": "ApexID",
        "log_artifacts": "best",
    }

    logger = train_script.build_experiment_logger(
        cfg,
        metadata={
            "num_classes": 2,
            "train_dataset_size": 4,
            "val_dataset_size": 3,
            "device": "cpu",
            "resume_path": None,
        },
    )

    assert isinstance(logger, NullLogger)


def test_train_script_requires_train_and_val_paths(tmp_path):
    train_script = _load_train_script()
    cfg = _base_cfg(tmp_path)
    cfg["data"]["train_path"] = None
    cfg["data"]["train_csv"] = None

    try:
        train_script.build_datasets(cfg)
    except ValueError as exc:
        assert "data.train_path" in str(exc)
    else:
        raise AssertionError("Expected missing train_path to raise ValueError.")
