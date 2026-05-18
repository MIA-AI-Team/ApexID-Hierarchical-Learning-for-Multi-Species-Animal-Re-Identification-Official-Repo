import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch


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
    val_csv = tmp_path / "val.csv"
    _make_csv(train_csv, image_dir, ["a", "a", "b", "b"])
    _make_csv(val_csv, image_dir, ["a", "b", "c"])
    return {
        "seed": 42,
        "data": {
            "train_path": str(train_csv),
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
        "checkpoint": {
            "output_dir": str(tmp_path / "checkpoints"),
            "every_n_epochs": 1,
            "filename": "last_checkpoint.pt",
            "delete_previous": True,
            "resume_path": None,
        },
        "early_stopping": {
            "enabled": False,
            "monitor": "val_mAP",
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


def test_train_script_builds_datasets_and_loaders_from_train_val_paths(tmp_path):
    train_script = _load_train_script()
    cfg = _base_cfg(tmp_path)

    train_dataset, val_dataset, num_classes = train_script.build_datasets(cfg)
    train_loader, val_loader = train_script.build_dataloaders(
        cfg,
        train_dataset,
        val_dataset,
        torch.device("cpu"),
    )

    assert num_classes == 2
    assert len(train_dataset) == 4
    assert len(val_dataset) == 3
    assert train_dataset.label_to_idx == {"a": 0, "b": 1}
    assert set(val_dataset.metrics_label_map.keys()) == {"a", "b", "c"}
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    assert tuple(train_batch[0].shape[1:]) == (3, 32, 32)
    assert tuple(val_batch[0].shape[1:]) == (3, 32, 32)


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
