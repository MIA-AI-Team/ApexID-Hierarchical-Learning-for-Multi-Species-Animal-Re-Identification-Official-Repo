#!/usr/bin/env python3
import argparse
import logging
import os
import sys
import warnings
from pathlib import Path


def _suppress_library_warnings():
    """Silence noisy third-party warnings/logs for training runs."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("ABSL_LOGGING_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
    warnings.filterwarnings("ignore")
    logging.captureWarnings(True)
    for logger_name in (
        "absl",
        "matplotlib",
        "numba",
        "tensorflow",
        "umap",
        "wandb",
    ):
        logging.getLogger(logger_name).setLevel(logging.ERROR)


_suppress_library_warnings()

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader

from apexid.data import build_eval_transforms, build_train_transforms
from apexid.data.datasets.dataset import ApexID_Dataset
from apexid.data.samplers import PKBatchSampler
from apexid.engine import Trainer, build_experiment_logger, load_training_checkpoint
from apexid.models import build_components
from apexid.utils.config import DEFAULT_CONFIG_PATH, load_config
from apexid.utils.device import get_device
from apexid.utils.embedding_umap import EmbeddingUMAPPlotter
from apexid.utils.lr_range_test import run_lr_range_test, write_lr_range_test_artifacts
from apexid.utils.seed import DEFAULT_SEED, make_torch_generator, seed_everything, seed_worker


def _as_dict(value):
    """Return ``value`` as a normal dictionary."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "items"):
        return dict(value.items())
    raise TypeError(f"Expected a dict-like value, got {type(value).__name__}.")


def _data_path(data_cfg, primary_key, legacy_key):
    """Read a required data path from the preferred or legacy config key."""
    value = data_cfg.get(primary_key)
    if value is None:
        value = data_cfg.get(legacy_key)
    if value in {None, ""}:
        raise ValueError(
            f"Missing data.{primary_key}. Set it to a CSV file with columns "
            "'identity' and 'path'."
        )
    return value


def build_datasets(cfg):
    """Build train/validation datasets from configured CSV files.

    Args:
        cfg (dict): Loaded experiment config. Requires ``data.train_path`` and
            ``data.val_path`` or their legacy aliases ``train_csv`` and
            ``val_csv``.

    Returns:
        tuple: ``(train_dataset, val_dataset, num_classes)``.
    """
    data_cfg = _as_dict(cfg.get("data", {}))
    train_path = _data_path(data_cfg, "train_path", "train_csv")
    val_path = _data_path(data_cfg, "val_path", "val_csv")
    root_dir = data_cfg.get("root_dir")

    train_transforms = build_train_transforms(config=cfg)
    eval_transforms = build_eval_transforms(config=cfg)

    label_to_idx = ApexID_Dataset.build_label_mapping(train_path)
    metrics_label_map = ApexID_Dataset.build_label_mapping_union(train_path, val_path)
    unknown_label = _as_dict(cfg.get("training", {})).get("classification_ignore_index", -100)

    train_dataset = ApexID_Dataset(
        train_path,
        transform=train_transforms,
        label_to_idx=label_to_idx,
        root_dir=root_dir,
        allow_unknown_identity=False,
        unknown_label=unknown_label,
        metrics_label_map=metrics_label_map,
    )
    val_dataset = ApexID_Dataset(
        val_path,
        transform=eval_transforms,
        label_to_idx=label_to_idx,
        root_dir=root_dir,
        allow_unknown_identity=True,
        unknown_label=unknown_label,
        metrics_label_map=metrics_label_map,
    )
    return train_dataset, val_dataset, len(label_to_idx)


def build_dataloaders(cfg, train_dataset, val_dataset, device):
    """Build train and validation DataLoaders.

    Args:
        cfg (dict): Loaded experiment config.
        train_dataset (torch.utils.data.Dataset): Training dataset.
        val_dataset (torch.utils.data.Dataset): Validation dataset.
        device (torch.device): Selected training device.

    Returns:
        tuple: ``(train_loader, val_loader)``.
    """
    seed = int(cfg.get("seed", DEFAULT_SEED) or DEFAULT_SEED)
    loader_cfg = _as_dict(cfg.get("dataloader", {}))
    sampler_cfg = _as_dict(loader_cfg.get("sampler", {}))
    num_workers = int(loader_cfg.get("num_workers", 0))
    pin_memory = bool(loader_cfg.get("pin_memory", device.type == "cuda"))
    persistent_workers = bool(loader_cfg.get("persistent_workers", False)) and num_workers > 0
    drop_last = bool(loader_cfg.get("drop_last", True))
    batch_size = int(loader_cfg.get("batch_size", 32))
    val_batch_size = int(loader_cfg.get("val_batch_size", batch_size))

    common_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "worker_init_fn": seed_worker,
    }

    sampler_name = str(sampler_cfg.get("name", "pk")).lower()
    if sampler_name in {"pk", "p_k", "pk_sampler"}:
        train_batch_sampler = PKBatchSampler(
            train_dataset,
            num_identities=int(sampler_cfg.get("num_identities", 8)),
            num_instances=int(sampler_cfg.get("num_instances", 4)),
            batches_per_epoch=sampler_cfg.get("batches_per_epoch"),
            seed=seed,
            drop_last=drop_last,
            shuffle=bool(sampler_cfg.get("shuffle", True)),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            **common_kwargs,
        )
    elif sampler_name in {"none", "random", "shuffle"}:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last,
            generator=make_torch_generator(seed),
            **common_kwargs,
        )
    else:
        raise ValueError("Unknown dataloader.sampler.name '{}'. Use 'pk' or 'none'.".format(sampler_name))

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        drop_last=False,
        generator=make_torch_generator(seed),
        **common_kwargs,
    )
    return train_loader, val_loader


def build_optimizer(cfg, model, head):
    """Build the optimizer from config."""
    optimizer_cfg = _as_dict(cfg.get("optimizer", {}))
    name = str(optimizer_cfg.get("name", "adam")).lower()
    kwargs = _as_dict(optimizer_cfg.get("kwargs", {}))
    params = list(model.parameters()) + list(head.parameters())

    if name == "adam":
        return torch.optim.Adam(params, **kwargs)
    if name == "adamw":
        return torch.optim.AdamW(params, **kwargs)
    if name == "sgd":
        return torch.optim.SGD(params, **kwargs)
    raise ValueError("Unknown optimizer '{}'. Available: adam, adamw, sgd.".format(name))


def build_scheduler(cfg, optimizer, train_loader=None, overrides=None):
    """Build an optional learning-rate scheduler from config."""
    scheduler_cfg = _as_dict(cfg.get("scheduler", {}))
    name = scheduler_cfg.get("name")
    if name in {None, "", "none", "null"}:
        return None

    name = str(name).lower()
    kwargs = dict(_as_dict(scheduler_cfg.get("kwargs", {})))
    if name in {"step_lr", "steplr"}:
        return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    if name in {"cosine_annealing_lr", "cosineannealinglr", "cosine"}:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    if name in {"reduce_on_plateau", "reduce_lr_on_plateau", "reducelronplateau"}:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    if name in {"cyclic_lr", "cycliclr", "cyclic"}:
        if overrides is not None:
            kwargs.update(_as_dict(overrides))
        step_size_epochs = kwargs.pop("step_size_epochs", None)
        if "step_size_up" not in kwargs:
            if train_loader is None:
                raise ValueError("train_loader is required when scheduler.step_size_up is omitted for CyclicLR.")
            step_size_epochs = 4.0 if step_size_epochs is None else float(step_size_epochs)
            kwargs["step_size_up"] = max(1, int(step_size_epochs * len(train_loader)))
        kwargs.setdefault("cycle_momentum", False)
        return torch.optim.lr_scheduler.CyclicLR(optimizer, **kwargs)
    raise ValueError(
        "Unknown scheduler '{}'. Available: step_lr, cosine_annealing_lr, "
        "reduce_on_plateau, cyclic_lr.".format(name)
    )


def maybe_run_lr_range_test(cfg, model, head, criterion, optimizer, train_loader, device, resume_path=None):
    """Run the optional LR range test and return scheduler overrides."""
    lr_test_cfg = _as_dict(cfg.get("lr_range_test", {}))
    if not bool(lr_test_cfg.get("enabled", False)):
        return None
    if resume_path not in {None, ""}:
        return None

    mode = str(lr_test_cfg.get("mode", "auto_apply")).lower()
    if mode not in {"auto_apply", "report_only"}:
        raise ValueError("lr_range_test.mode must be 'auto_apply' or 'report_only'.")

    experiment_dir = Path(_as_dict(cfg.get("experiment", {})).get("output_dir", "outputs/apexid_run"))
    output_dir = lr_test_cfg.get("output_dir")
    if output_dir in {None, ""}:
        output_dir = experiment_dir / "lr_range_test"

    result = run_lr_range_test(
        backbone=model,
        head=head,
        criterion=criterion,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        start_lr=float(lr_test_cfg.get("start_lr", 1e-7)),
        end_lr=float(lr_test_cfg.get("end_lr", 1e-2)),
        num_iter=int(lr_test_cfg.get("num_iter", 100)),
        smooth_beta=float(lr_test_cfg.get("smooth_beta", 0.98)),
        stop_factor=float(lr_test_cfg.get("stop_factor", 4.0)),
        base_lr_divisor=float(lr_test_cfg.get("base_lr_divisor", 4.0)),
        classification_ignore_index=_as_dict(cfg.get("training", {})).get("classification_ignore_index", -100),
        gradient_clip_norm=_as_dict(cfg.get("training", {})).get("gradient_clip_norm"),
    )
    artifact_paths = write_lr_range_test_artifacts(result, output_dir)

    print(
        "LR range test: "
        f"base_lr={result.suggested_base_lr:.3e}, "
        f"max_lr={result.suggested_max_lr:.3e}, "
        f"stop_reason={result.stop_reason}, "
        f"artifacts={artifact_paths['json']}"
    )

    if mode == "report_only":
        return None

    for group in optimizer.param_groups:
        group["lr"] = result.suggested_base_lr
    return {
        "base_lr": result.suggested_base_lr,
        "max_lr": result.suggested_max_lr,
    }


def build_embedding_plotter(cfg):
    """Build the optional validation embedding UMAP plotter from config."""
    plot_cfg = _as_dict(cfg.get("embedding_plot", {}))
    if not bool(plot_cfg.get("enabled", False)):
        return None

    split = str(plot_cfg.get("split", "val")).lower()
    if split not in {"val", "validation"}:
        raise ValueError("embedding_plot.split currently supports only 'val'.")

    experiment_dir = Path(_as_dict(cfg.get("experiment", {})).get("output_dir", "outputs/apexid_run"))
    checkpoint_cfg = _as_dict(cfg.get("checkpoint", {}))
    early_cfg = _as_dict(cfg.get("early_stopping", {}))
    output_dir = plot_cfg.get("output_dir")
    if output_dir in {None, ""}:
        output_dir = experiment_dir / "embedding_space"
    best_output_dir = plot_cfg.get("best_output_dir")
    if best_output_dir in {None, ""}:
        best_output_dir = early_cfg.get("output_dir", checkpoint_cfg.get("output_dir", "checkpoints"))

    random_state = plot_cfg.get("random_state")
    if random_state in {None, ""}:
        random_state = int(cfg.get("seed", DEFAULT_SEED) or DEFAULT_SEED)

    return EmbeddingUMAPPlotter(
        output_dir=output_dir,
        best_output_dir=best_output_dir,
        split="val",
        n_neighbors=int(plot_cfg.get("n_neighbors", 15)),
        min_dist=float(plot_cfg.get("min_dist", 0.1)),
        metric=plot_cfg.get("metric", "euclidean"),
        random_state=int(random_state),
        point_size=float(plot_cfg.get("point_size", 8)),
        alpha=float(plot_cfg.get("alpha", 0.85)),
        dpi=int(plot_cfg.get("dpi", 200)),
        max_points=plot_cfg.get("max_points"),
        filename_template=plot_cfg.get("filename_template", "epoch_{epoch_number:03d}_umap.png"),
        best_filename=plot_cfg.get("best_filename", "best_embedding_space.png"),
    )


def build_trainer(
    cfg,
    model,
    head,
    criterion,
    optimizer,
    scheduler,
    device,
    experiment_logger=None,
    embedding_plotter=None,
):
    """Build the ApexID Trainer from config-derived components."""
    training_cfg = _as_dict(cfg.get("training", {}))
    validation_cfg = _as_dict(cfg.get("validation", {}))
    checkpoint_cfg = _as_dict(cfg.get("checkpoint", {}))
    early_cfg = _as_dict(cfg.get("early_stopping", {}))

    return Trainer(
        model,
        head,
        loss=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=int(training_cfg.get("epochs", 100)),
        device=device,
        classification_ignore_index=training_cfg.get("classification_ignore_index", -100),
        metric_ignore_index=training_cfg.get("metric_ignore_index"),
        progress=bool(training_cfg.get("progress", True)),
        verbose=bool(training_cfg.get("verbose", training_cfg.get("progress", True))),
        gradient_clip_norm=training_cfg.get("gradient_clip_norm"),
        validation_every_n_epochs=int(validation_cfg.get("every_n_epochs", 1)),
        validation_metric_chunk_size=int(validation_cfg.get("metric_chunk_size", 1024)),
        checkpoint_dir=checkpoint_cfg.get("output_dir", "checkpoints"),
        checkpoint_every_n_epochs=int(checkpoint_cfg.get("every_n_epochs", 1)),
        checkpoint_filename=checkpoint_cfg.get("filename", "last_checkpoint.pt"),
        checkpoint_delete_previous=bool(checkpoint_cfg.get("delete_previous", True)),
        early_stopping_enabled=bool(early_cfg.get("enabled", True)),
        early_stopping_monitor=early_cfg.get("monitor", "val_mAP"),
        early_stopping_mode=early_cfg.get("mode", "max"),
        early_stopping_patience=int(early_cfg.get("patience", 7)),
        early_stopping_min_delta=float(early_cfg.get("min_delta", 0.0)),
        early_stopping_dir=early_cfg.get("output_dir", checkpoint_cfg.get("output_dir", "checkpoints")),
        early_stopping_full_model_name=early_cfg.get("full_model_name", "best_full_model.pt"),
        early_stopping_backbone_name=early_cfg.get("backbone_name", "best_backbone.pt"),
        early_stopping_metadata_name=early_cfg.get("metadata_name", "best_metadata.json"),
        metrics_base_seed=int(cfg.get("seed", DEFAULT_SEED) or DEFAULT_SEED),
        experiment_logger=experiment_logger,
        embedding_plotter=embedding_plotter,
    )


def train_from_config(cfg):
    """Run a full training job from a loaded config dictionary.

    Args:
        cfg (dict): Loaded experiment config.

    Returns:
        dict: Training history returned by ``Trainer.train``.
    """
    seed = int(cfg.get("seed", DEFAULT_SEED) or DEFAULT_SEED)
    seed_everything(seed)

    device_cfg = _as_dict(cfg.get("device", {}))
    device = get_device(
        prefer_gpu=bool(device_cfg.get("prefer_gpu", True)),
        gpu_index=int(device_cfg.get("gpu_index", 0)),
    )

    output_dir = Path(_as_dict(cfg.get("experiment", {})).get("output_dir", "outputs/apexid_run"))
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, num_classes = build_datasets(cfg)
    train_loader, val_loader = build_dataloaders(cfg, train_dataset, val_dataset, device)

    checkpoint_cfg = _as_dict(cfg.get("checkpoint", {}))
    resume_path = checkpoint_cfg.get("resume_path")
    tracking_metadata = {
        "num_classes": int(num_classes),
        "train_dataset_size": int(len(train_dataset)),
        "val_dataset_size": int(len(val_dataset)),
        "device": str(device),
        "resume_path": resume_path,
    }

    model, head, criterion = build_components(cfg, num_classes=num_classes)
    model.to(device)
    head.to(device)
    optimizer = build_optimizer(cfg, model, head)
    scheduler_overrides = maybe_run_lr_range_test(
        cfg,
        model,
        head,
        criterion,
        optimizer,
        train_loader,
        device,
        resume_path=resume_path,
    )
    scheduler = build_scheduler(cfg, optimizer, train_loader=train_loader, overrides=scheduler_overrides)
    experiment_logger = build_experiment_logger(cfg, metadata=tracking_metadata)
    embedding_plotter = build_embedding_plotter(cfg)
    trainer = build_trainer(cfg, model, head, criterion, optimizer, scheduler, device,
        experiment_logger=experiment_logger,
        embedding_plotter=embedding_plotter,
    )

    start_epoch = 0
    resume_history = None
    if resume_path not in {None, ""}:
        checkpoint = load_training_checkpoint(
            resume_path,
            model=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            map_location=device,
        )
        start_epoch = int(checkpoint.get("next_epoch", checkpoint.get("epoch", -1) + 1))
        resume_history = checkpoint.get("history")

    history = trainer.train(
        train_loader,
        val_loader,
        epochs=int(_as_dict(cfg.get("training", {})).get("epochs", 100)),
        start_epoch=start_epoch,
        resume_history=resume_history,
    )
    torch.save(history, output_dir / "history.pt")
    return history


def parse_args():
    parser = argparse.ArgumentParser(description="Train an ApexID re-identification model.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML config. Defaults to configs/default.yaml.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(config_path=args.config)
    train_from_config(cfg)


if __name__ == "__main__":
    main()
