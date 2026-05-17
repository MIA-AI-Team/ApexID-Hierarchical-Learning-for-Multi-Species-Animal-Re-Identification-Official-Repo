#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader

from apexid.data import build_eval_transforms, build_train_transforms
from apexid.data.datasets.dataset import ApexID_Dataset
from apexid.data.samplers import PKBatchSampler
from apexid.engine import Trainer, load_training_checkpoint
from apexid.models import build_components
from apexid.utils.config import DEFAULT_CONFIG_PATH, load_config
from apexid.utils.device import get_device
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


def build_scheduler(cfg, optimizer):
    """Build an optional learning-rate scheduler from config."""
    scheduler_cfg = _as_dict(cfg.get("scheduler", {}))
    name = scheduler_cfg.get("name")
    if name in {None, "", "none", "null"}:
        return None

    name = str(name).lower()
    kwargs = _as_dict(scheduler_cfg.get("kwargs", {}))
    if name in {"step_lr", "steplr"}:
        return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    if name in {"cosine_annealing_lr", "cosineannealinglr", "cosine"}:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    if name in {"reduce_on_plateau", "reduce_lr_on_plateau", "reducelronplateau"}:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    raise ValueError(
        "Unknown scheduler '{}'. Available: step_lr, cosine_annealing_lr, "
        "reduce_on_plateau.".format(name)
    )


def build_trainer(cfg, model, head, criterion, optimizer, scheduler, device):
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
        gradient_clip_norm=training_cfg.get("gradient_clip_norm"),
        validation_every_n_epochs=int(validation_cfg.get("every_n_epochs", 1)),
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

    model, head, criterion = build_components(cfg, num_classes=num_classes)
    optimizer = build_optimizer(cfg, model, head)
    scheduler = build_scheduler(cfg, optimizer)
    trainer = build_trainer(cfg, model, head, criterion, optimizer, scheduler, device)

    checkpoint_cfg = _as_dict(cfg.get("checkpoint", {}))
    resume_path = checkpoint_cfg.get("resume_path")
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
