#!/usr/bin/env python3
import argparse
import sys
import warnings
import copy
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader

from apexid.data import build_eval_transforms, build_train_transforms
from apexid.data.datasets.dataset import ApexID_Dataset
from apexid.data.species_metadata import (
    build_training_hierarchy,
    enrich_species_metadata,
)
from apexid.data.samplers import PKBatchSampler
from apexid.engine import (
    LegacyCheckpointWarning,
    Trainer,
    load_training_checkpoint,
)
from apexid.models import build_components
from apexid.models.model_factory import HIERARCHICAL_HEAD_NAMES
from apexid.utils.config import DEFAULT_CONFIG_PATH, load_config
from apexid.utils.device import get_device
from apexid.utils.provenance import (
    artifact_hashes,
    capture_environment,
    capture_git_state,
    command_record,
    require_clean_git,
    utc_now,
    write_json_atomic,
    write_yaml_atomic,
)
from apexid.utils.seed import DEFAULT_SEED, make_torch_generator, seed_everything, seed_worker


# Keep loss of exact resume reproducibility visible.
warnings.filterwarnings("always", category=LegacyCheckpointWarning)


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


def _optional_data_path(data_cfg, primary_key, legacy_key, fallback=None):
    """Read an optional data path from preferred, legacy, or fallback config."""
    value = data_cfg.get(primary_key)
    if value is None:
        value = data_cfg.get(legacy_key)
    if value in {None, ""}:
        return fallback
    return value


def _sibling_split_path(val_path, filename):
    """Return a sibling split path when the mixed validation CSV is configured."""
    if val_path in {None, ""}:
        return None
    return str(Path(val_path).with_name(filename))


def _build_metrics_label_map(*paths):
    """Build one metric label map across train and all validation CSVs."""
    sources = [
        source
        for source in paths
        if source is not None and not (isinstance(source, str) and source == "")
    ]
    frames = [ApexID_Dataset._standardize_columns(source) for source in sources]
    if not frames:
        raise ValueError("At least one CSV path is required to build metric labels.")
    return ApexID_Dataset.build_label_mapping(pd.concat(frames, ignore_index=True))


def _normalise_name(value):
    """Return a lower-cased string name helper."""
    if value is None:
        return None
    return str(value).strip().lower()


def _is_null_config_value(value):
    """Return True for empty YAML null-like sentinel values."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return True
    return False


def _resolve_resume_path(cfg, repo_root=None):
    """Resolve checkpoint resume path from config, including ``auto`` detection."""
    checkpoint_cfg = _as_dict(cfg.get("checkpoint", {}))
    resume_path = checkpoint_cfg.get("resume_path")
    if _is_null_config_value(resume_path):
        return None

    resume_key = str(resume_path).strip().lower()
    if resume_key != "auto":
        return str(resume_path)

    filename = checkpoint_cfg.get("filename", "last_checkpoint.pt")
    checkpoint_dir = checkpoint_cfg.get("output_dir")
    if checkpoint_dir in {None, ""}:
        experiment_dir = _as_dict(cfg.get("experiment", {})).get("output_dir", "outputs/apexid_run")
        checkpoint_dir = Path(experiment_dir) / "checkpoints"
    candidate = Path(checkpoint_dir)
    if not candidate.is_absolute() and repo_root is not None:
        candidate = Path(repo_root) / candidate
    candidate = candidate / filename
    if candidate.is_file():
        return str(candidate)
    return None


def _apply_hierarchy_config(cfg, train_dataset, num_classes, output_dir=None):
    """Inject hierarchy state derived exclusively from the training split."""
    cfg = copy.deepcopy(_as_dict(cfg))
    head_cfg = _as_dict(cfg.get("head", {}))
    if _normalise_name(head_cfg.get("name", "")) not in HIERARCHICAL_HEAD_NAMES:
        return cfg

    metadata = build_training_hierarchy(
        train_dataset.data_csv,
        train_dataset.label_to_idx,
        identity_column=train_dataset.identity_column,
        species_column="species",
    )
    if metadata.num_training_identities != int(num_classes):
        raise ValueError("ApexID hierarchy mapping does not match the training class count.")

    head_kwargs = dict(_as_dict(head_cfg.get("kwargs", {})))
    configured_balance = _normalise_name(
        head_kwargs.get("species_balance", "sqrt_inverse_identity_frequency")
    )
    if configured_balance != "sqrt_inverse_identity_frequency":
        raise ValueError(
            "ApexID supports species_balance='sqrt_inverse_identity_frequency' only."
        )
    head_kwargs["class_to_species"] = metadata.class_to_species
    head_kwargs["species_names"] = metadata.species_names
    head_kwargs["species_balance"] = configured_balance
    cfg["head"] = dict(head_cfg)
    cfg["head"]["kwargs"] = head_kwargs

    diagnostics = metadata.to_dict()
    enrichment_reports = dict(
        getattr(train_dataset, "species_enrichment_reports", {}) or {}
    )
    cfg["training_hierarchy"] = diagnostics
    cfg["species_enrichment"] = enrichment_reports
    if output_dir is not None:
        write_json_atomic(Path(output_dir) / "training_hierarchy.json", diagnostics)
        write_json_atomic(
            Path(output_dir) / "species_enrichment.json",
            enrichment_reports,
        )
    print(
        "ApexID training hierarchy: "
        f"identities={metadata.num_training_identities}, "
        f"species={metadata.num_training_species}, "
        f"known_parents={metadata.known_parent_identities} "
        f"({metadata.known_parent_percentage:.2f}%), "
        f"missing_parents={metadata.missing_parent_identities}, "
        f"IDs/species[min/median/max]={metadata.min_identities_per_species}/"
        f"{metadata.median_identities_per_species:g}/{metadata.max_identities_per_species}, "
        f"weights[min/mean/max]={metadata.min_species_weight:.4f}/"
        f"{metadata.mean_species_weight:.4f}/{metadata.max_species_weight:.4f}, "
        f"singleton_species={metadata.singleton_species}"
    )
    return cfg


def build_datasets(cfg):
    """Build train, gallery, and three validation datasets from configured CSV files.

    Args:
        cfg (dict): Loaded experiment config. Requires ``data.train_path``,
            ``data.val_known_path``, ``data.val_unknown_path``, and
            ``data.val_path`` or their legacy aliases.

    Returns:
        tuple: ``(train_dataset, validation_datasets, num_classes)`` where
        ``validation_datasets`` contains ``gallery``, ``known``, ``unknown``,
        and ``mixed`` datasets.
    """
    data_cfg = _as_dict(cfg.get("data", {}))
    train_path = _data_path(data_cfg, "train_path", "train_csv")
    val_path = _data_path(data_cfg, "val_path", "val_csv")
    val_known_path = _optional_data_path(
        data_cfg,
        "val_known_path",
        "val_known_csv",
        fallback=_sibling_split_path(val_path, "val_known.csv"),
    )
    val_unknown_path = _optional_data_path(
        data_cfg,
        "val_unknown_path",
        "val_unknown_csv",
        fallback=_sibling_split_path(val_path, "val_unknown.csv"),
    )
    if val_known_path in {None, ""}:
        raise ValueError("Missing data.val_known_path. Set it to val_known.csv.")
    if val_unknown_path in {None, ""}:
        raise ValueError("Missing data.val_unknown_path. Set it to val_unknown.csv.")
    root_dir = data_cfg.get("root_dir")

    train_transforms = build_train_transforms(config=cfg)
    eval_transforms = build_eval_transforms(config=cfg)

    uses_hierarchy = _normalise_name(_as_dict(cfg.get("head", {})).get("name", "")) in HIERARCHICAL_HEAD_NAMES
    split_sources = {
        "train": train_path,
        "val_known": val_known_path,
        "val_unknown": val_unknown_path,
        "val": val_path,
    }
    enrichment_reports = {}
    if uses_hierarchy:
        for split_name, source in list(split_sources.items()):
            standardized = ApexID_Dataset._standardize_columns(source)
            enriched, report = enrich_species_metadata(standardized, require_complete=False)
            split_sources[split_name] = enriched
            enrichment_reports[split_name] = report.to_dict()

    label_to_idx = ApexID_Dataset.build_label_mapping(split_sources["train"])
    metrics_label_map = _build_metrics_label_map(
        split_sources["train"],
        split_sources["val_known"],
        split_sources["val_unknown"],
        split_sources["val"],
    )
    unknown_label = _as_dict(cfg.get("training", {})).get("classification_ignore_index", -100)
    hierarchy_metadata = None
    if uses_hierarchy:
        hierarchy_metadata = build_training_hierarchy(
            split_sources["train"],
            label_to_idx,
        )
    species_to_idx = hierarchy_metadata.species_to_index if hierarchy_metadata else None

    train_dataset = ApexID_Dataset(
        split_sources["train"],
        transform=train_transforms,
        label_to_idx=label_to_idx,
        root_dir=root_dir,
        allow_unknown_identity=False,
        unknown_label=unknown_label,
        metrics_label_map=metrics_label_map,
        species_to_idx=species_to_idx,
        return_species_label=uses_hierarchy,
    )
    gallery_dataset = ApexID_Dataset(
        split_sources["train"],
        transform=eval_transforms,
        label_to_idx=label_to_idx,
        root_dir=root_dir,
        allow_unknown_identity=False,
        unknown_label=unknown_label,
        metrics_label_map=metrics_label_map,
        species_to_idx=species_to_idx,
        return_species_label=uses_hierarchy,
    )
    val_known_dataset = ApexID_Dataset(
        split_sources["val_known"],
        transform=eval_transforms,
        label_to_idx=label_to_idx,
        root_dir=root_dir,
        allow_unknown_identity=True,
        unknown_label=unknown_label,
        metrics_label_map=metrics_label_map,
        species_to_idx=species_to_idx,
        return_species_label=uses_hierarchy,
    )
    val_unknown_dataset = ApexID_Dataset(
        split_sources["val_unknown"],
        transform=eval_transforms,
        label_to_idx=label_to_idx,
        root_dir=root_dir,
        allow_unknown_identity=True,
        unknown_label=unknown_label,
        metrics_label_map=metrics_label_map,
        species_to_idx=species_to_idx,
        return_species_label=uses_hierarchy,
    )
    val_mixed_dataset = ApexID_Dataset(
        split_sources["val"],
        transform=eval_transforms,
        label_to_idx=label_to_idx,
        root_dir=root_dir,
        allow_unknown_identity=True,
        unknown_label=unknown_label,
        metrics_label_map=metrics_label_map,
        species_to_idx=species_to_idx,
        return_species_label=uses_hierarchy,
    )
    validation_datasets = {
        "gallery": gallery_dataset,
        "known": val_known_dataset,
        "unknown": val_unknown_dataset,
        "mixed": val_mixed_dataset,
    }
    train_dataset.species_enrichment_reports = enrichment_reports
    return train_dataset, validation_datasets, len(label_to_idx)


def build_dataloaders(cfg, train_dataset, validation_datasets, device):
    """Build train and three-split validation DataLoaders.

    Args:
        cfg (dict): Loaded experiment config.
        train_dataset (torch.utils.data.Dataset): Training dataset.
        validation_datasets (Mapping): Gallery/known/unknown/mixed datasets.
        device (torch.device): Selected training device.

    Returns:
        tuple: ``(train_loader, validation_loaders)``.
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

    validation_loaders = {}
    for split_name, dataset in validation_datasets.items():
        validation_loaders[split_name] = DataLoader(
            dataset,
            batch_size=val_batch_size,
            shuffle=False,
            drop_last=False,
            generator=make_torch_generator(seed),
            **common_kwargs,
        )
    return train_loader, validation_loaders


def build_optimizer(cfg, model, head):
    """Build the optimizer from config."""
    optimizer_cfg = _as_dict(cfg.get("optimizer", {}))
    name = str(optimizer_cfg.get("name", "adam")).lower()
    kwargs = dict(_as_dict(optimizer_cfg.get("kwargs", {})))
    shared_kwargs = {key: value for key, value in kwargs.items() if key != "lr"}

    backbone_lr = optimizer_cfg.get("backbone_lr", cfg.get("_finetune_backbone_lr"))
    head_lr = optimizer_cfg.get("head_lr", cfg.get("_finetune_head_lr"))
    if backbone_lr is not None or head_lr is not None:
        if backbone_lr is None or head_lr is None:
            raise ValueError("Set both optimizer.backbone_lr and optimizer.head_lr.")
        params = []
        for group_name, module, rate in (("backbone", model, backbone_lr), ("head", head, head_lr)):
            group = {"params": list(module.parameters()), "lr": float(rate), "param_group_name": group_name}
            historical_decay = cfg.get(f"_finetune_{group_name}_weight_decay")
            if historical_decay is not None:
                group["weight_decay"] = float(historical_decay)
            params.append(group)
        kwargs = shared_kwargs
    else:
        params = list(model.parameters()) + list(head.parameters())
    optimizers = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer '{name}'. Available: {', '.join(optimizers)}.")
    return optimizers[name](params, **kwargs)


class _LinearWarmup(torch.optim.lr_scheduler.LRScheduler):
    """Warm up each parameter group from the same absolute learning rate."""

    def __init__(self, optimizer, start_lr, total_iters):
        self.start_lrs = [float(start_lr)] * len(optimizer.param_groups)
        self.total_iters = int(total_iters)
        super().__init__(optimizer)

    def get_lr(self):
        fraction = min(self.last_epoch, self.total_iters) / self.total_iters
        return [start + fraction * (end - start) for start, end in zip(self.start_lrs, self.base_lrs)]

    def load_state_dict(self, state_dict):
        # Reuse historical relative warmup rates when resuming an older run.
        if "start_factor" in state_dict:
            state_dict = dict(state_dict)
            old_factor = state_dict.pop("start_factor")
            state_dict["start_lrs"] = [old_factor * lr for lr in state_dict["base_lrs"]]
        super().load_state_dict(state_dict)


def build_scheduler(cfg, optimizer, train_loader=None, total_epochs=None):
    """Build an optional learning-rate scheduler from config."""
    scheduler_cfg = _as_dict(cfg.get("scheduler", {}))
    name = scheduler_cfg.get("name")
    if name in {None, "", "none", "null"}:
        return None

    name = str(name).lower()
    kwargs = dict(_as_dict(scheduler_cfg.get("kwargs", {})))
    warmup_cfg = _as_dict(scheduler_cfg.get("warmup", {}))

    if name in {"step_lr", "steplr"}:
        return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    if name in {"cosine_annealing_lr", "cosineannealinglr", "cosine"}:
        epochs = int(total_epochs or _as_dict(cfg.get("training", {})).get("epochs", 100))
        if kwargs.get("T_max") in {None, ""}:
            kwargs["T_max"] = epochs
        if bool(warmup_cfg.get("enabled", False)):
            warmup_epochs = int(warmup_cfg.get("epochs", 0))
            if warmup_epochs <= 0:
                return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
            if warmup_epochs >= int(kwargs["T_max"]):
                raise ValueError("scheduler.warmup.epochs must be smaller than cosine T_max.")
            base_lrs = [group["lr"] for group in optimizer.param_groups]
            start_lr = float(warmup_cfg.get("start_lr", min(base_lrs) / 10.0))
            warmup = _LinearWarmup(optimizer, start_lr, warmup_epochs)
            cosine_kwargs = dict(kwargs)
            cosine_kwargs["T_max"] = int(kwargs["T_max"]) - warmup_epochs
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **cosine_kwargs)
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    if name in {"reduce_on_plateau", "reduce_lr_on_plateau", "reducelronplateau"}:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    if name in {"cyclic_lr", "cycliclr", "cyclic"}:
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


def build_trainer(
    cfg,
    model,
    head,
    criterion,
    optimizer,
    scheduler,
    device,
):
    """Build the ApexID Trainer from config-derived components."""
    training_cfg = _as_dict(cfg.get("training", {}))
    validation_cfg = _as_dict(cfg.get("validation", {}))
    checkpoint_cfg = _as_dict(cfg.get("checkpoint", {}))
    early_cfg = _as_dict(cfg.get("early_stopping", {}))
    experiment_dir = Path(_as_dict(cfg.get("experiment", {})).get("output_dir", "outputs/apexid_run"))
    checkpoint_dir = checkpoint_cfg.get("output_dir", experiment_dir / "checkpoints")

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
        checkpoint_dir=checkpoint_dir,
        checkpoint_every_n_epochs=int(checkpoint_cfg.get("every_n_epochs", 1)),
        checkpoint_filename=checkpoint_cfg.get("filename", "last_checkpoint.pt"),
        checkpoint_delete_previous=bool(checkpoint_cfg.get("delete_previous", True)),
        early_stopping_enabled=bool(early_cfg.get("enabled", True)),
        early_stopping_monitor=early_cfg.get("monitor", "mixed/NormAcc"),
        early_stopping_mode=early_cfg.get("mode", "max"),
        early_stopping_patience=int(early_cfg.get("patience", 7)),
        early_stopping_min_delta=float(early_cfg.get("min_delta", 0.0)),
        early_stopping_dir=early_cfg.get("output_dir", checkpoint_dir),
        early_stopping_full_model_name=early_cfg.get("full_model_name", "best_full_model.pt"),
        early_stopping_backbone_name=early_cfg.get("backbone_name", "best_backbone.pt"),
        early_stopping_metadata_name=early_cfg.get("metadata_name", "best_metadata.json"),
        metrics_base_seed=int(cfg.get("seed", DEFAULT_SEED) or DEFAULT_SEED),
        evaluation_config=_as_dict(cfg.get("evaluation", {})),
    )


def train_from_config(cfg, *, config_path=None, command=None, require_clean_git_state=False):
    """Run a full training job from a loaded config dictionary.

    Args:
        cfg (dict): Loaded experiment config.

    Returns:
        dict: Training history returned by ``Trainer.train``.
    """
    git_state = capture_git_state(REPO_ROOT)
    if require_clean_git_state:
        require_clean_git(git_state)
    seed = int(cfg.get("seed", DEFAULT_SEED) or DEFAULT_SEED)
    seed_everything(seed)

    device_cfg = _as_dict(cfg.get("device", {}))
    device = get_device(
        prefer_gpu=bool(device_cfg.get("prefer_gpu", True)),
        gpu_index=int(device_cfg.get("gpu_index", 0)),
    )

    output_dir = Path(_as_dict(cfg.get("experiment", {})).get("output_dir", "outputs/apexid_run"))
    output_dir.mkdir(parents=True, exist_ok=True)

    write_json_atomic(output_dir / "environment.json", capture_environment())
    write_yaml_atomic(output_dir / "config_resolved.yaml", cfg)
    data_cfg = _as_dict(cfg.get("data", {}))
    train_path = Path(_data_path(data_cfg, "train_path", "train_csv"))
    split_root = train_path.parent
    run_manifest = {
        "schema_version": 1,
        "run_type": "training",
        "status": "running",
        "started_at_utc": utc_now(),
        "command": command_record(command),
        "git": git_state,
        "seed": seed,
        "input_artifacts": artifact_hashes({
            "config": Path(config_path) if config_path else None,
            "requirements_lock": REPO_ROOT / "requirements-lock.txt",
            "train_csv": train_path,
            "val_known_csv": data_cfg.get("val_known_path") or data_cfg.get("val_known_csv"),
            "val_unknown_csv": data_cfg.get("val_unknown_path") or data_cfg.get("val_unknown_csv"),
            "val_csv": data_cfg.get("val_path") or data_cfg.get("val_csv"),
            "split_manifest": split_root / "split_manifest.csv",
            "split_summary": split_root / "split_summary.json",
            "metadata_data_root": Path(data_cfg.get("root_dir", "ApexID_Dataset")) / "metadata.csv",
            "metadata_repository_root": REPO_ROOT / "metadata.csv",
        }),
        "normalization": {
            "mean": _as_dict(data_cfg.get("transforms", {})).get("mean"),
            "std": _as_dict(data_cfg.get("transforms", {})).get("std"),
        },
    }
    write_json_atomic(output_dir / "run_manifest.json", run_manifest)

    train_dataset, validation_datasets, num_classes = build_datasets(cfg)
    cfg = _apply_hierarchy_config(
        cfg,
        train_dataset,
        num_classes,
        output_dir=output_dir,
    )
    write_yaml_atomic(output_dir / "config_resolved.yaml", cfg)
    train_loader, validation_loaders = build_dataloaders(cfg, train_dataset, validation_datasets, device)

    resume_path = _resolve_resume_path(cfg)
    if resume_path is not None:
        print(f"Resuming training from checkpoint: {resume_path}")
    run_manifest["dataset_counts"] = {
        "classes": int(num_classes),
        "train": int(len(train_dataset)),
        "gallery": int(len(validation_datasets["gallery"])),
        "val_known": int(len(validation_datasets["known"])),
        "val_unknown": int(len(validation_datasets["unknown"])),
        "val": int(len(validation_datasets["mixed"])),
    }

    model, head, criterion = build_components(cfg, num_classes=num_classes)
    run_manifest["model"] = {
        "backbone": _as_dict(cfg.get("model", {})).get("backbone"),
        "embedding_dim": _as_dict(cfg.get("model", {})).get("embedding_dim"),
        "pretrained_artifact": getattr(model.backbone, "pretrained_provenance", None),
    }
    write_json_atomic(output_dir / "run_manifest.json", run_manifest)
    model.to(device)
    head.to(device)
    optimizer = build_optimizer(cfg, model, head)
    scheduler = build_scheduler(cfg, optimizer, train_loader=train_loader)
    trainer = build_trainer(cfg, model, head, criterion, optimizer, scheduler, device)

    start_epoch = 0
    resume_history = None
    if resume_path not in {None, ""}:
        checkpoint = load_training_checkpoint(
            resume_path,
            model=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            map_location=device,
            early_stopping=trainer.early_stopping,
            data_loaders={"train": train_loader, "validation": validation_loaders},
        )
        start_epoch = int(checkpoint.get("next_epoch", checkpoint.get("epoch", -1) + 1))
        resume_history = checkpoint.get("history")

    history = trainer.train(
        train_loader,
        validation_loaders,
        epochs=int(_as_dict(cfg.get("training", {})).get("epochs", 100)),
        start_epoch=start_epoch,
        resume_history=resume_history,
    )
    torch.save(history, output_dir / "history.pt")
    checkpoint_dir = Path(_as_dict(cfg.get("checkpoint", {})).get("output_dir", output_dir / "checkpoints"))
    early_dir = Path(_as_dict(cfg.get("early_stopping", {})).get("output_dir", checkpoint_dir))
    run_manifest["status"] = "completed"
    run_manifest["completed_at_utc"] = utc_now()
    run_manifest["output_artifacts"] = artifact_hashes({
        "history": output_dir / "history.pt",
        "last_checkpoint": checkpoint_dir / _as_dict(cfg.get("checkpoint", {})).get("filename", "last_checkpoint.pt"),
        "best_full_model": early_dir / _as_dict(cfg.get("early_stopping", {})).get("full_model_name", "best_full_model.pt"),
        "best_backbone": early_dir / _as_dict(cfg.get("early_stopping", {})).get("backbone_name", "best_backbone.pt"),
        "best_metadata": early_dir / _as_dict(cfg.get("early_stopping", {})).get("metadata_name", "best_metadata.json"),
        "resolved_config": output_dir / "config_resolved.yaml",
        "environment": output_dir / "environment.json",
    })
    write_json_atomic(output_dir / "run_manifest.json", run_manifest)
    return history


def parse_args():
    parser = argparse.ArgumentParser(description="Train an ApexID re-identification model.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML config. Defaults to configs/default.yaml.",
    )
    parser.add_argument(
        "--require-clean-git",
        action="store_true",
        help="Reject dirty or unverifiable Git state before training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(config_path=args.config)
    train_from_config(
        cfg,
        config_path=args.config,
        command=sys.argv,
        require_clean_git_state=args.require_clean_git,
    )


if __name__ == "__main__":
    main()
