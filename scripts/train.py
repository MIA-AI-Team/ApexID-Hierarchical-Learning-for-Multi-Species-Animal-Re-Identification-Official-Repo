#!/usr/bin/env python3
import argparse
import logging
import os
import sys
import warnings
import copy
from pathlib import Path

import numpy as np
import pandas as pd


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
from apexid.data.species_metadata import (
    build_training_hierarchy,
    enrich_species_metadata,
)
from apexid.data.samplers import PKBatchSampler
from apexid.engine import (
    LegacyCheckpointWarning,
    Trainer,
    build_experiment_logger,
    load_training_checkpoint,
)
from apexid.models import build_components
from apexid.training.finetune import (
    apply_finetune_phase,
    apply_suggest_lr_from_range_test,
    collect_backbone_params,
    collect_head_params,
    finetune_lr_range_test_cfg,
    is_finetune_active,
    resolve_training_profile,
    should_run_lr_range_test,
)
from apexid.utils.config import DEFAULT_CONFIG_PATH, load_config
from apexid.utils.device import get_device
from apexid.utils.embedding_umap import EmbeddingUMAPPlotter
from apexid.utils.lr_range_test import run_lr_range_test, write_lr_range_test_artifacts
from apexid.utils.provenance import (
    artifact_hashes,
    capture_environment,
    capture_git_state,
    command_record,
    require_clean_git,
    utc_now,
    validate_normalization_stats,
    write_json_atomic,
    write_yaml_atomic,
)
from apexid.utils.seed import DEFAULT_SEED, make_torch_generator, seed_everything, seed_worker


# Keep this scientific-resume warning visible despite third-party warning suppression.
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


def _class_counts_from_train_dataset(train_dataset, num_classes):
    """Return per-class counts aligned with training class indices."""
    identity_series = train_dataset.data_csv[train_dataset.identity_column]
    label_to_idx = train_dataset.label_to_idx
    class_indices = [label_to_idx[identity] for identity in identity_series if identity in label_to_idx]
    if not class_indices:
        raise ValueError("Unable to compute class frequencies: training dataset is empty.")
    counts = np.bincount(np.asarray(class_indices, dtype=np.int64), minlength=int(num_classes)).astype(np.float32)
    if counts.shape[0] != int(num_classes):
        raise ValueError("Computed class counts do not match the configured number of classes.")
    if np.any(counts <= 0):
        raise ValueError(
            "Class frequency margins require at least one sample per class. "
            "Found one or more classes with zero samples."
        )
    return counts


def _build_frequency_bin_margins(train_dataset, num_classes, margin_min=0.2, margin_max=0.5):
    """Compute MiewID-style per-class margins from class frequencies."""
    margin_min = float(margin_min)
    margin_max = float(margin_max)
    if not 0 < margin_min < np.pi / 2:
        raise ValueError("head.kwargs.margin_min must be in the open interval (0, pi/2).")
    if not 0 < margin_max < np.pi / 2:
        raise ValueError("head.kwargs.margin_max must be in the open interval (0, pi/2).")
    if margin_max <= margin_min:
        raise ValueError("head.kwargs.margin_max must be greater than head.kwargs.margin_min.")

    class_counts = _class_counts_from_train_dataset(train_dataset, num_classes)
    # Match MiewID behavior: rarer classes get larger margins.
    raw = np.sqrt(1.0 / np.sqrt(class_counts))
    raw_min = float(raw.min())
    raw_max = float(raw.max())
    if np.isclose(raw_max, raw_min):
        margins = np.full((int(num_classes),), margin_max, dtype=np.float32)
    else:
        margins = (raw - raw_min) / (raw_max - raw_min)
        margins = margins * (margin_max - margin_min) + margin_min
    return margins.tolist()


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


def _apply_subcenter_margin_config(cfg, train_dataset, num_classes):
    """Resolve sub-center margin modes and inject explicit per-class margins."""
    cfg = copy.deepcopy(_as_dict(cfg))
    head_cfg = _as_dict(cfg.get("head", {}))
    head_name = _normalise_name(head_cfg.get("name", ""))
    if head_name not in {"arcface_subcenter_dynamic", "arcface_subcenter", "subcenter_arcface"}:
        return cfg

    head_kwargs = dict(_as_dict(head_cfg.get("kwargs", {})))
    if head_kwargs.get("margins") is not None:
        return cfg

    margins_source = _normalise_name(head_kwargs.get("margins_source", "uniform"))
    if margins_source != "frequency_bins":
        return cfg

    margin_min = head_kwargs.get("margin_min", 0.2)
    margin_max = head_kwargs.get("margin_max", head_kwargs.get("margin_default", 0.3))
    head_kwargs["margins"] = _build_frequency_bin_margins(
        train_dataset=train_dataset,
        num_classes=num_classes,
        margin_min=margin_min,
        margin_max=margin_max,
    )

    cfg["head"] = dict(head_cfg)
    cfg["head"]["kwargs"] = head_kwargs
    return cfg


BHCA_HEAD_NAMES = {
    "bhca_arcface",
    "bhca_a",
    "bhca-a",
    "bhca_subcenter_arcface",
    "bhca_sca",
    "bhca-sca",
    "bhca_subcenter_sphereface2",
    "bhca_scsf2",
    "bhca-scsf2",
}


def _apply_bhca_hierarchy_config(cfg, train_dataset, num_classes, output_dir=None):
    """Inject hierarchy state derived exclusively from the training split."""
    cfg = copy.deepcopy(_as_dict(cfg))
    head_cfg = _as_dict(cfg.get("head", {}))
    if _normalise_name(head_cfg.get("name", "")) not in BHCA_HEAD_NAMES:
        return cfg

    metadata = build_training_hierarchy(
        train_dataset.data_csv,
        train_dataset.label_to_idx,
        identity_column=train_dataset.identity_column,
        species_column="species",
    )
    if metadata.num_training_identities != int(num_classes):
        raise ValueError("BHCA hierarchy mapping does not match the training class count.")

    head_kwargs = dict(_as_dict(head_cfg.get("kwargs", {})))
    configured_balance = _normalise_name(
        head_kwargs.get("species_balance", "sqrt_inverse_identity_frequency")
    )
    if configured_balance != "sqrt_inverse_identity_frequency":
        raise ValueError(
            "BHCA currently supports species_balance='sqrt_inverse_identity_frequency' only."
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
    cfg["bhca_training_diagnostics"] = diagnostics
    cfg["bhca_species_enrichment"] = enrichment_reports
    if output_dir is not None:
        write_json_atomic(Path(output_dir) / "bhca_training_diagnostics.json", diagnostics)
        write_json_atomic(
            Path(output_dir) / "bhca_species_enrichment.json",
            enrichment_reports,
        )
    print(
        "BHCA training hierarchy: "
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

    is_bhca = _normalise_name(_as_dict(cfg.get("head", {})).get("name", "")) in BHCA_HEAD_NAMES
    split_sources = {
        "train": train_path,
        "val_known": val_known_path,
        "val_unknown": val_unknown_path,
        "val": val_path,
    }
    enrichment_reports = {}
    if is_bhca:
        for split_name, source in list(split_sources.items()):
            standardized = ApexID_Dataset._standardize_columns(source)
            enriched, report = enrich_species_metadata(standardized, require_complete=True)
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
    if is_bhca:
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
        return_species_label=is_bhca,
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
        return_species_label=is_bhca,
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
        return_species_label=is_bhca,
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
        return_species_label=is_bhca,
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
        return_species_label=is_bhca,
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

    if is_finetune_active(cfg):
        backbone_lr = float(cfg.get("_finetune_backbone_lr", kwargs.get("lr", 1.0e-5)))
        head_lr = float(cfg.get("_finetune_head_lr", backbone_lr * 10.0))
        backbone_group = {
            "params": collect_backbone_params(model),
            "lr": backbone_lr,
            "param_group_name": "backbone",
        }
        head_group = {
            "params": collect_head_params(head),
            "lr": head_lr,
            "param_group_name": "head",
        }
        backbone_weight_decay = cfg.get("_finetune_backbone_weight_decay")
        head_weight_decay = cfg.get("_finetune_head_weight_decay")
        if backbone_weight_decay not in {None, ""}:
            backbone_group["weight_decay"] = float(backbone_weight_decay)
        if head_weight_decay not in {None, ""}:
            head_group["weight_decay"] = float(head_weight_decay)
        param_groups = [
            backbone_group,
            head_group,
        ]
        if name == "adam":
            return torch.optim.Adam(param_groups, **shared_kwargs)
        if name == "adamw":
            return torch.optim.AdamW(param_groups, **shared_kwargs)
        if name == "sgd":
            return torch.optim.SGD(param_groups, **shared_kwargs)
        raise ValueError("Unknown optimizer '{}'. Available: adam, adamw, sgd.".format(name))

    params = list(model.parameters()) + list(head.parameters())
    if name == "adam":
        return torch.optim.Adam(params, **kwargs)
    if name == "adamw":
        return torch.optim.AdamW(params, **kwargs)
    if name == "sgd":
        return torch.optim.SGD(params, **kwargs)
    raise ValueError("Unknown optimizer '{}'. Available: adam, adamw, sgd.".format(name))


def build_scheduler(cfg, optimizer, train_loader=None, overrides=None, total_epochs=None):
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
            start_factors = [
                max(start_lr / max(float(lr), 1e-12), 1e-8) for lr in base_lrs
            ]
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=min(start_factors),
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
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
    lr_test_cfg = finetune_lr_range_test_cfg(cfg)
    if not should_run_lr_range_test(cfg, resume_path=resume_path):
        return None

    mode = str(lr_test_cfg.get("mode", "auto_apply")).lower()
    allowed_modes = {"auto_apply", "report_only", "suggest_lr"}
    if mode not in allowed_modes:
        raise ValueError("lr_range_test.mode must be one of: auto_apply, report_only, suggest_lr.")

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

    if mode == "suggest_lr" or is_finetune_active(cfg):
        apply_suggest_lr_from_range_test(cfg, optimizer, result)
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
        early_stopping_monitor=early_cfg.get("monitor", "mixed/NormAcc"),
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
        evaluation_config=_as_dict(cfg.get("evaluation", {})),
        finetune_config=_as_dict(cfg.get("finetune", {})) if is_finetune_active(cfg) else None,
    )


def train_from_config(cfg, *, config_path=None, command=None, require_clean_git_state=False):
    """Run a full training job from a loaded config dictionary.

    Args:
        cfg (dict): Loaded experiment config.

    Returns:
        dict: Training history returned by ``Trainer.train``.
    """
    cfg = resolve_training_profile(cfg)
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
    normalization_path = data_cfg.get("normalization_stats_path")
    if normalization_path in {None, ""}:
        normalization_path = split_root / "normalization_stats.json"
    split_manifest_path = split_root / "split_manifest.csv"
    normalization_record = None
    if Path(normalization_path).is_file():
        normalization_record = validate_normalization_stats(
            normalization_path,
            train_csv=train_path,
            split_manifest=split_manifest_path,
        )
        normalization_record = dict(normalization_record) | {
            "validated_stats_file": True,
            "path": str(normalization_path),
        }
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
            "split_manifest": split_manifest_path,
            "split_summary": split_root / "split_summary.json",
            "normalization_stats": normalization_path,
            "metadata_data_root": Path(data_cfg.get("root_dir", "ApexID_Dataset")) / "metadata.csv",
            "metadata_repository_root": REPO_ROOT / "metadata.csv",
        }),
        "normalization": normalization_record or {
            "mean": _as_dict(data_cfg.get("transforms", {})).get("mean"),
            "std": _as_dict(data_cfg.get("transforms", {})).get("std"),
            "validated_stats_file": False,
        },
    }
    write_json_atomic(output_dir / "run_manifest.json", run_manifest)

    train_dataset, validation_datasets, num_classes = build_datasets(cfg)
    cfg = _apply_subcenter_margin_config(cfg, train_dataset, num_classes)
    cfg = _apply_bhca_hierarchy_config(
        cfg,
        train_dataset,
        num_classes,
        output_dir=output_dir,
    )
    write_yaml_atomic(output_dir / "config_resolved.yaml", cfg)
    train_loader, validation_loaders = build_dataloaders(cfg, train_dataset, validation_datasets, device)

    checkpoint_cfg = _as_dict(cfg.get("checkpoint", {}))
    resume_path = _resolve_resume_path(cfg, repo_root=REPO_ROOT)
    if resume_path is not None:
        print(f"Resuming training from checkpoint: {resume_path}")
    tracking_metadata = {
        "num_classes": int(num_classes),
        "train_dataset_size": int(len(train_dataset)),
        "gallery_dataset_size": int(len(validation_datasets["gallery"])),
        "val_known_dataset_size": int(len(validation_datasets["known"])),
        "val_unknown_dataset_size": int(len(validation_datasets["unknown"])),
        "val_dataset_size": int(len(validation_datasets["mixed"])),
        "device": str(device),
        "resume_path": resume_path,
    }
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
    apply_finetune_phase(model, head, cfg, epoch=0)
    optimizer = build_optimizer(cfg, model, head)
    training_epochs = int(_as_dict(cfg.get("training", {})).get("epochs", 100))
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
    scheduler = build_scheduler(
        cfg,
        optimizer,
        train_loader=train_loader,
        overrides=scheduler_overrides,
        total_epochs=training_epochs,
    )
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
