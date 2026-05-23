#!/usr/bin/env python3
"""Print data-loading and identity-label sanity checks for ApexID training."""

import argparse
import copy
import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apexid.data.datasets.dataset import ApexID_Dataset
from apexid.engine.validator import _unpack_batch
from apexid.utils.config import DEFAULT_CONFIG_PATH, load_config
from apexid.utils.seed import DEFAULT_SEED
from scripts.train import build_dataloaders, build_datasets


def _as_dict(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "items"):
        return dict(value.items())
    raise TypeError(f"Expected a dict-like value, got {type(value).__name__}.")


def _data_path(data_cfg, primary_key, legacy_key):
    value = data_cfg.get(primary_key)
    if value is None:
        value = data_cfg.get(legacy_key)
    if value in {None, ""}:
        raise ValueError(f"Missing data.{primary_key} / data.{legacy_key}.")
    return str(value)


def _source_column(df):
    if "species" in df.columns:
        return "species"
    if "dataset" in df.columns:
        return "dataset"
    return None


def _resolve_path(path, root_dir=None):
    path = str(path)
    if os.path.exists(path):
        return path
    if root_dir is not None and not os.path.isabs(path):
        candidate = os.path.join(str(root_dir), path)
        if os.path.exists(candidate):
            return candidate
    if root_dir is not None:
        candidate = os.path.join(str(root_dir), os.path.basename(path))
        if os.path.exists(candidate):
            return candidate
    return path


def _missing_path_count(df, root_dir, path_column="path"):
    if path_column not in df.columns:
        return None
    missing = 0
    for path in df[path_column].tolist():
        if not os.path.exists(_resolve_path(path, root_dir=root_dir)):
            missing += 1
    return missing


def _print_split_summary(name, df, root_dir):
    source = _source_column(df)
    print(f"\n[{name}]")
    print(f"rows: {len(df)}")
    print(f"columns: {list(df.columns)}")
    if source is not None:
        print(f"{source} counts: {df[source].value_counts(dropna=False).sort_index().to_dict()}")
        print(f"identities per {source}: {df.groupby(source)['identity'].nunique().sort_index().to_dict()}")
    if "split" in df.columns:
        print(f"split counts: {df['split'].value_counts(dropna=False).to_dict()}")
    if "val_type" in df.columns:
        print(f"val_type counts: {df['val_type'].value_counts(dropna=False).to_dict()}")
    print(f"unique raw identities: {df['identity'].nunique(dropna=True)}")
    print(f"unique global identities: {df['identity'].nunique(dropna=True)}")
    print(f"missing image paths: {_missing_path_count(df, root_dir)}")

    counts = df.groupby("identity").size()
    if not counts.empty:
        print(
            "images per identity: "
            f"min={int(counts.min())}, median={float(counts.median()):.1f}, "
            f"max={int(counts.max())}"
        )


def _mapping_report(name, mapping):
    label_to_identities = defaultdict(set)
    identity_to_labels = defaultdict(set)
    for identity, label in mapping.items():
        label = int(label)
        label_to_identities[label].add(identity)
        identity_to_labels[identity].add(label)

    label_conflicts = {k: v for k, v in label_to_identities.items() if len(v) != 1}
    identity_conflicts = {k: v for k, v in identity_to_labels.items() if len(v) != 1}
    status = "PASS" if not label_conflicts and not identity_conflicts else "FAIL"

    print(f"\n[{name}]")
    print(f"unique global identities: {len(mapping)}")
    print(f"unique numeric labels: {len(label_to_identities)}")
    print(f"label_to_identity bijection: {status}")
    print(f"identity_to_label bijection: {status}")
    if label_conflicts:
        print(f"numeric label conflicts: {list(label_conflicts.items())[:5]}")
    if identity_conflicts:
        print(f"identity label conflicts: {list(identity_conflicts.items())[:5]}")


def _mapping_examples(df, label_to_idx, max_examples):
    source = _source_column(df)
    print(f"\n[first {max_examples} train mapping examples]")
    for row_idx, row in df.head(max_examples).iterrows():
        identity = row["identity"]
        source_value = row[source] if source is not None else "<none>"
        label = label_to_idx.get(identity, "<unknown>")
        print(
            f"{row_idx}: source={source_value} raw_identity={identity} "
            f"global_identity={identity} numeric_label={label} path={row['path']}"
        )


def _train_val_membership(train_df, val_df):
    train_ids = set(train_df["identity"].dropna().tolist())
    val_ids = set(val_df["identity"].dropna().tolist())
    print("\n[train/val identity membership]")
    print(f"train identities: {len(train_ids)}")
    print(f"val identities: {len(val_ids)}")
    print(f"known val identities in train: {len(train_ids & val_ids)}")
    print(f"val-only identities: {len(val_ids - train_ids)}")
    print(f"train-only identities: {len(train_ids - val_ids)}")
    if "val_type" in val_df.columns:
        membership = pd.crosstab(
            val_df["val_type"],
            val_df["identity"].isin(train_ids),
            rownames=["val_type"],
            colnames=["in_train"],
        )
        print("val_type vs in_train:")
        print(membership.to_string())


def _short_list(values, max_items):
    return [values[i] for i in range(min(len(values), max_items))]


def _batch_report(cfg, train_dataset, val_dataset, num_classes, max_examples):
    batch_cfg = copy.deepcopy(cfg)
    batch_cfg.setdefault("dataloader", {})
    batch_cfg["dataloader"]["num_workers"] = 0
    batch_cfg["dataloader"]["persistent_workers"] = False
    batch_cfg["dataloader"]["pin_memory"] = False

    train_loader, _ = build_dataloaders(
        batch_cfg,
        train_dataset,
        val_dataset,
        torch.device("cpu"),
    )
    images, cls_labels, metric_labels = _unpack_batch(next(iter(train_loader)))
    valid_range = bool((cls_labels >= 0).all() and (cls_labels < int(num_classes)).all())
    cls_names = [train_dataset.idx_to_label.get(int(label), "<unknown>") for label in cls_labels.tolist()]
    metric_names = [
        train_dataset.metric_idx_to_label.get(int(label), "<unknown>")
        for label in metric_labels.tolist()
    ]

    print("\n[first DataLoader batch]")
    print(f"image shape: {tuple(images.shape)}")
    print(f"cls label dtype: {cls_labels.dtype}")
    print(f"cls labels first {max_examples}: {_short_list(cls_labels.tolist(), max_examples)}")
    print(f"cls global identities first {max_examples}: {_short_list(cls_names, max_examples)}")
    print(f"metric labels first {max_examples}: {_short_list(metric_labels.tolist(), max_examples)}")
    print(f"metric global identities first {max_examples}: {_short_list(metric_names, max_examples)}")
    print(f"every cls label in [0, {int(num_classes) - 1}]: {valid_range}")


def run_debug(config_path, max_examples=20, skip_batch=False):
    cfg = load_config(config_path=config_path)
    data_cfg = _as_dict(cfg.get("data", {}))
    train_path = _data_path(data_cfg, "train_path", "train_csv")
    val_path = _data_path(data_cfg, "val_path", "val_csv")
    root_dir = data_cfg.get("root_dir")

    train_df = ApexID_Dataset._standardize_columns(train_path).reset_index(drop=True)
    val_df = ApexID_Dataset._standardize_columns(val_path).reset_index(drop=True)
    label_to_idx = ApexID_Dataset.build_label_mapping(train_df)
    metrics_label_map = ApexID_Dataset.build_label_mapping_union(train_df, val_df)

    print("[config]")
    print(f"config path: {config_path}")
    print(f"train csv: {train_path}")
    print(f"val csv: {val_path}")
    print(f"root_dir: {root_dir}")
    print(f"seed: {cfg.get('seed', DEFAULT_SEED)}")

    _print_split_summary("train CSV", train_df, root_dir)
    _print_split_summary("val CSV", val_df, root_dir)
    _mapping_report("classification label_to_idx", label_to_idx)
    _mapping_report("metric label map train+val", metrics_label_map)
    _mapping_examples(train_df, label_to_idx, max_examples=max_examples)
    _train_val_membership(train_df, val_df)

    if skip_batch:
        print("\n[first DataLoader batch]")
        print("skipped by --skip-batch")
        return

    train_dataset, val_dataset, num_classes = build_datasets(cfg)
    print("\n[constructed datasets]")
    print(f"num_classes: {num_classes}")
    print(f"train dataset rows: {len(train_dataset)}")
    if isinstance(val_dataset, dict):
        for split_name, split_dataset in val_dataset.items():
            print(f"{split_name} dataset rows: {len(split_dataset)}")
    else:
        print(f"val dataset rows: {len(val_dataset)}")
    print(f"PK sampler identities available: {len(train_dataset.identity_to_indices)}")
    _batch_report(cfg, train_dataset, val_dataset, num_classes, max_examples=max_examples)


def parse_args():
    parser = argparse.ArgumentParser(description="Audit ApexID CSV-to-loss label flow.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to an ApexID YAML config.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=20,
        help="Number of row/batch examples to print.",
    )
    parser.add_argument(
        "--skip-batch",
        "--skip-batches",
        action="store_true",
        dest="skip_batch",
        help="Skip loading one transformed DataLoader batch.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_debug(
        config_path=args.config,
        max_examples=int(args.max_examples),
        skip_batch=bool(args.skip_batch),
    )


if __name__ == "__main__":
    main()
