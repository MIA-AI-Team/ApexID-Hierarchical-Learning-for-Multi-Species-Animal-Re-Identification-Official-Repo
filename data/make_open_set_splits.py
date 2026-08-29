#!/usr/bin/env python3
"""Build or validate the ApexID Observation-Aware Split V2."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import fields
from pathlib import Path

import yaml
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apexid.data.split_v2 import (  # noqa: E402
    COMPONENT_SPLITS,
    FIELDNAMES,
    NEAR_DUPLICATE_FIELDS,
    OUTPUT_SPLITS,
    FeatureSpec,
    SplitConfig,
    annotate_dates,
    annotate_explicit_groups,
    annotate_image_hashes,
    audit_cross_split_near_duplicates,
    build_features_by_index,
    build_summary,
    choose_unknown_keys,
    group_rows_by_identity,
    identity_keys,
    load_or_extract_feature_cache,
    make_observation_aware_splits,
    manifest_rows,
    plan_identity_methods,
    public_row,
    read_rows,
    read_split,
    sha256_file,
    validate_splits,
    validate_written_v2,
    write_v2_artifacts,
)


DEFAULT_METADATA = Path("metadata.csv")
DEFAULT_DATA_ROOT = Path("ApexID_Dataset")
DEFAULT_OUTPUT_DIR = Path("ApexID_Dataset/open_set_splits_v2")
DEFAULT_FEATURE_CACHE = Path("ApexID_Dataset/split_v2_feature_cache")
DEFAULT_BASELINE_DIR = Path("ApexID_Dataset/open_set_splits")
DEFAULT_CONFIG = Path("data/split_v2.yaml")


def load_split_config(path: Path) -> SplitConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    raw = raw.get("split_v2", raw)
    feature_values = raw.pop("feature", {})
    feature = FeatureSpec(**feature_values)
    allowed = {item.name for item in fields(SplitConfig)} - {"feature"}
    unexpected = sorted(set(raw) - allowed)
    if unexpected:
        raise ValueError(f"Unexpected Split V2 configuration keys: {unexpected}")
    if "enabled_parsers" in raw:
        raw["enabled_parsers"] = tuple(raw["enabled_parsers"])
    return SplitConfig(feature=feature, **raw)


def _apply_overrides(config: SplitConfig, args: argparse.Namespace) -> SplitConfig:
    values = {
        "seed": args.seed,
        "unknown_identity_ratio": args.unknown_identity_ratio,
        "known_image_ratio": args.known_image_ratio,
        "max_unknown_images_per_identity": args.max_unknown_images_per_identity,
        "default_similarity_threshold": args.default_similarity_threshold,
    }
    for name, value in values.items():
        if value is not None:
            setattr(config, name, value)
    return config


def make_splits(
    rows,
    seed=42,
    unknown_identity_ratio=0.125,
    known_image_ratio=0.225,
    max_unknown_images_per_identity=100,
    features_by_index=None,
    **_ignored,
):
    """Compatibility entry point for already annotated programmatic rows."""
    config = SplitConfig(
        seed=seed,
        unknown_identity_ratio=unknown_identity_ratio,
        known_image_ratio=known_image_ratio,
        max_unknown_images_per_identity=max_unknown_images_per_identity,
    )
    for row in rows:
        row.setdefault("_parsed_date", "")
        row.setdefault("_date_format", "missing")
        row.setdefault("_image_sha256", f"synthetic-{row['_idx']}")
        row.setdefault("_duplicate_group_id", "")
        row.setdefault("_explicit_source", "")
        row.setdefault("_explicit_group_id", "")
        row.setdefault("_similarity_cluster_id", "")
    by_key = group_rows_by_identity(rows)
    _, unknown = choose_unknown_keys(by_key, seed, unknown_identity_ratio)
    methods, _ = plan_identity_methods(by_key, unknown)
    if features_by_index is None:
        features_by_index = {}
        fallback_indices = [
            int(row["_idx"])
            for key, items in by_key.items()
            if methods[key] == "similarity_fallback"
            for row in items
        ]
        row_by_index = {int(row["_idx"]): row for row in rows}
        for index in fallback_indices:
            digest = hashlib.sha256(
                str(row_by_index[index]["path"]).encode("utf-8")
            ).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
            feature = rng.normal(size=64).astype(np.float32)
            feature /= np.linalg.norm(feature)
            features_by_index[index] = feature
    return make_observation_aware_splits(
        rows, config, methods, unknown, features_by_index
    )


def write_splits(output_dir, splits, clear_existing=False):
    """Compatibility writer that never clears arbitrary files."""
    if clear_existing:
        raise ValueError(
            "Split V2 never clears an output directory; use explicit atomic overwrite"
        )
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing occupied output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    import csv

    for name in OUTPUT_SPLITS:
        with (output_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(public_row(row) for row in splits[name])


def validate_outputs(output_dir, source_rows=None, root_dir=None, check_images=True):
    """Compatibility validation for the six legacy CSVs."""
    del root_dir, check_images
    output_dir = Path(output_dir)
    splits = {name: read_split(output_dir / f"{name}.csv") for name in OUTPUT_SPLITS}
    if splits["val"] != splits["val_known"] + splits["val_unknown"]:
        raise ValueError("val.csv is not exactly val_known.csv + val_unknown.csv")
    train_ids = identity_keys(splits["train"])
    known_ids = identity_keys(splits["val_known"])
    unknown_ids = identity_keys(splits["val_unknown"])
    if train_ids & unknown_ids or known_ids - train_ids:
        raise ValueError("Identity-disjointness invariants failed")
    if source_rows is not None:
        source_paths = Counter(str(row["path"]) for row in source_rows)
        split_paths = Counter(
            row["path"] for name in COMPONENT_SPLITS for row in splits[name]
        )
        if source_paths != split_paths:
            raise ValueError("Component splits do not account for source rows exactly once")
    return {f"{name}_rows": len(splits[name]) for name in OUTPUT_SPLITS}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Create the ApexID Observation-Aware Open-Set Split V2."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--metadata",
        "--source",
        dest="metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help="Authoritative metadata CSV; --source is a compatibility alias.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-cache", type=Path, default=DEFAULT_FEATURE_CACHE)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--feature-checkpoint", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--unknown-identity-ratio", type=float)
    parser.add_argument("--known-image-ratio", type=float)
    parser.add_argument("--max-unknown-images-per-identity", type=int)
    parser.add_argument("--default-similarity-threshold", type=float)
    parser.add_argument("--recompute-features", action="store_true")
    parser.add_argument("--verify-images", action="store_true")
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Validate an existing V2 destination without changing files.",
    )
    parser.add_argument(
        "--overwrite-v2",
        action="store_true",
        help="Atomically replace only the named V2 directory; V1 is never touched.",
    )
    parser.add_argument("--near-duplicate-audit", action="store_true")
    parser.add_argument("--dhash-threshold", type=int)
    parser.add_argument("--phash-threshold", type=int)
    return parser.parse_args(argv)


def _print_summary(summary):
    for name in OUTPUT_SPLITS:
        counts = summary["splits"][name]
        print(f"{name}: rows={counts['rows']} identities={counts['identities']}")
    print("date_statistics=" + json.dumps(summary["date_statistics"], sort_keys=True))
    print("method_statistics=" + json.dumps(summary["method_statistics"], sort_keys=True))
    print("integrity=passed")


def _crossing_counts(train_rows, val_rows):
    exact_groups = {}
    date_groups = {}
    for split, items in (("train", train_rows), ("val_known", val_rows)):
        for row in items:
            exact_groups.setdefault(row["_image_sha256"], set()).add(split)
            date = row.get("_parsed_date", "")
            if date:
                date_groups.setdefault((row["_key"], date), set()).add(split)
    return {
        "exact_content_groups_crossing_train_val_known": sum(
            values == {"train", "val_known"} for values in exact_groups.values()
        ),
        "identity_date_groups_crossing_train_val_known": sum(
            values == {"train", "val_known"} for values in date_groups.values()
        ),
    }


def _baseline_comparison(baseline_dir, rows, splits):
    baseline_dir = Path(baseline_dir)
    required = [baseline_dir / f"{name}.csv" for name in OUTPUT_SPLITS]
    if not all(path.is_file() for path in required):
        return {"available": False, "path": str(baseline_dir)}
    by_path = {str(row["path"]): row for row in rows}
    baseline = {}
    for name in OUTPUT_SPLITS:
        public = read_split(baseline_dir / f"{name}.csv")
        missing = [row["path"] for row in public if row["path"] not in by_path]
        if missing:
            raise ValueError(f"Baseline contains paths absent from metadata: {missing[:5]}")
        baseline[name] = [by_path[row["path"]] for row in public]
    return {
        "available": True,
        "path": str(baseline_dir),
        "v1": {
            name: {"rows": len(baseline[name]), "identities": len(identity_keys(baseline[name]))}
            for name in OUTPUT_SPLITS
        },
        "v2": {
            name: {"rows": len(splits[name]), "identities": len(identity_keys(splits[name]))}
            for name in OUTPUT_SPLITS
        },
        "v1_integrity_observations": _crossing_counts(
            baseline["train"], baseline["val_known"]
        ),
        "v2_integrity_observations": _crossing_counts(
            splits["train"], splits["val_known"]
        ),
        "_baseline_splits": baseline,
    }


def main(argv=None):
    args = parse_args(argv)
    if args.validation_only:
        result = validate_written_v2(
            args.output_dir,
            args.metadata,
            data_root=args.data_root,
            verify_images=args.verify_images,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite_v2:
        raise FileExistsError(
            f"Refusing occupied Split V2 destination: {args.output_dir}. "
            "Use --overwrite-v2 explicitly."
        )

    config = _apply_overrides(load_split_config(args.config), args)
    metadata_sha256 = sha256_file(args.metadata)
    rows = read_rows(args.metadata)
    date_stats = annotate_dates(rows)
    annotate_explicit_groups(rows, config.enabled_parsers)
    duplicate_stats = annotate_image_hashes(rows, args.data_root, args.verify_images)

    by_key = group_rows_by_identity(rows)
    _, unknown_keys = choose_unknown_keys(
        by_key, config.seed, config.unknown_identity_ratio
    )
    methods, candidate_rows = plan_identity_methods(by_key, unknown_keys)
    features_by_index = {}
    if candidate_rows:
        features = load_or_extract_feature_cache(
            cache_dir=args.feature_cache,
            candidate_rows=candidate_rows,
            metadata_sha256=metadata_sha256,
            feature=config.feature,
            data_root=args.data_root,
            checkpoint=args.feature_checkpoint,
            device=args.device,
            batch_size=args.batch_size,
            recompute=args.recompute_features,
        )
        features_by_index = build_features_by_index(candidate_rows, features)

    splits = make_observation_aware_splits(
        rows, config, methods, unknown_keys, features_by_index
    )
    integrity = validate_splits(rows, splits)
    integrity["date_statistics"] = dict(date_stats)
    integrity["duplicate_statistics"] = duplicate_stats
    summary = build_summary(
        rows,
        splits,
        config,
        args.metadata,
        args.feature_cache if candidate_rows else None,
        integrity,
        REPO_ROOT,
    )
    comparison = _baseline_comparison(args.baseline_dir, rows, splits)
    baseline_splits = comparison.pop("_baseline_splits", None)
    summary["v1_v2_comparison"] = comparison
    extras = {}
    if args.near_duplicate_audit:
        dhash_threshold = (
            args.dhash_threshold
            if args.dhash_threshold is not None
            else int(config.near_duplicate_audit["dhash_threshold"])
        )
        phash_threshold = (
            args.phash_threshold
            if args.phash_threshold is not None
            else int(config.near_duplicate_audit["phash_threshold"])
        )
        perceptual_hashes = {}
        candidates = audit_cross_split_near_duplicates(
            splits,
            args.data_root,
            dhash_threshold=dhash_threshold,
            phash_threshold=phash_threshold,
            perceptual_hashes=perceptual_hashes,
        )
        extras["near_duplicate_candidates.csv"] = (
            candidates,
            NEAR_DUPLICATE_FIELDS,
        )
        baseline_candidates = []
        if baseline_splits:
            baseline_candidates = audit_cross_split_near_duplicates(
                baseline_splits,
                args.data_root,
                dhash_threshold=dhash_threshold,
                phash_threshold=phash_threshold,
                perceptual_hashes=perceptual_hashes,
            )
            extras["v1_near_duplicate_candidates.csv"] = (
                baseline_candidates,
                NEAR_DUPLICATE_FIELDS,
            )
        summary["near_duplicate_audit"] = {
            "enabled": True,
            "dhash_threshold": dhash_threshold,
            "phash_threshold": phash_threshold,
            "candidate_pairs": len(candidates),
            "v1_candidate_pairs": len(baseline_candidates) if baseline_splits else None,
        }
    else:
        summary["near_duplicate_audit"] = {"enabled": False}

    hashes = write_v2_artifacts(
        args.output_dir,
        splits,
        manifest_rows(rows, config),
        summary,
        overwrite=args.overwrite_v2,
        extra_csvs=extras,
    )
    _print_summary(summary)
    print("output_hashes=" + json.dumps(hashes, sort_keys=True))


if __name__ == "__main__":
    main()
