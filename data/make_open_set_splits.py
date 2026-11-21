#!/usr/bin/env python3
"""Build or validate the ApexID Observation-Aware Split V2."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apexid.data.split_v2 import (  # noqa: E402
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
    load_or_extract_feature_cache,
    make_observation_aware_splits,
    manifest_rows,
    plan_identity_methods,
    read_rows,
    sha256_file,
    validate_splits,
    validate_written_v2,
    write_v2_artifacts,
)


DEFAULT_METADATA = Path("metadata.csv")
DEFAULT_DATA_ROOT = Path("ApexID_Dataset")
DEFAULT_OUTPUT_DIR = Path("ApexID_Dataset/open_set_splits_v2")
DEFAULT_FEATURE_CACHE = Path("ApexID_Dataset/split_v2_feature_cache")
DEFAULT_CONFIG = REPO_ROOT / "data" / "split_v2.yaml"


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
        help="Curated metadata CSV (default for generation: metadata.csv).",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-cache", type=Path, default=DEFAULT_FEATURE_CACHE)
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
        help="Atomically replace the named output directory.",
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

    metadata = args.metadata or DEFAULT_METADATA
    config = _apply_overrides(load_split_config(args.config), args)
    metadata_sha256 = sha256_file(metadata)
    rows = read_rows(metadata)
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
        metadata,
        args.feature_cache if candidate_rows else None,
        integrity,
        REPO_ROOT,
    )
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
        candidates = audit_cross_split_near_duplicates(
            splits,
            args.data_root,
            dhash_threshold=dhash_threshold,
            phash_threshold=phash_threshold,
        )
        extras["near_duplicate_candidates.csv"] = (
            candidates,
            NEAR_DUPLICATE_FIELDS,
        )
        summary["near_duplicate_audit"] = {
            "enabled": True,
            "dhash_threshold": dhash_threshold,
            "phash_threshold": phash_threshold,
            "candidate_pairs": len(candidates),
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
