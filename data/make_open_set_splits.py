import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


FIELDNAMES = ["identity", "path", "date", "orientation", "species", "dataset"]
DEFAULT_SOURCE = Path("ApexID_Dataset/metadata.csv")
DEFAULT_OUTPUT_DIR = Path("ApexID_Dataset/open_set_splits")
NEW_WHALE_KEY = ("humpback-whale-identification", "new_whale")
COMPONENT_SPLITS = [
    "train",
    "val_known",
    "val_unknown",
    "val_unknown_extra",
    "unlabeled_ood",
]
OUTPUT_SPLITS = ["train", "val_known", "val_unknown", "val", "val_unknown_extra", "unlabeled_ood"]


def read_rows(source):
    with source.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDNAMES:
            raise ValueError(f"Unexpected columns in {source}: {reader.fieldnames!r}")
        rows = []
        for idx, row in enumerate(reader):
            row["_idx"] = idx
            row["_key"] = (row["dataset"], row["identity"])
            rows.append(row)
    return rows


def public_row(row):
    return {field: row.get(field, "") for field in FIELDNAMES}


def identity_keys(rows):
    return {(row["dataset"], row["identity"]) for row in rows}


def count_new_whale(rows):
    return sum((row["dataset"], row["identity"]) == NEW_WHALE_KEY for row in rows)


def group_rows_by_identity(rows):
    by_key = defaultdict(list)
    for row in rows:
        by_key[row["_key"]].append(row)
    return by_key


def choose_unknown_keys(by_key, seed, unknown_identity_ratio):
    rng = random.Random(seed)
    eligible_keys = [
        key
        for key, rows in by_key.items()
        if key != NEW_WHALE_KEY and len(rows) >= 2
    ]
    keys_by_dataset = defaultdict(list)
    for key in eligible_keys:
        keys_by_dataset[key[0]].append(key)

    unknown_keys = set()
    for _dataset, keys in sorted(keys_by_dataset.items()):
        eligible = sorted(keys)
        if not eligible:
            continue

        rng.shuffle(eligible)
        if len(eligible) >= 10:
            target = max(1, round(len(eligible) * unknown_identity_ratio))
        else:
            target = max(0, round(len(eligible) * unknown_identity_ratio))
        unknown_keys.update(eligible[: min(target, len(eligible))])

    return sorted(eligible_keys), unknown_keys, rng


def make_splits(
    rows,
    seed=42,
    unknown_identity_ratio=0.125,
    known_image_ratio=0.225,
    min_train_images_per_known_identity=2,
    min_val_images_per_known_identity=2,
    max_unknown_images_per_identity=100,
):
    by_key = group_rows_by_identity(rows)
    eligible_keys, unknown_keys, rng = choose_unknown_keys(
        by_key=by_key,
        seed=seed,
        unknown_identity_ratio=unknown_identity_ratio,
    )

    train = []
    val_known = []
    val_unknown = []
    val_unknown_extra = []
    unlabeled_ood = []

    for key in sorted(by_key):
        if key == NEW_WHALE_KEY:
            unlabeled_ood.extend(by_key[key])
            continue

        items = by_key[key][:]
        if len(items) == 1:
            unlabeled_ood.extend(items)
            continue

        rng.shuffle(items)
        if key in unknown_keys:
            keep = min(len(items), max_unknown_images_per_identity)
            val_unknown.extend(items[:keep])
            val_unknown_extra.extend(items[keep:])
            continue

        if key not in eligible_keys:
            train.extend(items)
            continue

        min_total = min_train_images_per_known_identity + min_val_images_per_known_identity
        if len(items) < min_total:
            train.extend(items)
            continue

        n_val = max(
            min_val_images_per_known_identity,
            round(len(items) * known_image_ratio),
        )
        n_val = min(n_val, len(items) - min_train_images_per_known_identity)

        if n_val < min_val_images_per_known_identity:
            train.extend(items)
            continue

        val_known.extend(items[:n_val])
        train.extend(items[n_val:])

    return {
        "train": train,
        "val_known": val_known,
        "val_unknown": val_unknown,
        "val": val_known + val_unknown,
        "val_unknown_extra": val_unknown_extra,
        "unlabeled_ood": unlabeled_ood,
    }


def clear_output_csvs(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob("*.csv"):
        path.unlink()


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(public_row(row))


def write_splits(output_dir, splits, clear_existing=True):
    output_dir = Path(output_dir)
    if clear_existing:
        clear_output_csvs(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    for name in OUTPUT_SPLITS:
        write_csv(output_dir / f"{name}.csv", splits[name])


def read_split(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDNAMES:
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames!r}")
        return list(reader)


def resolve_image_path(path, root_dir):
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    if root_dir is not None:
        candidate = Path(root_dir) / path
        if candidate.exists():
            return candidate
    return path


def assert_readable_images(rows, root_dir=None):
    bad = []
    for row in rows:
        path = resolve_image_path(row["path"], root_dir=root_dir)
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:
            bad.append((str(path), type(exc).__name__))
            if len(bad) >= 5:
                break
    if bad:
        raise ValueError(f"Unreadable image paths remain: {bad}")


def assert_row_accounting(source_rows, splits):
    source_paths = Counter(row["path"] for row in source_rows)
    component_paths = Counter(
        row["path"]
        for name in COMPONENT_SPLITS
        for row in splits[name]
    )
    if source_paths != component_paths:
        missing = list((source_paths - component_paths).elements())[:5]
        extra = list((component_paths - source_paths).elements())[:5]
        raise ValueError(
            "Component splits do not account for source rows exactly once. "
            f"Missing examples: {missing}; extra examples: {extra}"
        )


def validate_outputs(output_dir, source_rows=None, root_dir=None, check_images=True):
    splits = {
        name: read_split(output_dir / f"{name}.csv")
        for name in OUTPUT_SPLITS
    }

    train = splits["train"]
    val_known = splits["val_known"]
    val_unknown = splits["val_unknown"]
    val_unknown_extra = splits["val_unknown_extra"]
    unlabeled_ood = splits["unlabeled_ood"]
    val = splits["val"]

    train_ids = identity_keys(train)
    val_known_ids = identity_keys(val_known)
    val_unknown_ids = identity_keys(val_unknown)
    val_unknown_extra_ids = identity_keys(val_unknown_extra)

    if train_ids & val_unknown_ids:
        raise ValueError("train and val_unknown share identities")
    if train_ids & val_unknown_extra_ids:
        raise ValueError("train and val_unknown_extra share identities")
    if val_known_ids - train_ids:
        raise ValueError("val_known contains identities not present in train")
    if val_known_ids & val_unknown_ids:
        raise ValueError("val_known and val_unknown share identities")
    if val != val_known + val_unknown:
        raise ValueError("val.csv is not exactly val_known.csv + val_unknown.csv")

    component_paths = [
        row["path"]
        for name in COMPONENT_SPLITS
        for row in splits[name]
    ]
    duplicate_paths = [path for path, count in Counter(component_paths).items() if count > 1]
    if duplicate_paths:
        raise ValueError(f"Component split paths are not disjoint: {duplicate_paths[:5]}")

    new_whale_counts = {
        "train": count_new_whale(train),
        "val_known": count_new_whale(val_known),
        "val_unknown": count_new_whale(val_unknown),
        "val_unknown_extra": count_new_whale(val_unknown_extra),
        "unlabeled_ood": count_new_whale(unlabeled_ood),
    }
    expected_new_whale = count_new_whale(source_rows or unlabeled_ood)
    if new_whale_counts != {
        "train": 0,
        "val_known": 0,
        "val_unknown": 0,
        "val_unknown_extra": 0,
        "unlabeled_ood": expected_new_whale,
    }:
        raise ValueError(f"Unexpected new_whale counts: {new_whale_counts}")

    if source_rows is not None:
        source_by_key = group_rows_by_identity(source_rows)
        singleton_paths = {
            rows[0]["path"]
            for key, rows in source_by_key.items()
            if key != NEW_WHALE_KEY and len(rows) == 1
        }
        non_ood_paths = set(component_paths) - {row["path"] for row in unlabeled_ood}
        misplaced_singletons = singleton_paths & non_ood_paths
        if misplaced_singletons:
            raise ValueError(
                "Singleton identities appeared outside unlabeled_ood: "
                f"{sorted(misplaced_singletons)[:5]}"
            )
        assert_row_accounting(source_rows, splits)

    if check_images:
        for name in COMPONENT_SPLITS:
            assert_readable_images(splits[name], root_dir=root_dir)

    return {
        "train_rows": len(train),
        "val_known_rows": len(val_known),
        "val_unknown_rows": len(val_unknown),
        "val_unknown_extra_rows": len(val_unknown_extra),
        "unlabeled_ood_rows": len(unlabeled_ood),
        "val_rows": len(val),
        "train_identities": len(train_ids),
        "val_known_identities": len(val_known_ids),
        "val_unknown_identities": len(val_unknown_ids),
        "val_unknown_extra_identities": len(val_unknown_extra_ids),
        "unlabeled_ood_identities": len(identity_keys(unlabeled_ood)),
        "new_whale_counts": new_whale_counts,
    }


def print_summary(splits, validation):
    main_total = len(splits["train"]) + len(splits["val"])
    train_ratio = len(splits["train"]) / main_total if main_total else 0.0
    val_ratio = len(splits["val"]) / main_total if main_total else 0.0

    for name in OUTPUT_SPLITS:
        rows = splits[name]
        dataset_counts = Counter(row["dataset"] for row in rows)
        print(f"{name}: rows={len(rows)} identities={len(identity_keys(rows))}")
        print(f"{name}: top_datasets={dataset_counts.most_common(8)}")

    print(f"main_split_ratio=train:{train_ratio:.4f} val:{val_ratio:.4f}")
    print("validation:")
    for key, value in validation.items():
        print(f"{key}={value}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create clean open-set animal re-identification train/validation splits."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unknown-identity-ratio", type=float, default=0.125)
    parser.add_argument("--known-image-ratio", type=float, default=0.225)
    parser.add_argument("--min-train-images-per-known-identity", type=int, default=2)
    parser.add_argument("--min-val-images-per-known-identity", type=int, default=2)
    parser.add_argument("--max-unknown-images-per-identity", type=int, default=100)
    parser.add_argument(
        "--keep-existing-csvs",
        action="store_true",
        help="Do not delete existing CSV files in the output directory before writing.",
    )
    parser.add_argument(
        "--skip-image-check",
        action="store_true",
        help="Skip verifying image readability after writing split files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_rows(args.source)
    splits = make_splits(
        rows=rows,
        seed=args.seed,
        unknown_identity_ratio=args.unknown_identity_ratio,
        known_image_ratio=args.known_image_ratio,
        min_train_images_per_known_identity=args.min_train_images_per_known_identity,
        min_val_images_per_known_identity=args.min_val_images_per_known_identity,
        max_unknown_images_per_identity=args.max_unknown_images_per_identity,
    )

    write_splits(
        output_dir=args.output_dir,
        splits=splits,
        clear_existing=not args.keep_existing_csvs,
    )

    validation = validate_outputs(
        args.output_dir,
        source_rows=rows,
        root_dir=args.source.parent,
        check_images=not args.skip_image_check,
    )
    print_summary(splits, validation)


if __name__ == "__main__":
    main()
