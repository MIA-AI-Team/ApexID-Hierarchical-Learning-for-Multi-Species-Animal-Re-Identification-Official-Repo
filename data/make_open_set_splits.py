import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


FIELDNAMES = ["identity", "path", "date", "orientation", "species", "dataset"]
DEFAULT_SOURCE = Path("metadata_with_jaguar_cleaned_readable.csv")
DEFAULT_OUTPUT_DIR = Path("open_set_splits")
NEW_WHALE_KEY = ("humpback-whale-identification", "new_whale")


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


def choose_unknown_keys(by_key, seed, unknown_identity_ratio):
    rng = random.Random(seed)
    normal_keys = [key for key in by_key if key != NEW_WHALE_KEY]
    keys_by_dataset = defaultdict(list)
    for key in normal_keys:
        keys_by_dataset[key[0]].append(key)

    unknown_keys = set()
    for _dataset, keys in keys_by_dataset.items():
        eligible = [key for key in keys if len(by_key[key]) >= 2]
        if not eligible:
            continue

        rng.shuffle(eligible)
        if len(eligible) >= 10:
            target = max(1, round(len(eligible) * unknown_identity_ratio))
        else:
            target = max(0, round(len(eligible) * unknown_identity_ratio))
        unknown_keys.update(eligible[: min(target, len(eligible))])

    return normal_keys, unknown_keys, rng


def make_splits(
    rows,
    seed=42,
    unknown_identity_ratio=0.10,
    known_image_ratio=0.20,
    min_known_images=2,
    max_unknown_images_per_identity=50,
    max_new_whale_unknown=1000,
):
    by_key = defaultdict(list)
    for row in rows:
        by_key[row["_key"]].append(row)

    normal_keys, unknown_keys, rng = choose_unknown_keys(
        by_key=by_key,
        seed=seed,
        unknown_identity_ratio=unknown_identity_ratio,
    )

    train = []
    val_known = []
    val_unknown = []
    unused = []

    for key in sorted(normal_keys):
        items = by_key[key][:]
        rng.shuffle(items)

        if key in unknown_keys:
            keep = min(len(items), max_unknown_images_per_identity)
            val_unknown.extend(items[:keep])
            unused.extend(items[keep:])
            continue

        if len(items) >= min_known_images:
            n_val = max(1, round(len(items) * known_image_ratio))
            n_val = min(n_val, len(items) - 1)
            val_known.extend(items[:n_val])
            train.extend(items[n_val:])
        else:
            train.extend(items)

    new_whale_rows = by_key.get(NEW_WHALE_KEY, [])[:]
    rng.shuffle(new_whale_rows)
    keep_new_whale = min(len(new_whale_rows), max_new_whale_unknown)
    val_unknown.extend(new_whale_rows[:keep_new_whale])
    unused.extend(new_whale_rows[keep_new_whale:])

    return {
        "train": train,
        "val_known": val_known,
        "val_unknown": val_unknown,
        "val": val_known + val_unknown,
        "val_unknown_extra": unused,
        "unused": unused,
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(public_row(row))


def read_split(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDNAMES:
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames!r}")
        return list(reader)


def identity_keys(rows):
    return {(row["dataset"], row["identity"]) for row in rows}


def assert_readable_images(rows):
    bad = []
    for row in rows:
        path = Path(row["path"])
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:
            bad.append((row["path"], type(exc).__name__))
            if len(bad) >= 5:
                break
    if bad:
        raise ValueError(f"Unreadable image paths remain: {bad}")


def validate_outputs(output_dir):
    train = read_split(output_dir / "train.csv")
    val_known = read_split(output_dir / "val_known.csv")
    val_unknown = read_split(output_dir / "val_unknown.csv")
    val_unknown_extra = read_split(output_dir / "val_unknown_extra.csv")
    val = read_split(output_dir / "val.csv")

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
    if len(val) != len(val_known) + len(val_unknown):
        raise ValueError("val.csv row count is not val_known + val_unknown")

    new_whale_counts = {
        "train": sum(row["identity"] == "new_whale" for row in train),
        "val_known": sum(row["identity"] == "new_whale" for row in val_known),
        "val_unknown": sum(row["identity"] == "new_whale" for row in val_unknown),
    }
    if new_whale_counts != {"train": 0, "val_known": 0, "val_unknown": 1000}:
        raise ValueError(f"Unexpected new_whale counts: {new_whale_counts}")

    assert_readable_images(train)
    assert_readable_images(val_known)
    assert_readable_images(val_unknown)
    assert_readable_images(val_unknown_extra)

    return {
        "train_rows": len(train),
        "val_known_rows": len(val_known),
        "val_unknown_rows": len(val_unknown),
        "val_unknown_extra_rows": len(val_unknown_extra),
        "val_rows": len(val),
        "train_identities": len(train_ids),
        "val_known_identities": len(val_known_ids),
        "val_unknown_identities": len(val_unknown_ids),
        "val_unknown_extra_identities": len(val_unknown_extra_ids),
        "new_whale_counts": new_whale_counts,
        "new_whale_extra_rows": sum(
            row["identity"] == "new_whale" for row in val_unknown_extra
        ),
    }


def print_summary(splits, validation):
    for name in ["train", "val_known", "val_unknown", "val", "val_unknown_extra"]:
        rows = splits[name]
        dataset_counts = Counter(row["dataset"] for row in rows)
        print(f"{name}: rows={len(rows)} identities={len(identity_keys(rows))}")
        print(f"{name}: top_datasets={dataset_counts.most_common(8)}")

    print("validation:")
    for key, value in validation.items():
        print(f"{key}={value}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create open-set animal re-identification train/validation splits."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unknown-identity-ratio", type=float, default=0.10)
    parser.add_argument("--known-image-ratio", type=float, default=0.20)
    parser.add_argument("--max-unknown-images-per-identity", type=int, default=50)
    parser.add_argument("--max-new-whale-unknown", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_rows(args.source)
    splits = make_splits(
        rows=rows,
        seed=args.seed,
        unknown_identity_ratio=args.unknown_identity_ratio,
        known_image_ratio=args.known_image_ratio,
        max_unknown_images_per_identity=args.max_unknown_images_per_identity,
        max_new_whale_unknown=args.max_new_whale_unknown,
    )

    for name in ["train", "val_known", "val_unknown", "val", "val_unknown_extra"]:
        write_csv(args.output_dir / f"{name}.csv", splits[name])

    validation = validate_outputs(args.output_dir)
    print_summary(splits, validation)


if __name__ == "__main__":
    main()
