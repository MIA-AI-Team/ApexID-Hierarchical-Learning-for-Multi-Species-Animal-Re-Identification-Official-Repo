#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml

from apexid.data import calculate_dataset_mean_std
from apexid.utils.config import DEFAULT_CONFIG_PATH, load_config


def _as_dict(value):
    """Return ``value`` as a normal dictionary."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "items"):
        return dict(value.items())
    raise TypeError(f"Expected a dict-like value, got {type(value).__name__}.")


def _resolve_csv_path(data_cfg, split, csv_path=None):
    """Resolve the CSV path from an override or config split."""
    if csv_path:
        return csv_path

    if split == "train":
        path = data_cfg.get("train_path") or data_cfg.get("train_csv")
        key = "data.train_path"
    elif split == "val":
        path = data_cfg.get("val_path") or data_cfg.get("val_csv")
        key = "data.val_path"
    else:
        raise ValueError("split must be 'train' or 'val'.")

    if path in {None, ""}:
        raise ValueError(f"Missing {key}. Pass --csv or set {key} in the config.")
    return path


def _round_values(values, precision):
    """Round float values for display and config writing."""
    return [round(float(value), int(precision)) for value in values]


def _write_stats_to_config(config_path, mean, std):
    """Write calculated statistics back into ``data.transforms``."""
    config_path = Path(config_path)
    cfg = load_config(config_path=config_path)
    cfg.setdefault("data", {}).setdefault("transforms", {})
    cfg["data"]["transforms"]["mean"] = mean
    cfg["data"]["transforms"]["std"] = std

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(repo_root):
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip()


def write_normalization_provenance(
    output_path,
    config_path,
    csv_path,
    manifest_path,
    mean,
    std,
    command=None,
):
    """Atomically write train-only normalization provenance for Split V2."""
    output_path = Path(output_path)
    config_path = Path(config_path)
    csv_path = Path(csv_path)
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split V2 manifest not found: {manifest_path}")
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        row_count = sum(1 for _ in csv.DictReader(handle))
    cfg = load_config(config_path=config_path)
    data_cfg = _as_dict(cfg.get("data", {}))
    transforms = _as_dict(data_cfg.get("transforms", {}))
    payload = {
        "schema_version": 1,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "train_csv_path": str(csv_path),
        "train_csv_sha256": _sha256_file(csv_path),
        "row_count": row_count,
        "preprocessing": {
            "color_mode": "RGB",
            "image_size": data_cfg.get("image_size"),
            "wavelet": bool(data_cfg.get("wavelet", False)),
            "transforms": transforms,
        },
        "command": command or " ".join(shlex.quote(value) for value in sys.argv),
        "code_revision": _git_revision(REPO_ROOT),
        "mean": [float(value) for value in mean],
        "standard_deviation": [float(value) for value in std],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, output_path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise
    return payload


def calculate_from_config(
    config_path=DEFAULT_CONFIG_PATH,
    split="train",
    csv_path=None,
    root_dir=None,
    path_column="path",
    max_images=None,
    precision=6,
    write_config=False,
):
    """Calculate dataset mean/std using an ApexID config.

    Args:
        config_path (str or pathlib.Path): YAML config path.
        split (str): ``"train"`` or ``"val"``. Used only when ``csv_path`` is
            not provided.
        csv_path (str or pathlib.Path, optional): Explicit CSV path override.
        root_dir (str or pathlib.Path, optional): Image root override. If omitted,
            ``data.root_dir`` from the config is used.
        path_column (str): CSV column containing image paths after standardizing
            common aliases.
        max_images (int, optional): Limit the calculation to the first N rows.
        precision (int): Number of decimal places in returned values.
        write_config (bool): Whether to write values back into
            ``data.transforms.mean/std`` in ``config_path``.

    Returns:
        tuple[list[float], list[float]]: Rounded mean and std lists. Length is 3
        for RGB mode and 6 when ``data.wavelet`` is true.
    """
    cfg = load_config(config_path=config_path)
    data_cfg = _as_dict(cfg.get("data", {}))
    csv_path = _resolve_csv_path(data_cfg, split=split, csv_path=csv_path)
    root_dir = data_cfg.get("root_dir") if root_dir is None else root_dir

    mean, std = calculate_dataset_mean_std(
        data_csv=csv_path,
        root_dir=root_dir,
        config=cfg,
        path_column=path_column,
        max_images=max_images,
    )
    mean = _round_values(mean, precision)
    std = _round_values(std, precision)

    if write_config:
        _write_stats_to_config(config_path, mean, std)

    return mean, std


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate ApexID dataset mean/std after configured preprocessing."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML config. Defaults to configs/default.yaml.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="train",
        help="Config split to use when --csv is not provided.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional CSV path override. Expected columns include identity and path.",
    )
    parser.add_argument(
        "--root-dir",
        default=None,
        help="Optional image root override. Defaults to data.root_dir from config.",
    )
    parser.add_argument(
        "--path-column",
        default="path",
        help="Standardized CSV path column name.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional image limit for quick estimates.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Number of decimal places to print and write.",
    )
    parser.add_argument(
        "--write-config",
        action="store_true",
        help="Write calculated mean/std into data.transforms in the config file.",
    )
    parser.add_argument(
        "--additional-config",
        action="append",
        default=[],
        help="Additional config to receive the same mean/std when --write-config is set.",
    )
    parser.add_argument(
        "--provenance-output",
        default=None,
        help="Write Split V2 normalization_stats.json with hashes and preprocessing.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Split V2 split_manifest.csv required with --provenance-output.",
    )
    return parser.parse_args()


def main():
    """CLI entrypoint."""
    args = parse_args()
    mean, std = calculate_from_config(
        config_path=args.config,
        split=args.split,
        csv_path=args.csv,
        root_dir=args.root_dir,
        path_column=args.path_column,
        max_images=args.max_images,
        precision=args.precision,
        write_config=args.write_config,
    )

    print("data:")
    print("  transforms:")
    print(f"    mean: {mean}")
    print(f"    std: {std}")
    if args.write_config:
        for config_path in args.additional_config:
            _write_stats_to_config(config_path, mean, std)
        print(f"\nUpdated {args.config}")
        for config_path in args.additional_config:
            print(f"Updated {config_path}")
    if args.provenance_output:
        if not args.manifest:
            raise ValueError("--manifest is required with --provenance-output")
        cfg = load_config(config_path=args.config)
        data_cfg = _as_dict(cfg.get("data", {}))
        csv_path = _resolve_csv_path(data_cfg, args.split, args.csv)
        payload = write_normalization_provenance(
            args.provenance_output,
            args.config,
            csv_path,
            args.manifest,
            mean,
            std,
        )
        print(
            f"Wrote {args.provenance_output} "
            f"(rows={payload['row_count']}, manifest={payload['manifest_sha256']})"
        )


if __name__ == "__main__":
    main()
