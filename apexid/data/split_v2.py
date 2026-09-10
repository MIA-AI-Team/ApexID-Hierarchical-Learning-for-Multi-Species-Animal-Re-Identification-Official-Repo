"""Observation-aware, reproducible open-set split construction for ApexID.

The public CSV interface deliberately remains the six columns used by the
training pipeline.  Internal provenance is retained in ``split_manifest.csv``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image


FIELDNAMES = ["identity", "path", "date", "orientation", "species", "dataset"]
COMPONENT_SPLITS = [
    "train",
    "val_known",
    "val_unknown",
    "val_unknown_extra",
    "unlabeled_ood",
]
OUTPUT_SPLITS = [
    "train",
    "val_known",
    "val_unknown",
    "val",
    "val_unknown_extra",
    "unlabeled_ood",
]
NEW_WHALE_KEY = ("humpback-whale-identification", "new_whale")

DEFAULT_THRESHOLDS = {
    "CTai": 0.95,
    "HyenaID2022": 0.98,
    "IPanda50": 0.97,
    "LeopardID2022": 0.96,
    "NDD20": 0.97,
    "NyalaData": 0.999,
    "SealID": 0.96,
    "WhaleSharkID": 0.98,
}

MANIFEST_FIELDS = FIELDNAMES + [
    "source_index",
    "canonical_identity",
    "parsed_date",
    "date_format",
    "image_sha256",
    "split",
    "split_method",
    "observation_group_id",
    "duplicate_group_id",
    "explicit_group_id",
    "similarity_cluster_id",
    "split_seed",
    "feature_model",
    "feature_revision",
]


@dataclass(frozen=True)
class FeatureSpec:
    model: str = "timm/vit_large_patch14_dinov2.lvd142m"
    revision: str = "4741e1cafbf45415e77074bb0cb42dba76c8684a"
    weight_sha256: str = (
        "0424a5d1b515278cba3c6640ccbeaacc41de59d3a93df0dd5e494285eea2b355"
    )
    dimension: int = 1024
    dtype: str = "float32"
    resize: tuple[int, int] = (518, 518)
    interpolation: str = "bilinear"
    color_mode: str = "RGB"
    scale: str = "uint8/[255]"
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    normalization: str = "L2"


@dataclass
class SplitConfig:
    version: str = "2.0"
    seed: int = 42
    unknown_identity_ratio: float = 0.125
    known_image_ratio: float = 0.225
    max_unknown_images_per_identity: int = 100
    default_similarity_threshold: float = 0.98
    similarity_thresholds: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_THRESHOLDS)
    )
    enabled_parsers: tuple[str, ...] = (
        "CTai",
        "Giraffes",
        "IPanda50",
        "NDD20",
        "NyalaData",
    )
    threshold_provenance: str = "WildlifeDatasets v1.0.9 numeric values; fixed 0.98 default"
    metadata_provenance: dict[str, object] = field(
        default_factory=lambda: {
            "authoritative_path": "metadata.csv",
            "rows": 158293,
            "sha256": "fd8a921f7ab5566b02fa166bfbbfc948375ee68226b15c29caab8b7fc3ebacf0",
        }
    )
    cache_policy: dict[str, object] = field(
        default_factory=lambda: {
            "format": "numpy_npy_memory_map",
            "schema_version": 1,
            "reject_stale": True,
            "reject_incomplete": True,
            "ordered_rows_required": True,
            "artifact_hashes_required": True,
        }
    )
    near_duplicate_audit: dict[str, object] = field(
        default_factory=lambda: {
            "enabled_by_default": False,
            "dhash_threshold": 4,
            "phash_threshold": 6,
            "scope": "within_identity_train_to_val_known",
        }
    )
    scientific_provenance: dict[str, str] = field(
        default_factory=lambda: {
            "wildlife_datasets_version": "1.0.9",
            "wildlife_datasets_commit": "67380d7951b89f02e48c84a7d3546188b9dbec8a",
            "wildlife_datasets_license": "AGPL-3.0-or-later",
            "timm_version": "1.0.27",
            "torch_version": "2.10.0",
            "torchvision_version": "0.25.0",
            "numpy_version": "2.1.3",
            "pillow_version": "11.1.0",
            "huggingface_hub_version": "1.13.0",
            "safetensors_version": "0.7.0",
        }
    )
    feature: FeatureSpec = field(default_factory=FeatureSpec)


class UnionFind:
    def __init__(self, values: Iterable[int]):
        values = list(values)
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1

    def groups(self) -> list[list[int]]:
        grouped: dict[int, list[int]] = defaultdict(list)
        for value in self.parent:
            grouped[self.find(value)].append(value)
        return [sorted(values) for values in grouped.values()]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(prefix: str, *parts: object) -> str:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return f"{prefix}_{_sha256_bytes(payload.encode('utf-8'))[:20]}"


def canonical_identity(row: Mapping[str, object]) -> str:
    return f"{row['dataset']}::{row['identity']}"


def public_row(row: Mapping[str, object]) -> dict[str, object]:
    return {name: row.get(name, "") for name in FIELDNAMES}


def read_rows(source: Path) -> list[dict[str, object]]:
    source = Path(source)
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDNAMES:
            raise ValueError(f"Unexpected columns in {source}: {reader.fieldnames!r}")
        rows: list[dict[str, object]] = []
        for index, public in enumerate(reader):
            row: dict[str, object] = dict(public)
            row["_idx"] = index
            row["_key"] = (row["dataset"], row["identity"])
            rows.append(row)
    return rows


def resolve_image_path(path: str | Path, data_root: Path | None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if data_root is not None:
        return Path(data_root) / candidate
    return candidate


def parse_strict_date(value: object) -> tuple[str, str]:
    text = str(value or "").strip()
    if not text:
        return "", "missing"
    for date_format, label in (("%Y-%m-%d", "iso"), ("%d-%m-%Y", "czech")):
        try:
            parsed = datetime.strptime(text, date_format)
        except ValueError:
            continue
        if parsed.strftime(date_format) == text:
            return parsed.date().isoformat(), label
    return "", "malformed"


def annotate_dates(rows: Sequence[dict[str, object]]) -> Counter:
    counts: Counter = Counter()
    for row in rows:
        parsed, label = parse_strict_date(row.get("date", ""))
        row["_parsed_date"] = parsed
        row["_date_format"] = label
        counts[label] += 1
    return counts


def explicit_source_key(row: Mapping[str, object], enabled: Iterable[str]) -> str:
    import re

    dataset = str(row["dataset"])
    if dataset not in set(enabled):
        return ""
    basename = Path(str(row["path"])).name
    patterns = {
        "CTai": r"^img-id(?P<group>\d+)-object-\d+_\d+\.[^.]+$",
        "Giraffes": r"^(?P<group>.+?)_\d+_[0-9a-f]{16}\.[^.]+$",
        "NDD20": r"^(?P<group>\d+)_\d+\.[^.]+$",
        "NyalaData": (
            r"^(?P<group>\d+)_(?:left|right)photo(?:\(\d+\))?_"
            r"[0-9a-f]{16}\.[^.]+$"
        ),
    }
    if dataset == "IPanda50":
        for pattern in (
            r"^(?P<group>\d+_v\d+)_f\d+_[0-9a-f]{16}\.[^.]+$",
            r"^(?P<group>\d+_\d{8}-\d{6})(?:_[^.]*)?\.[^.]+$",
        ):
            match = re.match(pattern, basename, flags=re.IGNORECASE)
            if match:
                return f"{dataset}:{match.group('group')}"
        return ""
    pattern = patterns.get(dataset)
    if not pattern:
        return ""
    match = re.match(pattern, basename, flags=re.IGNORECASE)
    return f"{dataset}:{match.group('group')}" if match else ""


def annotate_explicit_groups(rows: Sequence[dict[str, object]], enabled: Iterable[str]) -> Counter:
    stats: Counter = Counter()
    for row in rows:
        source = explicit_source_key(row, enabled)
        row["_explicit_source"] = source
        # Explicit-source assignment is performed within each canonical
        # identity. Scope the manifest ID the same way so a source token that
        # happens to repeat between identities cannot create a false
        # cross-identity train/val_known violation.
        row["_explicit_group_id"] = (
            stable_id("src", row["dataset"], row["identity"], source) if source else ""
        )
        stats[(str(row["dataset"]), "matched" if source else "unmatched")] += 1
    return stats


def annotate_image_hashes(
    rows: Sequence[dict[str, object]],
    data_root: Path,
    verify_images: bool = False,
) -> dict[str, object]:
    by_hash: dict[str, list[dict[str, object]]] = defaultdict(list)
    unreadable: list[dict[str, object]] = []
    for row in rows:
        image_path = resolve_image_path(str(row["path"]), data_root)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found for source row {row['_idx']}: {image_path}")
        digest = sha256_file(image_path)
        row["_image_sha256"] = digest
        by_hash[digest].append(row)
        if verify_images:
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as exc:  # pragma: no cover - PIL backend-specific
                unreadable.append(
                    {"path": str(image_path), "error": type(exc).__name__}
                )
                if len(unreadable) >= 20:
                    break
    if unreadable:
        raise ValueError(f"Unreadable images: {unreadable}")

    conflicts = []
    for digest, hash_rows in by_hash.items():
        labels = sorted({canonical_identity(row) for row in hash_rows})
        if len(labels) > 1:
            conflicts.append(
                {
                    "sha256": digest,
                    "labels": labels,
                    "paths": sorted(str(row["path"]) for row in hash_rows),
                }
            )
    if conflicts:
        raise ValueError(
            "Identical image bytes have conflicting canonical labels:\n"
            + json.dumps(conflicts[:50], indent=2)
        )

    duplicate_groups = 0
    duplicate_rows = 0
    for digest, hash_rows in by_hash.items():
        group_id = stable_id("dup", digest) if len(hash_rows) > 1 else ""
        for row in hash_rows:
            row["_duplicate_group_id"] = group_id
        if len(hash_rows) > 1:
            duplicate_groups += 1
            duplicate_rows += len(hash_rows) - 1
    return {
        "unique_hashes": len(by_hash),
        "duplicate_groups": duplicate_groups,
        "duplicate_extra_rows": duplicate_rows,
        "conflicts": 0,
    }


def group_rows_by_identity(rows: Sequence[dict[str, object]]) -> dict[tuple[str, str], list[dict[str, object]]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[row["_key"]].append(row)  # type: ignore[index]
    return grouped


def identity_keys(rows: Sequence[Mapping[str, object]]) -> set[tuple[str, str]]:
    return {(str(row["dataset"]), str(row["identity"])) for row in rows}


def choose_unknown_keys(
    by_key: Mapping[tuple[str, str], Sequence[dict[str, object]]],
    seed: int,
    unknown_identity_ratio: float,
) -> tuple[list[tuple[str, str]], set[tuple[str, str]]]:
    """Preserve the established dataset-wise identity selection policy."""
    eligible = [key for key, items in by_key.items() if key != NEW_WHALE_KEY and len(items) >= 2]
    by_dataset: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in eligible:
        by_dataset[key[0]].append(key)
    rng = random.Random(seed)
    unknown: set[tuple[str, str]] = set()
    for dataset, keys in sorted(by_dataset.items()):
        del dataset
        candidates = sorted(keys)
        rng.shuffle(candidates)
        target = (
            max(1, round(len(candidates) * unknown_identity_ratio))
            if len(candidates) >= 10
            else max(0, round(len(candidates) * unknown_identity_ratio))
        )
        unknown.update(candidates[:target])
    return sorted(eligible), unknown


def _union_equal(uf: UnionFind, rows: Sequence[dict[str, object]], field: str) -> None:
    first: dict[str, int] = {}
    for row in rows:
        value = str(row.get(field, ""))
        if not value:
            continue
        index = int(row["_idx"])
        if value in first:
            uf.union(first[value], index)
        else:
            first[value] = index


def hard_observation_components(rows: Sequence[dict[str, object]]) -> list[list[int]]:
    uf = UnionFind(int(row["_idx"]) for row in rows)
    _union_equal(uf, rows, "_image_sha256")
    _union_equal(uf, rows, "_explicit_source")
    _union_equal(uf, rows, "_parsed_date")
    return sorted(uf.groups(), key=lambda values: values[0])


def _stable_group_order(seed: int, key: tuple[str, str], group_position: int) -> str:
    return stable_id("order", seed, key[0], key[1], group_position)


def closest_group_subset(
    groups: Sequence[Sequence[int]],
    target_images: int,
    seed: int,
    key: tuple[str, str],
    require_train: bool = True,
    cap: int | None = None,
) -> set[int]:
    """Return group indices with a deterministic image-count closest to target."""
    order = sorted(
        range(len(groups)),
        key=lambda index: _stable_group_order(seed, key, index),
    )
    states: dict[int, tuple[int, ...]] = {0: ()}
    for index in order:
        size = len(groups[index])
        for subtotal, chosen in list(states.items()):
            new_total = subtotal + size
            if cap is not None and new_total > cap:
                continue
            candidate = chosen + (index,)
            incumbent = states.get(new_total)
            if incumbent is None or candidate < incumbent:
                states[new_total] = candidate
    total = sum(len(group) for group in groups)
    candidates = []
    for subtotal, chosen in states.items():
        if subtotal == 0:
            continue
        if require_train and subtotal == total:
            continue
        candidates.append((abs(subtotal - target_images), subtotal, chosen))
    if not candidates:
        return set()
    return set(min(candidates)[2])


def plan_identity_methods(
    by_key: Mapping[tuple[str, str], Sequence[dict[str, object]]],
    unknown_keys: set[tuple[str, str]],
) -> tuple[dict[tuple[str, str], str], list[dict[str, object]]]:
    methods: dict[tuple[str, str], str] = {}
    candidates: list[dict[str, object]] = []
    for key in sorted(by_key):
        items = list(by_key[key])
        if key == NEW_WHALE_KEY:
            methods[key] = "new_whale_ood"
        elif len(items) == 1:
            methods[key] = "singleton_ood"
        elif key in unknown_keys:
            methods[key] = "unknown_identity"
        else:
            dates = [str(row.get("_parsed_date", "")) for row in items]
            explicit = [str(row.get("_explicit_source", "")) for row in items]
            if all(dates) and len(set(dates)) >= 2:
                methods[key] = "temporal"
            elif all(explicit) and len(set(explicit)) >= 2:
                methods[key] = "explicit_source"
            else:
                methods[key] = "similarity_fallback"
                if len(hard_observation_components(items)) >= 2:
                    candidates.extend(items)
    return methods, sorted(candidates, key=lambda row: int(row["_idx"]))


def feature_cache_fingerprint(
    candidate_rows: Sequence[dict[str, object]], metadata_sha256: str, feature: FeatureSpec
) -> str:
    payload = {
        "metadata_sha256": metadata_sha256,
        "feature": asdict(feature),
        "rows": [
            [int(row["_idx"]), str(row["path"]), str(row["_image_sha256"])]
            for row in candidate_rows
        ],
    }
    return _sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def validate_feature_cache(
    cache_dir: Path,
    candidate_rows: Sequence[dict[str, object]],
    metadata_sha256: str,
    feature: FeatureSpec,
) -> np.memmap:
    cache_dir = Path(cache_dir)
    metadata_path = cache_dir / "cache_metadata.json"
    features_path = cache_dir / "features.npy"
    rows_path = cache_dir / "feature_rows.csv"
    missing = [str(path) for path in (metadata_path, features_path, rows_path) if not path.is_file()]
    if missing:
        raise ValueError(f"Incomplete feature cache; missing: {missing}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_fingerprint = feature_cache_fingerprint(candidate_rows, metadata_sha256, feature)
    if metadata.get("fingerprint") != expected_fingerprint:
        raise ValueError("Stale feature cache: input/model/preprocessing fingerprint changed")
    if metadata.get("features_sha256") != sha256_file(features_path):
        raise ValueError("Corrupt feature cache: features.npy SHA-256 mismatch")
    if metadata.get("rows_sha256") != sha256_file(rows_path):
        raise ValueError("Corrupt feature cache: feature_rows.csv SHA-256 mismatch")
    features = np.load(features_path, mmap_mode="r")
    expected_shape = (len(candidate_rows), feature.dimension)
    if tuple(features.shape) != expected_shape or str(features.dtype) != feature.dtype:
        raise ValueError(
            f"Feature cache shape/dtype mismatch: got {features.shape}/{features.dtype}, "
            f"expected {expected_shape}/{feature.dtype}"
        )
    with rows_path.open(newline="", encoding="utf-8") as handle:
        ordered = list(csv.DictReader(handle))
    expected_rows = [
        {
            "source_index": str(row["_idx"]),
            "path": str(row["path"]),
            "image_sha256": str(row["_image_sha256"]),
        }
        for row in candidate_rows
    ]
    if ordered != expected_rows:
        raise ValueError("Stale feature cache: ordered row manifest changed")
    return features


def _resolve_checkpoint(feature: FeatureSpec, checkpoint: Path | None) -> Path:
    if checkpoint is None:
        from huggingface_hub import hf_hub_download

        checkpoint = Path(
            hf_hub_download(
                repo_id=feature.model,
                filename="model.safetensors",
                revision=feature.revision,
            )
        )
    checkpoint = Path(checkpoint)
    actual = sha256_file(checkpoint)
    if actual != feature.weight_sha256:
        raise ValueError(
            f"DINOv2 checkpoint SHA-256 mismatch: expected {feature.weight_sha256}, got {actual}"
        )
    return checkpoint


def extract_feature_cache(
    cache_dir: Path,
    candidate_rows: Sequence[dict[str, object]],
    metadata_sha256: str,
    feature: FeatureSpec,
    data_root: Path,
    checkpoint: Path | None,
    device: str,
    batch_size: int,
) -> np.memmap:
    import timm
    import torch
    import torch.nn.functional as torch_functional

    if not candidate_rows:
        raise ValueError("No similarity-fallback rows require a feature cache")
    checkpoint = _resolve_checkpoint(feature, checkpoint)
    resolved_device = device
    if device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    if resolved_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    model_name = feature.model.removeprefix("timm/")
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=0,
        checkpoint_path=str(checkpoint),
    )
    model.eval().to(resolved_device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    mean = np.asarray(feature.mean, dtype=np.float32)[:, None, None]
    std = np.asarray(feature.std, dtype=np.float32)[:, None, None]

    class ImageDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(candidate_rows)

        def __getitem__(self, index: int):
            row = candidate_rows[index]
            path = resolve_image_path(str(row["path"]), data_root)
            with Image.open(path) as image:
                image = image.convert(feature.color_mode).resize(
                    feature.resize,
                    resample=Image.Resampling.BILINEAR,
                )
                array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
            return torch.from_numpy((array - mean) / std)

    loader = torch.utils.data.DataLoader(
        ImageDataset(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(8, os.cpu_count() or 1),
        pin_memory=resolved_device.startswith("cuda"),
    )
    output = np.empty((len(candidate_rows), feature.dimension), dtype=np.float32)
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            embedding = model(batch.to(resolved_device, non_blocking=True))
            embedding = torch_functional.normalize(embedding.float(), p=2, dim=1)
            values = embedding.cpu().numpy().astype(np.float32, copy=False)
            output[offset : offset + len(values)] = values
            offset += len(values)
    if offset != len(output):
        raise RuntimeError(f"Feature extraction stopped early: {offset}/{len(output)}")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    features_path = cache_dir / "features.npy"
    fd, temporary = tempfile.mkstemp(prefix=".features.", suffix=".npy", dir=cache_dir)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, output, allow_pickle=False)
        os.replace(temporary, features_path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise
    row_records = [
        {
            "source_index": row["_idx"],
            "path": row["path"],
            "image_sha256": row["_image_sha256"],
        }
        for row in candidate_rows
    ]
    rows_path = cache_dir / "feature_rows.csv"
    _atomic_csv(rows_path, row_records, ["source_index", "path", "image_sha256"])
    _atomic_json(
        cache_dir / "cache_metadata.json",
        {
            "schema_version": 1,
            "fingerprint": feature_cache_fingerprint(candidate_rows, metadata_sha256, feature),
            "metadata_sha256": metadata_sha256,
            "feature": asdict(feature),
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "features_sha256": sha256_file(features_path),
            "rows_sha256": sha256_file(rows_path),
        },
    )
    return validate_feature_cache(cache_dir, candidate_rows, metadata_sha256, feature)


def load_or_extract_feature_cache(
    cache_dir: Path,
    candidate_rows: Sequence[dict[str, object]],
    metadata_sha256: str,
    feature: FeatureSpec,
    data_root: Path,
    checkpoint: Path | None = None,
    device: str = "auto",
    batch_size: int = 8,
    recompute: bool = False,
) -> np.memmap:
    cache_dir = Path(cache_dir)
    cache_files = [
        cache_dir / "cache_metadata.json",
        cache_dir / "features.npy",
        cache_dir / "feature_rows.csv",
    ]
    if not recompute and all(path.is_file() for path in cache_files):
        return validate_feature_cache(cache_dir, candidate_rows, metadata_sha256, feature)
    if not recompute and cache_dir.exists() and any(cache_dir.iterdir()):
        raise ValueError(
            f"Incomplete feature cache at {cache_dir}; use --recompute-features to replace it"
        )
    return extract_feature_cache(
        cache_dir,
        candidate_rows,
        metadata_sha256,
        feature,
        data_root,
        checkpoint,
        device,
        batch_size,
    )


def _assign(rows: Iterable[dict[str, object]], split: str, method: str) -> None:
    for row in rows:
        row["_split"] = split
        row["_split_method"] = method


def _unknown_components(rows: Sequence[dict[str, object]]) -> list[list[int]]:
    uf = UnionFind(int(row["_idx"]) for row in rows)
    _union_equal(uf, rows, "_image_sha256")
    _union_equal(uf, rows, "_explicit_source")
    return sorted(uf.groups(), key=lambda group: group[0])


def _temporal_assignment(
    key: tuple[str, str], items: Sequence[dict[str, object]], config: SplitConfig
) -> None:
    index_to_row = {int(row["_idx"]): row for row in items}
    groups = hard_observation_components(items)
    ordered_dates = sorted({str(row["_parsed_date"]) for row in items})
    target = round(len(items) * config.known_image_ratio)
    choices = []
    for split_index in range(1, len(ordered_dates)):
        train_dates = set(ordered_dates[:split_index])
        val_dates = set(ordered_dates[split_index:])
        group_sides = []
        for group in groups:
            dates = {str(index_to_row[index]["_parsed_date"]) for index in group}
            if dates <= train_dates:
                group_sides.append("train")
            elif dates <= val_dates:
                group_sides.append("val_known")
            else:
                break
        else:
            count = sum(
                len(group)
                for group, side in zip(groups, group_sides)
                if side == "val_known"
            )
            choices.append((abs(count - target), count, split_index, group_sides))
    if not choices:
        _assign(items, "train", "temporal_must_link_train_only")
        for group in groups:
            observation = stable_id(
                "obs_date",
                key[0],
                key[1],
                sorted(str(index_to_row[index]["path"]) for index in group),
            )
            for index in group:
                index_to_row[index]["_observation_group_id"] = observation
        return
    _, _, _, group_sides = min(choices)
    for group, split in zip(groups, group_sides):
        group_rows = [index_to_row[index] for index in group]
        _assign(group_rows, split, "temporal_chronological_suffix")
        observation = stable_id(
            "obs_date",
            key[0],
            key[1],
            sorted(str(index_to_row[index]["path"]) for index in group),
        )
        for row in group_rows:
            row["_observation_group_id"] = observation


def _explicit_assignment(
    key: tuple[str, str], items: Sequence[dict[str, object]], config: SplitConfig
) -> None:
    index_to_row = {int(row["_idx"]): row for row in items}
    groups = hard_observation_components(items)
    target = round(len(items) * config.known_image_ratio)
    selected = closest_group_subset(groups, target, config.seed, key, require_train=True)
    if not selected:
        _assign(items, "train", "explicit_source_train_only")
    for group_index, group in enumerate(groups):
        split = "val_known" if group_index in selected else "train"
        method = "explicit_source_group_subset" if selected else "explicit_source_train_only"
        group_rows = [index_to_row[index] for index in group]
        _assign(group_rows, split, method)
        observation = stable_id(
            "obs_src",
            key[0],
            key[1],
            sorted(str(index_to_row[index]["path"]) for index in group),
        )
        for row in group_rows:
            row["_observation_group_id"] = observation


def _similarity_assignment(
    key: tuple[str, str],
    items: Sequence[dict[str, object]],
    config: SplitConfig,
    features_by_index: Mapping[int, np.ndarray],
) -> None:
    indices = [int(row["_idx"]) for row in items]
    index_to_row = {int(row["_idx"]): row for row in items}
    uf = UnionFind(indices)
    _union_equal(uf, items, "_image_sha256")
    _union_equal(uf, items, "_explicit_source")
    _union_equal(uf, items, "_parsed_date")
    similarity_touched: set[int] = set()
    if all(index in features_by_index for index in indices):
        matrix = np.asarray([features_by_index[index] for index in indices], dtype=np.float32)
        threshold = config.similarity_thresholds.get(key[0], config.default_similarity_threshold)
        similarity = matrix @ matrix.T
        for left in range(len(indices)):
            for right in np.flatnonzero(similarity[left, left + 1 :] > threshold) + left + 1:
                uf.union(indices[left], indices[int(right)])
                similarity_touched.update((indices[left], indices[int(right)]))
    elif len(hard_observation_components(items)) >= 2:
        missing = [index for index in indices if index not in features_by_index]
        raise ValueError(f"Missing fallback features for {key}: source indices {missing[:10]}")

    groups = sorted(uf.groups(), key=lambda group: group[0])
    train_groups = {index for index, group in enumerate(groups) if len(group) > 1}
    singleton_groups = [group for group in groups if len(group) == 1]
    target = round(len(items) * config.known_image_ratio)
    selected_singletons: set[int] = set()
    if singleton_groups:
        require_train = not train_groups
        local_selected = closest_group_subset(
            singleton_groups,
            target,
            config.seed,
            key,
            require_train=require_train,
        )
        selected_singletons = {singleton_groups[index][0] for index in local_selected}

    for group_index, group in enumerate(groups):
        is_val = len(group) == 1 and group[0] in selected_singletons
        split = "val_known" if is_val else "train"
        if is_val:
            method = "similarity_unclustered_singleton"
        elif len(group) > 1:
            method = "similarity_or_must_link_cluster_train"
        else:
            method = "similarity_train_only_singleton"
        group_rows = [index_to_row[index] for index in group]
        _assign(group_rows, split, method)
        group_paths = sorted(str(index_to_row[index]["path"]) for index in group)
        observation = stable_id("obs_sim", key[0], key[1], group_paths)
        similarity_id = (
            stable_id("sim", key[0], key[1], group_paths)
            if len(group) > 1 and any(index in similarity_touched for index in group)
            else ""
        )
        for row in group_rows:
            row["_observation_group_id"] = observation
            row["_similarity_cluster_id"] = similarity_id


def make_observation_aware_splits(
    rows: Sequence[dict[str, object]],
    config: SplitConfig,
    methods: Mapping[tuple[str, str], str],
    unknown_keys: set[tuple[str, str]],
    features_by_index: Mapping[int, np.ndarray],
) -> dict[str, list[dict[str, object]]]:
    by_key = group_rows_by_identity(rows)
    for key in sorted(by_key):
        items = sorted(by_key[key], key=lambda row: int(row["_idx"]))
        method = methods[key]
        if method in {"new_whale_ood", "singleton_ood"}:
            _assign(items, "unlabeled_ood", method)
            for row in items:
                row["_observation_group_id"] = stable_id("obs_ood", key, row["path"])
        elif key in unknown_keys:
            index_to_row = {int(row["_idx"]): row for row in items}
            groups = _unknown_components(items)
            if len(items) <= config.max_unknown_images_per_identity:
                selected = set(range(len(groups)))
            else:
                selected = closest_group_subset(
                    groups,
                    config.max_unknown_images_per_identity,
                    config.seed,
                    key,
                    require_train=False,
                    cap=config.max_unknown_images_per_identity,
                )
                if not selected:
                    selected = {min(range(len(groups)), key=lambda index: (len(groups[index]), groups[index]))}
            for group_index, group in enumerate(groups):
                split = "val_unknown" if group_index in selected else "val_unknown_extra"
                group_rows = [index_to_row[index] for index in group]
                _assign(group_rows, split, "unknown_identity_group_cap")
                observation = stable_id(
                    "obs_unknown",
                    key[0],
                    key[1],
                    sorted(str(index_to_row[index]["path"]) for index in group),
                )
                for row in group_rows:
                    row["_observation_group_id"] = observation
        elif method == "temporal":
            _temporal_assignment(key, items, config)
        elif method == "explicit_source":
            _explicit_assignment(key, items, config)
        else:
            _similarity_assignment(key, items, config, features_by_index)

    splits = {
        name: sorted(
            [row for row in rows if row.get("_split") == name],
            key=lambda row: int(row["_idx"]),
        )
        for name in COMPONENT_SPLITS
    }
    splits["val"] = splits["val_known"] + splits["val_unknown"]
    return splits


def validate_splits(
    rows: Sequence[dict[str, object]],
    splits: Mapping[str, Sequence[dict[str, object]]],
) -> dict[str, object]:
    violations: list[str] = []
    train_ids = identity_keys(splits["train"])
    known_ids = identity_keys(splits["val_known"])
    unknown_ids = identity_keys(splits["val_unknown"])
    unknown_extra_ids = identity_keys(splits["val_unknown_extra"])
    if train_ids & unknown_ids:
        violations.append("train and val_unknown share canonical identities")
    if train_ids & unknown_extra_ids:
        violations.append("train and val_unknown_extra share canonical identities")
    if unknown_extra_ids - unknown_ids:
        violations.append("val_unknown_extra contains identities absent from val_unknown")
    if known_ids - train_ids:
        violations.append("val_known contains identities absent from train")
    if list(splits["val"]) != list(splits["val_known"]) + list(splits["val_unknown"]):
        violations.append("val is not exactly val_known + val_unknown")

    component_rows = [row for split in COMPONENT_SPLITS for row in splits[split]]
    source_indices = [int(row["_idx"]) for row in component_rows]
    expected_indices = [int(row["_idx"]) for row in rows]
    if Counter(source_indices) != Counter(expected_indices):
        violations.append("metadata rows are not accounted for exactly once")
    paths = [str(row["path"]) for row in component_rows]
    if len(paths) != len(set(paths)):
        violations.append("repeated paths appear across component outputs")

    split_by_index = {int(row["_idx"]): str(row["_split"]) for row in component_rows}
    for field, label in (
        ("_image_sha256", "exact-content"),
        ("_duplicate_group_id", "duplicate"),
        ("_explicit_group_id", "explicit-source"),
        ("_similarity_cluster_id", "similarity"),
        ("_observation_group_id", "observation"),
    ):
        grouped: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            value = str(row.get(field, ""))
            if value:
                grouped[value].add(split_by_index[int(row["_idx"])])
        for group_splits in grouped.values():
            if field in {"_image_sha256", "_duplicate_group_id"} and len(group_splits) > 1:
                violations.append(f"{label} group crosses component partitions")
                break
            if "train" in group_splits and "val_known" in group_splits:
                violations.append(f"{label} group crosses train/val_known")
                break

    date_groups: dict[tuple[tuple[str, str], str], set[str]] = defaultdict(set)
    for row in rows:
        date = str(row.get("_parsed_date", ""))
        if date:
            date_groups[(row["_key"], date)].add(split_by_index[int(row["_idx"])])  # type: ignore[index]
    if any("train" in values and "val_known" in values for values in date_groups.values()):
        violations.append("same identity-date group crosses train/val_known")

    by_key = group_rows_by_identity(rows)
    unlabeled_indices = {int(row["_idx"]) for row in splits["unlabeled_ood"]}
    for key, items in by_key.items():
        if key == NEW_WHALE_KEY or len(items) == 1:
            if any(int(row["_idx"]) not in unlabeled_indices for row in items):
                violations.append(f"singleton/new_whale routing failed for {key}")
                break
    if violations:
        raise ValueError("Split V2 integrity validation failed: " + "; ".join(violations))
    return {
        "passed": True,
        "checks": {
            "zero_repeated_paths": True,
            "zero_exact_content_crossings": True,
            "zero_train_unknown_identity_overlap": True,
            "val_known_identity_subset_train": True,
            "zero_observation_crossings": True,
            "complete_accounting": True,
            "val_concat_exact": True,
        },
    }


def _split_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "rows": len(rows),
        "identities": len(identity_keys(rows)),
        "datasets": dict(sorted(Counter(str(row["dataset"]) for row in rows).items())),
    }


def _git_state(repo_root: Path) -> dict[str, object]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=repo_root, text=True, capture_output=True, check=False
        )
        return result.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def build_summary(
    rows: Sequence[dict[str, object]],
    splits: Mapping[str, Sequence[dict[str, object]]],
    config: SplitConfig,
    metadata_path: Path,
    cache_dir: Path | None,
    integrity: Mapping[str, object],
    repo_root: Path,
) -> dict[str, object]:
    date_stats = Counter(str(row["_date_format"]) for row in rows)
    methods = Counter(str(row["_split_method"]) for row in rows)
    parser_statistics = {}
    for dataset in config.enabled_parsers:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        parser_statistics[dataset] = {
            "matched_rows": sum(bool(row.get("_explicit_source")) for row in dataset_rows),
            "unmatched_rows": sum(not bool(row.get("_explicit_source")) for row in dataset_rows),
            "explicit_groups": len(
                {str(row["_explicit_source"]) for row in dataset_rows if row.get("_explicit_source")}
            ),
        }
    cache_metadata = None
    if cache_dir and (Path(cache_dir) / "cache_metadata.json").is_file():
        cache_metadata = json.loads(
            (Path(cache_dir) / "cache_metadata.json").read_text(encoding="utf-8")
        )
    main_total = len(splits["train"]) + len(splits["val"])
    per_dataset: dict[str, object] = {}
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        source_rows = [row for row in rows if row["dataset"] == dataset]
        component_total = len(source_rows)
        split_counts = {}
        for name in OUTPUT_SPLITS:
            dataset_rows = [row for row in splits[name] if row["dataset"] == dataset]
            split_counts[name] = {
                "rows": len(dataset_rows),
                "identities": len(identity_keys(dataset_rows)),
                "ratio_of_metadata_rows": (
                    len(dataset_rows) / component_total if component_total else 0.0
                ),
            }
        per_dataset[dataset] = {
            "metadata_rows": component_total,
            "metadata_identities": len(identity_keys(source_rows)),
            "splits": split_counts,
        }
    return {
        "schema_version": 2,
        "configuration": {
            **asdict(config),
        },
        "git": _git_state(repo_root),
        "metadata": {
            "path": str(metadata_path),
            "sha256": sha256_file(metadata_path),
            "rows": len(rows),
        },
        "splits": {name: _split_counts(splits[name]) for name in OUTPUT_SPLITS},
        "per_dataset": per_dataset,
        "main_ratios": {
            "train": len(splits["train"]) / main_total if main_total else 0.0,
            "validation": len(splits["val"]) / main_total if main_total else 0.0,
        },
        "date_statistics": dict(sorted(date_stats.items())),
        "parser_statistics": parser_statistics,
        "method_statistics": dict(sorted(methods.items())),
        "unresolved_cases": {
            "malformed_date_rows": date_stats["malformed"],
            "train_only_similarity_rows": sum(
                count for method, count in methods.items() if "train_only" in method
            ),
        },
        "feature_cache": cache_metadata,
        "integrity": dict(integrity),
        "release_limitations": [
            "WildlifeDatasets is AGPL-3.0-or-later; its code is provenance only and is not imported by the splitter.",
            "The DINOv2 Hugging Face model card declares Apache-2.0 while its pretrained configuration declares CC-BY-NC-4.0; downstream release review is required.",
        ],
    }


def manifest_rows(rows: Sequence[dict[str, object]], config: SplitConfig) -> list[dict[str, object]]:
    output = []
    for row in sorted(rows, key=lambda item: int(item["_idx"])):
        record = public_row(row)
        record.update(
            {
                "source_index": row["_idx"],
                "canonical_identity": canonical_identity(row),
                "parsed_date": row.get("_parsed_date", ""),
                "date_format": row.get("_date_format", ""),
                "image_sha256": row.get("_image_sha256", ""),
                "split": row.get("_split", ""),
                "split_method": row.get("_split_method", ""),
                "observation_group_id": row.get("_observation_group_id", ""),
                "duplicate_group_id": row.get("_duplicate_group_id", ""),
                "explicit_group_id": row.get("_explicit_group_id", ""),
                "similarity_cluster_id": row.get("_similarity_cluster_id", ""),
                "split_seed": config.seed,
                "feature_model": config.feature.model,
                "feature_revision": config.feature.revision,
            }
        )
        output.append(record)
    return output


def write_v2_artifacts(
    output_dir: Path,
    splits: Mapping[str, Sequence[dict[str, object]]],
    manifest: Sequence[Mapping[str, object]],
    summary: dict[str, object],
    overwrite: bool = False,
    extra_csvs: Mapping[str, tuple[Sequence[Mapping[str, object]], Sequence[str]]] | None = None,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Refusing occupied Split V2 destination: {output_dir}. Use --overwrite-v2 explicitly."
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        for name in OUTPUT_SPLITS:
            _atomic_csv(staging / f"{name}.csv", [public_row(row) for row in splits[name]], FIELDNAMES)
        _atomic_csv(staging / "split_manifest.csv", manifest, MANIFEST_FIELDS)
        for filename, (records, fieldnames) in (extra_csvs or {}).items():
            if Path(filename).name != filename or not filename.endswith(".csv"):
                raise ValueError(f"Invalid extra artifact filename: {filename}")
            _atomic_csv(staging / filename, records, fieldnames)
        artifact_hashes = {
            path.name: sha256_file(path) for path in sorted(staging.iterdir()) if path.is_file()
        }
        summary["output_hashes"] = artifact_hashes
        _atomic_json(staging / "split_summary.json", summary)
        if output_dir.exists():
            if not overwrite:
                raise FileExistsError(f"Destination appeared while staging: {output_dir}")
            backup = output_dir.with_name(f".{output_dir.name}.previous")
            if backup.exists():
                raise FileExistsError(f"Refusing overwrite because backup path exists: {backup}")
            os.replace(output_dir, backup)
            try:
                os.replace(staging, output_dir)
            except Exception:
                os.replace(backup, output_dir)
                raise
            shutil.rmtree(backup)
        else:
            os.replace(staging, output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {path.name: sha256_file(path) for path in sorted(output_dir.iterdir()) if path.is_file()}


def read_split(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDNAMES:
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames!r}")
        return list(reader)


def validate_written_v2(
    output_dir: Path,
    metadata_path: Path,
    data_root: Path | None = None,
    verify_images: bool = False,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    manifest_path = output_dir / "split_manifest.csv"
    if not manifest_path.is_file():
        raise ValueError("split_manifest.csv is required for Split V2 validation")
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    source = read_rows(metadata_path)
    if verify_images:
        failures = []
        for row in source:
            path = resolve_image_path(str(row["path"]), data_root)
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception as exc:  # pragma: no cover - backend-specific
                failures.append({"path": str(path), "error": type(exc).__name__})
                if len(failures) == 20:
                    break
        if failures:
            raise ValueError(f"Image verification failed: {failures}")
    if len(manifest) != len(source):
        raise ValueError("Manifest row count does not match authoritative metadata")
    splits = {name: read_split(output_dir / f"{name}.csv") for name in OUTPUT_SPLITS}
    if splits["val"] != splits["val_known"] + splits["val_unknown"]:
        raise ValueError("val.csv is not byte-order equivalent to val_known + val_unknown")
    component_paths = [row["path"] for name in COMPONENT_SPLITS for row in splits[name]]
    source_paths = [str(row["path"]) for row in source]
    if Counter(component_paths) != Counter(source_paths):
        raise ValueError("Written component CSVs do not account for metadata exactly once")
    train_ids = identity_keys(splits["train"])
    known_ids = identity_keys(splits["val_known"])
    unknown_ids = identity_keys(splits["val_unknown"])
    unknown_extra_ids = identity_keys(splits["val_unknown_extra"])
    if (
        train_ids & unknown_ids
        or known_ids - train_ids
        or unknown_extra_ids - unknown_ids
    ):
        raise ValueError("Written identity-disjointness invariants failed")
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in manifest:
        for field in (
            "image_sha256",
            "observation_group_id",
            "duplicate_group_id",
            "explicit_group_id",
            "similarity_cluster_id",
        ):
            value = row[field]
            if value:
                grouped[(field, value)].add(row["split"])
    bad = [
        key
        for key, values in grouped.items()
        if (
            key[0] in {"image_sha256", "duplicate_group_id"} and len(values) > 1
        )
        or ("train" in values and "val_known" in values)
    ]
    if bad:
        raise ValueError(f"Manifest group crosses train/val_known: {bad[:5]}")
    return {
        "passed": True,
        "metadata_rows": len(source),
        "images_verified": verify_images,
        "manifest_sha256": sha256_file(manifest_path),
        "output_hashes": {
            path.name: sha256_file(path) for path in sorted(output_dir.iterdir()) if path.is_file()
        },
    }


def build_features_by_index(
    candidate_rows: Sequence[dict[str, object]], features: np.ndarray
) -> dict[int, np.ndarray]:
    if len(candidate_rows) != len(features):
        raise ValueError("Candidate row and feature counts differ")
    return {int(row["_idx"]): features[index] for index, row in enumerate(candidate_rows)}


NEAR_DUPLICATE_FIELDS = [
    "dataset",
    "identity",
    "train_path",
    "val_known_path",
    "train_source_index",
    "val_known_source_index",
    "dhash_distance",
    "phash_distance",
]


def _image_perceptual_hashes(path: Path) -> tuple[int, int]:
    import cv2

    with Image.open(path) as image:
        gray = image.convert("L")
        dhash_pixels = np.asarray(
            gray.resize((9, 8), resample=Image.Resampling.BILINEAR), dtype=np.uint8
        )
        dhash_bits = dhash_pixels[:, 1:] > dhash_pixels[:, :-1]
        dhash = int.from_bytes(np.packbits(dhash_bits).tobytes(), "big")
        phash_pixels = np.asarray(
            gray.resize((32, 32), resample=Image.Resampling.BILINEAR), dtype=np.float32
        )
    dct = cv2.dct(phash_pixels)
    low = dct[:8, :8]
    median = float(np.median(low.reshape(-1)[1:]))
    phash = int.from_bytes(np.packbits(low > median).tobytes(), "big")
    return dhash, phash


def audit_cross_split_near_duplicates(
    splits: Mapping[str, Sequence[dict[str, object]]],
    data_root: Path,
    dhash_threshold: int = 4,
    phash_threshold: int = 6,
    perceptual_hashes: dict[int, tuple[int, int]] | None = None,
) -> list[dict[str, object]]:
    """Return conservative perceptual candidates within shared known identities."""
    from concurrent.futures import ThreadPoolExecutor

    train_by_key = group_rows_by_identity(splits["train"])
    val_by_key = group_rows_by_identity(splits["val_known"])
    involved = {
        int(row["_idx"]): row
        for key in sorted(val_by_key)
        for row in train_by_key[key] + val_by_key[key]
    }
    hashes = perceptual_hashes if perceptual_hashes is not None else {}
    missing = [row for index, row in sorted(involved.items()) if index not in hashes]

    def calculate(row):
        return int(row["_idx"]), _image_perceptual_hashes(
            resolve_image_path(str(row["path"]), data_root)
        )

    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as executor:
        for index, values in executor.map(calculate, missing):
            hashes[index] = values
    candidates: list[dict[str, object]] = []
    for key in sorted(val_by_key):
        for val_row in val_by_key[key]:
            val_dhash, val_phash = hashes[int(val_row["_idx"])]
            for train_row in train_by_key[key]:
                train_dhash, train_phash = hashes[int(train_row["_idx"])]
                dhash_distance = (val_dhash ^ train_dhash).bit_count()
                if dhash_distance > dhash_threshold:
                    continue
                phash_distance = (val_phash ^ train_phash).bit_count()
                if phash_distance > phash_threshold:
                    continue
                candidates.append(
                    {
                        "dataset": key[0],
                        "identity": key[1],
                        "train_path": train_row["path"],
                        "val_known_path": val_row["path"],
                        "train_source_index": train_row["_idx"],
                        "val_known_source_index": val_row["_idx"],
                        "dhash_distance": dhash_distance,
                        "phash_distance": phash_distance,
                    }
                )
    return sorted(
        candidates,
        key=lambda item: (
            int(item["phash_distance"]),
            int(item["dhash_distance"]),
            str(item["dataset"]),
            str(item["identity"]),
            str(item["val_known_path"]),
            str(item["train_path"]),
        ),
    )
