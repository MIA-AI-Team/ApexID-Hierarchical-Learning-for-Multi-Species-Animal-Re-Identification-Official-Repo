#!/usr/bin/env python3
"""Build leakage-resistant open-world animal re-identification benchmarks.

The public files written by this module deliberately contain no ground-truth
identity labels.  Files whose names contain ``ground_truth``, ``mapping``, or
``manifest`` are evaluator-only and must not be supplied to an inference
system.  The generated streams support both batch evaluation and stateful
online evaluation; this script constructs streams only and performs no model
inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


LOGGER = logging.getLogger(__name__)
SCRIPT_VERSION = "1.0.0"
INPUT_FIELDS = ("path", "identity", "dataset")
DEFAULT_DATASET_TO_SPECIES = {
    "bristol_gorillas": "gorilla",
    "spac_wild": "gorilla",
    "elephants": "elephant",
    "gcn_id": "gcn",
}
REFERENCE_DATASETS = ("bristol_gorillas", "spac_wild", "elephants")
DISCOVERY_DATASET = "gcn_id"
PROTOCOLS = {"open20": 0.20, "open40": 0.40, "open60": 0.60}
PUBLIC_FINAL_FIELDS = ("image_id", "arrival_index", "path", "species", "dataset")
PUBLIC_DEVELOPMENT_FIELDS = ("image_id", "path", "species", "dataset")
PRIVATE_GROUND_TRUTH_FIELDS = (
    "image_id",
    "true_identity",
    "original_identity",
    "scenario",
    "species",
    "dataset",
    "known_database_id",
    "encounter_id",
)

MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
MONTH_PATTERN = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?"
)
SPAC_PATTERN = re.compile(r"^([A-Za-z]+\d+)_([0-9]{8})_")


@dataclass(frozen=True)
class BenchmarkConfig:
    input_path: Path
    output_dir: Path
    seed: int = 42
    dev_novel_fraction: float = 0.30
    gallery_encounter_fraction: float = 0.60
    num_stream_orders: int = 10
    dataset_to_species: Mapping[str, str] | None = None
    data_root: Path | None = None
    audit_image_hashes: bool = False


@dataclass(frozen=True)
class ProtocolConfig:
    name: str
    openness: float

    @property
    def known_fraction(self) -> float:
        return 1.0 - self.openness


def _stable_seed(seed: int, *parts: object) -> int:
    text = "|".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def _rng(seed: int, *parts: object) -> random.Random:
    return random.Random(_stable_seed(seed, *parts))


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _identity_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["dataset"]), str(row["identity"])


def _species_prefix(species: str) -> str:
    known = {"gorilla": "GOR", "elephant": "ELP", "gcn": "GCN"}
    return known.get(species.lower(), re.sub(r"[^A-Z]", "", species.upper())[:3] or "UNK")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json_object(value: str) -> dict[str, str]:
    candidate = Path(value)
    raw = candidate.read_text(encoding="utf-8") if candidate.exists() else value
    parsed = json.loads(raw)
    if not isinstance(parsed, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in parsed.items()
    ):
        raise ValueError("Species mapping must be a JSON object with string keys and values")
    return parsed


def infer_species(dataset: str, dataset_to_species: Mapping[str, str]) -> str:
    try:
        return dataset_to_species[dataset]
    except KeyError as exc:
        raise ValueError(f"No species mapping configured for dataset {dataset!r}") from exc


def infer_spac_encounter(identity: str, filename: str) -> tuple[str, bool]:
    """Return a camera/day encounter for Gorilla-SPAC-Wild filenames.

    The published convention is camera/location plus calendar day.  The data
    also contains non-``R`` camera prefixes (for example ``Tm002``), so the
    camera token is intentionally broader than the illustrative ``R\\d+``.
    """

    match = SPAC_PATTERN.match(Path(filename).stem)
    if match:
        camera, date = match.groups()
        return f"spac_wild:{identity}:{camera}_{date}", True
    return f"spac_wild:{identity}:unknown_encounter", False


def infer_bristol_encounter(identity: str, filename: str) -> tuple[str, bool]:
    """Group numbered sessions and trial sequences without splitting frames."""

    stem = Path(filename).stem
    escaped_identity = re.escape(identity)
    session = re.match(rf"^{escaped_identity}-(\d+)-", stem, flags=re.IGNORECASE)
    if session:
        return f"bristol_gorillas:{identity}:session_{session.group(1)}", True
    trial = re.match(
        rf"^{escaped_identity}-(?:(?:[A-Za-z]+)-)?trial(?:[-_]|$)",
        stem,
        flags=re.IGNORECASE,
    )
    if trial:
        return f"bristol_gorillas:{identity}:trial", True
    return f"bristol_gorillas:{identity}:unknown_sequence", False


def _normalise_elephant_date(stem: str) -> str | None:
    """Extract the rightmost explicit date token without inventing precision.

    ``YYYY-MM-DD``, ``YYYY-MM``, and ``YYYY`` preserve the precision actually
    present in filenames.  A year-only token is still a useful conservative
    encounter group: every image for that identity with that year remains
    together.
    """

    patterns = (
        re.compile(
            rf"(?i)(?P<day>\d{{1,2}})\s*(?P<month>{MONTH_PATTERN})\s*(?P<year>\d{{4}})"
        ),
        re.compile(
            rf"(?i)(?P<month>{MONTH_PATTERN})\s*(?P<day>\d{{1,2}})\s*,?\s*(?P<year>\d{{4}})"
        ),
        re.compile(rf"(?i)(?P<month>{MONTH_PATTERN})\s*(?P<year>\d{{4}})"),
        re.compile(r"(?<!\d)(?P<year>(?:19|20)\d{2})(?!\d)"),
    )
    candidates: list[tuple[int, int, dict[str, str | None]]] = []
    for pattern in patterns:
        for match in pattern.finditer(stem):
            candidates.append((match.end(), match.start(), match.groupdict()))
    if not candidates:
        return None
    _end, _start, groups = max(candidates, key=lambda value: (value[0], value[0] - value[1]))
    year = groups["year"]
    month_text = groups.get("month")
    day_text = groups.get("day")
    fields = [year]
    if month_text:
        fields.append(f"{MONTHS[month_text[:3].lower()]:02d}")
    if day_text:
        fields.append(f"{int(day_text):02d}")
    return "-".join(fields)


def infer_elephant_encounter(identity: str, filename: str) -> tuple[str, bool]:
    date = _normalise_elephant_date(Path(filename).stem)
    if date is not None:
        return f"elephants:{identity}:{date}", True
    return f"elephants:{identity}:unknown_date", False


def infer_encounter_id(dataset: str, identity: str, path: str) -> tuple[str, bool]:
    if dataset == "spac_wild":
        return infer_spac_encounter(identity, path)
    if dataset == "bristol_gorillas":
        return infer_bristol_encounter(identity, path)
    if dataset == "elephants":
        return infer_elephant_encounter(identity, path)
    if dataset == DISCOVERY_DATASET:
        return f"gcn_id:{identity}:all_samples", True
    return f"{dataset}:{identity}:unknown_encounter", False


def load_and_validate_csv(
    input_path: Path, dataset_to_species: Mapping[str, str]
) -> tuple[list[dict[str, str]], list[str]]:
    """Load source rows, attach stable metadata, and reject ambiguous inputs."""

    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != INPUT_FIELDS:
            raise ValueError(
                f"Expected exactly {list(INPUT_FIELDS)!r} in {input_path}; got {reader.fieldnames!r}"
            )
        raw_rows = list(reader)
    if not raw_rows:
        raise ValueError(f"Input CSV {input_path} is empty")
    invalid = [row for row in raw_rows if any(not row[field].strip() for field in INPUT_FIELDS)]
    if invalid:
        raise ValueError(f"Input CSV contains {len(invalid)} blank required rows")
    paths = [row["path"] for row in raw_rows]
    duplicates = [path for path, count in Counter(paths).items() if count > 1]
    if duplicates:
        raise ValueError(f"Input CSV contains duplicate paths, e.g. {duplicates[:5]}")

    rows: list[dict[str, str]] = []
    warnings: list[str] = []
    failed_parse_counts: Counter[str] = Counter()
    for image_index, row in enumerate(sorted(raw_rows, key=lambda item: item["path"]), start=1):
        dataset, identity = row["dataset"], row["identity"]
        species = infer_species(dataset, dataset_to_species)
        encounter_id, parsed = infer_encounter_id(dataset, identity, row["path"])
        if not parsed:
            failed_parse_counts[dataset] += 1
        rows.append(
            {
                **row,
                "species": species,
                "encounter_id": encounter_id,
                "image_id": f"IMG_{image_index:07d}",
            }
        )
    for dataset, count in sorted(failed_parse_counts.items()):
        warnings.append(
            f"{dataset}: {count} filenames used the conservative identity-scoped encounter fallback"
        )
    return rows, warnings


def group_by_identity(rows: Iterable[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[_identity_key(row)].append(row)
    return {key: sorted(value, key=lambda row: row["path"]) for key, value in groups.items()}


def compute_identity_statistics(rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    groups = group_by_identity(rows)
    by_dataset: dict[str, list[list[dict[str, str]]]] = defaultdict(list)
    for (dataset, _identity), identity_rows in groups.items():
        by_dataset[dataset].append(identity_rows)
    statistics: list[dict[str, Any]] = []
    for dataset in sorted(by_dataset):
        identity_rows = by_dataset[dataset]
        counts = sorted(len(items) for items in identity_rows)
        encounter_counts = [len({item["encounter_id"] for item in items}) for items in identity_rows]
        statistics.append(
            {
                "dataset": dataset,
                "identities": len(identity_rows),
                "min_images": min(counts),
                "median_images": _median(counts),
                "mean_images": sum(counts) / len(counts),
                "max_images": max(counts),
                "singletons": sum(count == 1 for count in counts),
                "identities_with_at_least_two_images": sum(count >= 2 for count in counts),
                "identities_with_at_least_two_encounters": sum(
                    count >= 2 for count in encounter_counts
                ),
            }
        )
    return statistics


def _median(values: Sequence[int]) -> float:
    middle = len(values) // 2
    return float(values[middle]) if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def determine_known_eligible_identities(
    identity_rows: Mapping[tuple[str, str], Sequence[dict[str, str]]]
) -> set[tuple[str, str]]:
    return {
        key
        for key, rows in identity_rows.items()
        if len({row["encounter_id"] for row in rows}) >= 2
    }


def stratified_identity_split(
    keys: Sequence[tuple[str, str]],
    identity_rows: Mapping[tuple[str, str], Sequence[dict[str, str]]],
    take: int,
    seed: int,
    purpose: str,
) -> set[tuple[str, str]]:
    """Select identities proportionally across log-count quantile bins."""

    ordered = sorted(keys, key=lambda key: (math.log(len(identity_rows[key])), key))
    if take <= 0:
        return set()
    if take >= len(ordered):
        return set(ordered)
    bin_count = min(5, len(ordered) // 2) if len(ordered) >= 10 else 1
    bins: list[list[tuple[str, str]]] = []
    for index in range(bin_count):
        start = index * len(ordered) // bin_count
        end = (index + 1) * len(ordered) // bin_count
        bins.append(ordered[start:end])
    allocations = [take * len(bin_rows) / len(ordered) for bin_rows in bins]
    counts = [min(len(bin_rows), int(math.floor(value))) for bin_rows, value in zip(bins, allocations)]
    remainder = take - sum(counts)
    for index in sorted(range(len(bins)), key=lambda idx: (-(allocations[idx] - counts[idx]), idx)):
        if remainder <= 0:
            break
        if counts[index] < len(bins[index]):
            counts[index] += 1
            remainder -= 1
    selected: set[tuple[str, str]] = set()
    for index, (bin_rows, count) in enumerate(zip(bins, counts)):
        shuffled = list(bin_rows)
        _rng(seed, purpose, index).shuffle(shuffled)
        selected.update(shuffled[:count])
    return selected


def split_known_identity_encounters(
    rows: Sequence[dict[str, str]], gallery_fraction: float, seed: int, identity_key: tuple[str, str]
) -> dict[str, list[dict[str, str]]]:
    """Split one known identity into encounter-disjoint database/dev/final rows."""

    encounters: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        encounters[row["encounter_id"]].append(row)
    encounter_ids = sorted(encounters)
    if len(encounter_ids) < 2:
        raise ValueError(f"Known identity {identity_key!r} has fewer than two encounters")
    _rng(seed, "known-encounters", *identity_key).shuffle(encounter_ids)
    if len(encounter_ids) == 2:
        assignments = {"database": encounter_ids[:1], "development": [], "final_test": encounter_ids[1:]}
    else:
        gallery_count = min(
            len(encounter_ids) - 2,
            max(1, _round_half_up(len(encounter_ids) * gallery_fraction)),
        )
        remaining = encounter_ids[gallery_count:]
        development_count = len(remaining) // 2
        assignments = {
            "database": encounter_ids[:gallery_count],
            "development": remaining[:development_count],
            "final_test": remaining[development_count:],
        }
    return {
        partition: [row for encounter_id in encounter_ids_for_partition for row in encounters[encounter_id]]
        for partition, encounter_ids_for_partition in assignments.items()
    }


def _select_identity_roles(
    identity_rows: Mapping[tuple[str, str], Sequence[dict[str, str]]],
    protocol: ProtocolConfig,
    config: BenchmarkConfig,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]], set[tuple[str, str]], list[str]]:
    """Return known, development-novel, and final-novel keys for one protocol."""

    known_keys: set[tuple[str, str]] = set()
    dev_novel_keys: set[tuple[str, str]] = set()
    final_novel_keys: set[tuple[str, str]] = set()
    warnings: list[str] = []
    eligible = determine_known_eligible_identities(identity_rows)
    for dataset in REFERENCE_DATASETS:
        dataset_keys = sorted(key for key in identity_rows if key[0] == dataset)
        if not dataset_keys:
            warnings.append(f"{dataset}: no rows were available")
            continue
        eligible_keys = sorted(key for key in dataset_keys if key in eligible)
        if len(dataset_keys) == 1:
            target_known = 0
        else:
            target_known = max(1, min(len(dataset_keys) - 1, _round_half_up(len(dataset_keys) * protocol.known_fraction)))
        actual_known = min(target_known, len(eligible_keys))
        if actual_known != target_known:
            warnings.append(
                f"{dataset}: requested {target_known} known identities, but only {len(eligible_keys)} are encounter-eligible"
            )
        chosen = stratified_identity_split(
            eligible_keys, identity_rows, actual_known, config.seed, f"{protocol.name}:{dataset}:known"
        )
        known_keys.update(chosen)
        novel_keys = sorted(set(dataset_keys) - chosen)
        if not novel_keys:
            warnings.append(f"{dataset}: no novel identities remain after known selection")
            continue
        dev_target = _round_half_up(len(novel_keys) * config.dev_novel_fraction)
        if len(novel_keys) == 1:
            dev_target = 0
            warnings.append(
                f"{dataset}: one novel identity remains; it is reserved for final test rather than development"
            )
        else:
            dev_target = min(dev_target, len(novel_keys) - 1)
        dev_selected = stratified_identity_split(
            novel_keys, identity_rows, dev_target, config.seed, f"{protocol.name}:{dataset}:dev-novel"
        )
        dev_novel_keys.update(dev_selected)
        final_novel_keys.update(set(novel_keys) - dev_selected)
    return known_keys, dev_novel_keys, final_novel_keys, warnings


def assign_canonical_ids(
    identity_rows: Mapping[tuple[str, str], Sequence[dict[str, str]]],
    known_keys: set[tuple[str, str]],
    final_or_dev_novel_keys: set[tuple[str, str]],
) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], str | None]]:
    """Create public DB IDs and private true IDs, scoped deterministically by species."""

    database_ids: dict[tuple[str, str], str] = {}
    true_ids: dict[tuple[str, str], str | None] = {}
    by_species: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in known_keys:
        by_species[identity_rows[key][0]["species"]].append(key)
    for species, keys in by_species.items():
        for index, key in enumerate(sorted(keys), start=1):
            database_ids[key] = f"DB_{_species_prefix(species)}_{index:04d}"
            true_ids[key] = database_ids[key]

    novel_by_species: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in final_or_dev_novel_keys:
        novel_by_species[identity_rows[key][0]["species"]].append(key)
    for species, keys in novel_by_species.items():
        for index, key in enumerate(sorted(keys), start=1):
            true_ids[key] = f"GT_NOVEL_{_species_prefix(species)}_{index:04d}"

    gcn_keys = sorted(key for key in identity_rows if key[0] == DISCOVERY_DATASET)
    for index, key in enumerate(gcn_keys, start=1):
        true_ids[key] = f"GT_GCN_{index:04d}"
    return database_ids, true_ids


def _private_row(row: Mapping[str, str], true_identity: str, scenario: str, database_id: str | None) -> dict[str, str]:
    return {
        "image_id": row["image_id"],
        "true_identity": true_identity,
        "original_identity": row["identity"],
        "scenario": scenario,
        "species": row["species"],
        "dataset": row["dataset"],
        "known_database_id": database_id or "",
        "encounter_id": row["encounter_id"],
    }


def build_protocol(
    rows: Sequence[dict[str, str]], protocol: ProtocolConfig, config: BenchmarkConfig
) -> dict[str, Any]:
    identity_rows = group_by_identity(rows)
    known_keys, dev_novel_keys, final_novel_keys, warnings = _select_identity_roles(
        identity_rows, protocol, config
    )
    database_ids, true_ids = assign_canonical_ids(
        identity_rows, known_keys, dev_novel_keys | final_novel_keys
    )
    database: list[dict[str, str]] = []
    development: list[dict[str, str]] = []
    development_ground_truth: list[dict[str, str]] = []
    final_private: list[dict[str, str]] = []
    manifest: list[dict[str, str]] = []

    def add_manifest(
        row: Mapping[str, str], partition: str, scenario: str, database_id: str | None, canonical: str
    ) -> None:
        manifest.append(
            {
                "path": row["path"],
                "original_identity": row["identity"],
                "canonical_identity": canonical,
                "dataset": row["dataset"],
                "species": row["species"],
                "encounter_id": row["encounter_id"],
                "protocol": protocol.name,
                "partition": partition,
                "scenario": scenario,
                "database_id": database_id or "",
                "image_id": row["image_id"],
            }
        )

    for key in sorted(known_keys):
        database_id = database_ids[key]
        split = split_known_identity_encounters(
            identity_rows[key], config.gallery_encounter_fraction, config.seed, key
        )
        for row in split["database"]:
            database.append(
                {
                    "path": row["path"],
                    "database_id": database_id,
                    "species": row["species"],
                    "dataset": row["dataset"],
                    "encounter_id": row["encounter_id"],
                }
            )
            add_manifest(row, "database", "known_identity", database_id, database_id)
        for row in split["development"]:
            development.append({field: row[field] for field in PUBLIC_DEVELOPMENT_FIELDS})
            development_ground_truth.append(_private_row(row, database_id, "known_identity", database_id))
            add_manifest(row, "development", "known_identity", database_id, database_id)
        for row in split["final_test"]:
            final_private.append(_private_row(row, database_id, "known_identity", database_id))
            add_manifest(row, "final_test", "known_identity", database_id, database_id)

    for key in sorted(dev_novel_keys):
        canonical = str(true_ids[key])
        for row in identity_rows[key]:
            development.append({field: row[field] for field in PUBLIC_DEVELOPMENT_FIELDS})
            development_ground_truth.append(
                _private_row(row, canonical, "novel_identity_known_species", None)
            )
            add_manifest(row, "development", "novel_identity_known_species", None, canonical)

    for key in sorted(final_novel_keys):
        canonical = str(true_ids[key])
        for row in identity_rows[key]:
            final_private.append(_private_row(row, canonical, "novel_identity_known_species", None))
            add_manifest(row, "final_test", "novel_identity_known_species", None, canonical)

    for key in sorted(key for key in identity_rows if key[0] == DISCOVERY_DATASET):
        canonical = str(true_ids[key])
        for row in identity_rows[key]:
            final_private.append(_private_row(row, canonical, "discovery_only_species", None))
            add_manifest(row, "final_test", "discovery_only_species", None, canonical)

    ordering = list(final_private)
    _rng(config.seed, protocol.name, "final-order").shuffle(ordering)
    path_by_image_id = {row["image_id"]: row["path"] for row in rows}
    final_test = [
        {
            "image_id": row["image_id"],
            "arrival_index": str(index),
            "path": path_by_image_id[row["image_id"]],
            "species": row["species"],
            "dataset": row["dataset"],
        }
        for index, row in enumerate(ordering, start=1)
    ]
    final_no_species = [
        {field: row[field] for field in ("image_id", "arrival_index", "path")} for row in final_test
    ]
    private_by_image_id = {row["image_id"]: row for row in final_private}
    final_ground_truth = [private_by_image_id[row["image_id"]] for row in final_test]

    registry = [
        {
            "database_id": database_ids[key],
            "dataset": key[0],
            "species": identity_rows[key][0]["species"],
        }
        for key in sorted(known_keys)
    ]
    mapping_rows: list[dict[str, str]] = []
    for key in sorted(identity_rows):
        if key[0] == DISCOVERY_DATASET:
            role = "discovery_only_species"
        elif key in known_keys:
            role = "known_identity"
        else:
            role = "novel_identity_known_species"
        mapping_rows.append(
            {
                "dataset": key[0],
                "species": identity_rows[key][0]["species"],
                "original_identity": key[1],
                "canonical_identity": str(true_ids.get(key, "")),
                "database_id": database_ids.get(key, ""),
                "role": role,
            }
        )
    streams = build_stream_orders(final_test, config.seed, config.num_stream_orders)
    result: dict[str, Any] = {
        "protocol": protocol.name,
        "openness": protocol.openness,
        "database": sorted(database, key=lambda row: (row["database_id"], row["path"])),
        "development": sorted(development, key=lambda row: row["image_id"]),
        "development_ground_truth": sorted(development_ground_truth, key=lambda row: row["image_id"]),
        "final_test": final_test,
        "final_test_no_species": final_no_species,
        "final_test_ground_truth": final_ground_truth,
        "known_identity_registry": registry,
        "private_identity_mapping": mapping_rows,
        "split_manifest": sorted(manifest, key=lambda row: row["image_id"]),
        "prediction_template": [
            {"image_id": row["image_id"], "predicted_identity": ""} for row in final_test
        ],
        "streams": streams,
        "warnings": warnings,
    }
    result["validation_report"] = validate_benchmark(result, rows, config)
    result["split_summary"] = build_split_summary(result, compute_identity_statistics(rows))
    return result


def build_stream_orders(final_test: Sequence[Mapping[str, str]], seed: int, count: int) -> dict[int, list[dict[str, str]]]:
    if count < 1:
        raise ValueError("num_stream_orders must be at least one")
    streams: dict[int, list[dict[str, str]]] = {}
    for stream_seed in range(seed, seed + count):
        ordered = [dict(row) for row in final_test]
        _rng(stream_seed, "online-stream").shuffle(ordered)
        streams[stream_seed] = [
            {**row, "arrival_index": str(index)} for index, row in enumerate(ordered, start=1)
        ]
    return streams


def validate_benchmark(
    benchmark: Mapping[str, Any], rows: Sequence[Mapping[str, str]], config: BenchmarkConfig
) -> dict[str, Any]:
    """Fail loudly if membership, leakage, or public-schema invariants break."""

    database = benchmark["database"]
    development = benchmark["development"]
    final_test = benchmark["final_test"]
    manifest = benchmark["split_manifest"]
    database_paths = {row["path"] for row in database}
    development_paths = {row["path"] for row in development}
    final_paths = {row["path"] for row in final_test}
    partitions = {"database": database_paths, "development": development_paths, "final_test": final_paths}
    for left, right in (("database", "development"), ("database", "final_test"), ("development", "final_test")):
        if partitions[left] & partitions[right]:
            raise ValueError(f"Path leakage between {left} and {right}")
    manifest_paths = [row["path"] for row in manifest]
    source_paths = [row["path"] for row in rows]
    if Counter(manifest_paths) != Counter(source_paths):
        raise ValueError("Split manifest does not cover every input row exactly once")
    if len(set(manifest_paths)) != len(manifest_paths):
        raise ValueError("Duplicate path appears in split manifest")

    gcn_source = {row["path"] for row in rows if row["dataset"] == DISCOVERY_DATASET}
    if any(row["dataset"] == DISCOVERY_DATASET for row in database + development):
        raise ValueError("GCN rows leaked into database or development")
    if {row["path"] for row in final_test if row["dataset"] == DISCOVERY_DATASET} != gcn_source:
        raise ValueError("Final test does not contain exactly all GCN rows")

    manifest_by_image = {row["image_id"]: row for row in manifest}
    database_encounters: dict[str, set[str]] = defaultdict(set)
    query_encounters: dict[str, set[str]] = defaultdict(set)
    for row in manifest:
        if row["scenario"] != "known_identity" or not row["database_id"]:
            continue
        if row["partition"] == "database":
            database_encounters[row["database_id"]].add(row["encounter_id"])
        else:
            query_encounters[row["database_id"]].add(row["encounter_id"])
    leaking = [key for key in database_encounters if database_encounters[key] & query_encounters[key]]
    if leaking:
        raise ValueError(f"Known identity encounter leakage for database IDs {leaking[:5]}")

    dev_novel = {
        (row["dataset"], row["original_identity"])
        for row in benchmark["development_ground_truth"]
        if row["scenario"] == "novel_identity_known_species"
    }
    final_novel = {
        (row["dataset"], row["original_identity"])
        for row in benchmark["final_test_ground_truth"]
        if row["scenario"] == "novel_identity_known_species"
    }
    database_original = {
        (row["dataset"], row["original_identity"])
        for row in manifest
        if row["partition"] == "database"
    }
    if dev_novel & final_novel or final_novel & database_original:
        raise ValueError("Novel identities overlap database or development-novel identities")

    if final_test and tuple(final_test[0].keys()) != PUBLIC_FINAL_FIELDS:
        raise ValueError("final_test.csv public schema is incorrect")
    sensitive = {"identity", "original_identity", "true_identity", "scenario", "is_known"}
    for public_rows in (benchmark["development"], final_test, benchmark["final_test_no_species"]):
        if public_rows and sensitive & set(public_rows[0]):
            raise ValueError("A public output exposes ground-truth information")
    final_ids = [row["image_id"] for row in final_test]
    if len(final_ids) != len(set(final_ids)):
        raise ValueError("Final test contains duplicate image IDs")
    for stream_seed, stream in benchmark["streams"].items():
        if {row["image_id"] for row in stream} != set(final_ids):
            raise ValueError(f"Stream {stream_seed} membership differs from final test")
        if [int(row["arrival_index"]) for row in stream] != list(range(1, len(stream) + 1)):
            raise ValueError(f"Stream {stream_seed} has invalid arrival indices")

    image_hash_report: dict[str, Any] | None = None
    if config.audit_image_hashes:
        image_hash_report = audit_image_hashes(manifest, config.data_root)
    return {
        "status": "PASS",
        "audits": {
            "path_disjointness": "PASS",
            "input_coverage": "PASS",
            "duplicate_paths": "PASS",
            "novel_identity_disjointness": "PASS",
            "known_encounter_disjointness": "PASS",
            "gcn_isolation": "PASS",
            "public_schema": "PASS",
            "stream_membership": "PASS",
            "image_hashes": "PASS" if config.audit_image_hashes else "NOT_REQUESTED",
        },
        "image_hash_report": image_hash_report,
        "warnings": list(benchmark.get("warnings", [])),
    }


def audit_image_hashes(manifest: Sequence[Mapping[str, str]], data_root: Path | None) -> dict[str, Any]:
    hashes: dict[str, set[str]] = defaultdict(set)
    for row in manifest:
        candidate = Path(row["path"])
        if not candidate.is_absolute() and data_root is not None:
            candidate = data_root / candidate
        if not candidate.exists():
            raise FileNotFoundError(f"Cannot hash missing image {candidate}; disable --audit-image-hashes or pass --data-root")
        hashes[_sha256_file(candidate)].add(row["partition"])
    cross_partition = [digest for digest, partitions in hashes.items() if len(partitions) > 1]
    if cross_partition:
        raise ValueError(f"Exact image-content leakage across partitions: {cross_partition[:5]}")
    return {"unique_hashes": len(hashes), "cross_partition_duplicate_hashes": 0}


def build_split_summary(benchmark: Mapping[str, Any], identity_statistics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    manifest = benchmark["split_manifest"]
    rows: list[dict[str, Any]] = []
    for partition in ("database", "development", "final_test"):
        for dataset in sorted({row["dataset"] for row in manifest}):
            for scenario in ("known_identity", "novel_identity_known_species", "discovery_only_species"):
                subset = [
                    row
                    for row in manifest
                    if row["partition"] == partition
                    and row["dataset"] == dataset
                    and row["scenario"] == scenario
                ]
                if subset:
                    rows.append(
                        {
                            "partition": partition,
                            "dataset": dataset,
                            "scenario": scenario,
                            "images": len(subset),
                            "identities": len({(row["dataset"], row["original_identity"]) for row in subset}),
                            "encounters": len({row["encounter_id"] for row in subset}),
                        }
                    )
    return {
        "protocol": benchmark["protocol"],
        "openness": benchmark["openness"],
        "partitions": rows,
        "identity_statistics": list(identity_statistics),
    }


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_protocol(benchmark: Mapping[str, Any], config: BenchmarkConfig, input_sha256: str) -> Path:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = str(benchmark["protocol"])
    temporary = Path(tempfile.mkdtemp(prefix=f".{protocol}.", dir=output_dir))
    try:
        _write_csv(temporary / "database.csv", ("path", "database_id", "species", "dataset", "encounter_id"), benchmark["database"])
        _write_csv(temporary / "development.csv", PUBLIC_DEVELOPMENT_FIELDS, benchmark["development"])
        _write_csv(temporary / "development_ground_truth.csv", PRIVATE_GROUND_TRUTH_FIELDS, benchmark["development_ground_truth"])
        _write_csv(temporary / "final_test.csv", PUBLIC_FINAL_FIELDS, benchmark["final_test"])
        _write_csv(temporary / "final_test_no_species.csv", ("image_id", "arrival_index", "path"), benchmark["final_test_no_species"])
        _write_csv(temporary / "final_test_ground_truth.csv", PRIVATE_GROUND_TRUTH_FIELDS, benchmark["final_test_ground_truth"])
        _write_csv(temporary / "known_identity_registry.csv", ("database_id", "dataset", "species"), benchmark["known_identity_registry"])
        _write_csv(temporary / "private_identity_mapping.csv", ("dataset", "species", "original_identity", "canonical_identity", "database_id", "role"), benchmark["private_identity_mapping"])
        _write_csv(temporary / "split_manifest.csv", ("path", "original_identity", "canonical_identity", "dataset", "species", "encounter_id", "protocol", "partition", "scenario", "database_id", "image_id"), benchmark["split_manifest"])
        _write_csv(temporary / "prediction_template.csv", ("image_id", "predicted_identity"), benchmark["prediction_template"])
        summary_rows = benchmark["split_summary"]["partitions"]
        _write_csv(temporary / "split_summary.csv", ("partition", "dataset", "scenario", "images", "identities", "encounters"), summary_rows)
        _write_json(temporary / "split_summary.json", benchmark["split_summary"])
        _write_json(temporary / "validation_report.json", benchmark["validation_report"])
        _write_json(
            temporary / "config.json",
            {
                "script_version": SCRIPT_VERSION,
                "input_csv_sha256": input_sha256,
                "seed": config.seed,
                "protocol": protocol,
                "openness": benchmark["openness"],
                "dev_novel_fraction": config.dev_novel_fraction,
                "gallery_encounter_fraction": config.gallery_encounter_fraction,
                "num_stream_orders": config.num_stream_orders,
                "dataset_to_species": dict(config.dataset_to_species or {}),
                "encounter_rules_version": "1.0.0",
            },
        )
        streams_dir = temporary / "streams"
        streams_dir.mkdir()
        for stream_seed, stream in benchmark["streams"].items():
            _write_csv(streams_dir / f"stream_seed_{stream_seed}.csv", PUBLIC_FINAL_FIELDS, stream)
            _write_csv(
                streams_dir / f"stream_seed_{stream_seed}_no_species.csv",
                ("image_id", "arrival_index", "path"),
                ({field: row[field] for field in ("image_id", "arrival_index", "path")} for row in stream),
            )
        destination = output_dir / protocol
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
        return destination
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def print_summary(benchmark: Mapping[str, Any]) -> None:
    summary = benchmark["split_summary"]["partitions"]
    print("=" * 60)
    print(f"ApexID Benchmark: {benchmark['protocol']}")
    print("=" * 60)
    for partition in ("database", "development", "final_test"):
        rows = [row for row in summary if row["partition"] == partition]
        print(f"\n{partition.upper()}")
        print("-" * len(partition))
        for row in rows:
            print(
                f"{row['dataset']:18} {row['scenario']:29} "
                f"images={row['images']:5} identities={row['identities']:4} encounters={row['encounters']:4}"
            )
    print("\nLEAKAGE AUDIT")
    print("-" * 13)
    for name, status in benchmark["validation_report"]["audits"].items():
        print(f"{name}: {status}")
    for warning in benchmark["validation_report"]["warnings"]:
        print(f"WARNING: {warning}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("merged.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("apexid_benchmark"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-novel-fraction", type=float, default=0.30)
    parser.add_argument("--gallery-encounter-fraction", type=float, default=0.60)
    parser.add_argument("--num-stream-orders", type=int, default=10)
    parser.add_argument(
        "--species-map-json",
        help="JSON object or path to a JSON file overriding the default dataset/species map.",
    )
    parser.add_argument("--data-root", type=Path, help="Root used to resolve relative image paths for hashing.")
    parser.add_argument("--audit-image-hashes", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")
    if not 0.0 <= args.dev_novel_fraction < 1.0:
        raise ValueError("--dev-novel-fraction must be in [0, 1)")
    if not 0.0 < args.gallery_encounter_fraction < 1.0:
        raise ValueError("--gallery-encounter-fraction must be in (0, 1)")
    mapping = dict(DEFAULT_DATASET_TO_SPECIES)
    if args.species_map_json:
        mapping.update(_load_json_object(args.species_map_json))
    config = BenchmarkConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        seed=args.seed,
        dev_novel_fraction=args.dev_novel_fraction,
        gallery_encounter_fraction=args.gallery_encounter_fraction,
        num_stream_orders=args.num_stream_orders,
        dataset_to_species=mapping,
        data_root=args.data_root,
        audit_image_hashes=args.audit_image_hashes,
    )
    rows, parse_warnings = load_and_validate_csv(config.input_path, mapping)
    input_sha256 = _sha256_file(config.input_path)
    for warning in parse_warnings:
        LOGGER.warning(warning)
    for name, openness in PROTOCOLS.items():
        benchmark = build_protocol(rows, ProtocolConfig(name, openness), config)
        benchmark["warnings"] = parse_warnings + benchmark["warnings"]
        # Revalidate after attaching source-parser warnings so the report captures all compromises.
        benchmark["validation_report"] = validate_benchmark(benchmark, rows, config)
        destination = write_protocol(benchmark, config, input_sha256)
        print_summary(benchmark)
        LOGGER.info("Wrote %s", destination)


if __name__ == "__main__":
    main()
