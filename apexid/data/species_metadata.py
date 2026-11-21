"""Training-only identity-to-species metadata for hierarchical objectives."""

from dataclasses import asdict, dataclass, field
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd


MISSING_SPECIES_INDEX = -100

# Explicitly approved single-species dataset fills at the shared category level.
# Humpbacks use "whale" to match HappyWhale's coarse whale/dolphin supervision.
BHCA_DATASET_TO_SPECIES = {
    "CzechLynx": "lynx",
    "SalamanderID2025": "salamander",
    "humpback-whale-identification": "whale",
}

FROZEN_SPLIT_MANIFEST = Path(__file__).resolve().parents[2] / "open_set_splits_v2" / "split_manifest.csv"


@dataclass(frozen=True)
class SpeciesEnrichmentReport:
    """Audit record for deterministic dataset-to-species enrichment."""

    num_rows: int
    null_species_before: int
    null_species_after: int
    rows_filled: int
    rows_by_dataset: dict
    canonical_dataset_species: dict
    unresolved_datasets: list
    happywhale_rows_filled_from_official: int = 0
    rows_normalized: int = 0
    rows_normalized_by_dataset: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def _known_species(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    if not value or value.lower() in {"nan", "none", "null", "unknown", "-100"}:
        return None
    return value


def _canonicalize_whale_category(value):
    """Normalize explicit whale aliases without inferring from other columns."""
    known = _known_species(value)
    if known is not None and known.casefold() in {
        "whale", "humpback-whale", "humpback_whale", "humpback whale"
    }:
        return "whale"
    return known


def canonicalize_happywhale_species(value):
    """Map HappyWhale competition species to ``dolphin`` or ``whale``.

    If the official species string contains ``dolphin`` (or the known typo
    ``dolpin``), the label is ``dolphin``; everything else is ``whale``.
    """
    known = _known_species(value)
    if known is None:
        return None
    lowered = known.casefold()
    if "dolphin" in lowered or "dolpin" in lowered:
        return "dolphin"
    if known in {"dolphin", "whale"}:
        return known
    return "whale"


def load_happywhale_official_species(path):
    """Load official HappyWhale labels collapsed to whale/dolphin."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"HappyWhale official species CSV not found: {path}")
    official = pd.read_csv(path)
    required = {"image", "species", "individual_id"}
    missing = required - set(official.columns)
    if missing:
        raise ValueError(
            f"HappyWhale species CSV {path} missing columns: {sorted(missing)}"
        )
    frame = official.loc[:, ["image", "species", "individual_id"]].copy()
    frame["image"] = frame["image"].astype(str).str.strip()
    frame["individual_id"] = frame["individual_id"].astype(str).str.strip()
    frame["species"] = frame["species"].map(canonicalize_happywhale_species)
    if frame["image"].duplicated().any():
        dupes = frame.loc[frame["image"].duplicated(), "image"].head(5).tolist()
        raise ValueError(f"HappyWhale species CSV has duplicate images, e.g. {dupes}")
    if frame["species"].isna().any():
        raise ValueError(f"HappyWhale species CSV has blank species rows: {path}")
    bad = sorted(set(frame["species"]) - {"dolphin", "whale"})
    if bad:
        raise ValueError(f"HappyWhale species collapsed to unexpected values: {bad}")
    per_id = frame.groupby("individual_id")["species"].nunique()
    conflicts = per_id[per_id > 1]
    if not conflicts.empty:
        raise ValueError(
            "HappyWhale official labels assign both whale and dolphin to one identity; "
            f"examples: {conflicts.head(5).index.tolist()}"
        )
    return frame.set_index("image", drop=False)


def fill_happywhale_species_from_official(
    frame,
    official,
    *,
    dataset_column="dataset",
    species_column="species",
    path_column="path",
    identity_column="identity",
    dataset_name="HappyWhale",
):
    """Fill HappyWhale species as ``whale``/``dolphin`` from official labels.

    Matching is by image basename under ``path``. When ``identity`` is present it
    must equal the official ``individual_id``. Returns ``(species_series, rows_filled)``.
    """
    if not isinstance(official, pd.DataFrame):
        official = load_happywhale_official_species(official)

    if dataset_column not in frame.columns:
        raise KeyError(f"Missing required column: '{dataset_column}'")
    if path_column not in frame.columns:
        raise KeyError(f"Missing required column: '{path_column}'")
    if species_column not in frame.columns:
        raise KeyError(f"Missing required column: '{species_column}'")

    species = frame[species_column].map(_known_species)
    dataset_mask = frame[dataset_column].astype(str).str.strip().eq(dataset_name)
    if not dataset_mask.any():
        return species, 0

    basenames = (
        frame.loc[dataset_mask, path_column]
        .astype(str)
        .map(lambda value: Path(str(value).replace("\\", "/")).name)
    )
    missing_images = sorted(set(basenames) - set(map(str, official.index)))
    if missing_images:
        raise ValueError(
            f"HappyWhale rows missing from official species CSV "
            f"({len(missing_images)}), e.g. {missing_images[:5]}"
        )

    looked_up = official.loc[basenames.to_numpy()]
    if identity_column in frame.columns:
        split_ids = frame.loc[dataset_mask, identity_column].astype(str).str.strip()
        official_ids = looked_up["individual_id"].astype(str).to_numpy()
        id_mismatch = split_ids.to_numpy() != official_ids
        if id_mismatch.any():
            bad = split_ids.to_numpy()[id_mismatch][:5].tolist()
            raise ValueError(
                "HappyWhale identity does not match official individual_id; "
                f"examples: {bad}"
            )

    existing = species.loc[dataset_mask].map(canonicalize_happywhale_species)
    official_species = pd.Series(
        looked_up["species"].to_numpy(), index=existing.index, dtype=object
    )
    both_present = existing.notna()
    conflicts = both_present & (existing != official_species)
    if conflicts.any():
        samples = sorted({
            f"{old}->{new}"
            for old, new in zip(
                existing.loc[conflicts].tolist(),
                official_species.loc[conflicts].tolist(),
            )
        })[:5]
        raise ValueError(
            "HappyWhale species conflicts with official whale/dolphin labels; "
            f"examples: {samples}"
        )

    rows_filled = int((dataset_mask & species.isna()).sum())
    species.loc[dataset_mask] = existing.where(existing.notna(), official_species)
    return species, rows_filled


def _frozen_happywhale_categories():
    if not FROZEN_SPLIT_MANIFEST.is_file():
        return {}
    manifest = pd.read_csv(
        FROZEN_SPLIT_MANIFEST, usecols=["dataset", "identity", "path", "species"], dtype=str
    )
    happywhale = manifest.loc[manifest["dataset"].eq("HappyWhale")].dropna()
    return {
        (row.identity, Path(row.path).name): row.species
        for row in happywhale.itertuples(index=False)
    }


def enrich_species_metadata(
    data,
    *,
    dataset_to_species=None,
    dataset_column="dataset",
    species_column="species",
    path_column="path",
    identity_column="identity",
    happywhale_species_csv=None,
    require_complete=True,
):
    """Fill missing species and normalize whale categories without changing rows.

    HappyWhale categories are harmonized to ``whale`` or ``dolphin``. Missing
    labels use matching identities and image basenames in the frozen split
    manifest. An explicitly supplied official CSV also validates existing labels.
    Other datasets use ``BHCA_DATASET_TO_SPECIES``.
    Existing ``humpback-whale``, ``humpback_whale`` and ``humpback whale``
    labels (case-insensitive) become ``whale`` before conflict validation.
    Identity labels, dataset names, paths and row membership are preserved.
    """
    frame = data.copy() if isinstance(data, pd.DataFrame) else pd.read_csv(data)
    if dataset_column not in frame.columns:
        raise KeyError(
            f"Species enrichment requires the explicit '{dataset_column}' column."
        )
    if species_column not in frame.columns:
        frame[species_column] = pd.NA

    mapping = dict(BHCA_DATASET_TO_SPECIES)
    if dataset_to_species is not None:
        if not isinstance(dataset_to_species, Mapping):
            raise TypeError("dataset_to_species must be a mapping.")
        mapping.update({str(key): str(value) for key, value in dataset_to_species.items()})
    if "HappyWhale" in mapping:
        raise ValueError(
            "HappyWhale must not use a single dataset-level species fill; "
            "pass official per-image labels via happywhale_species_csv instead."
        )
    if not mapping or any(
        not str(key).strip() or not str(value).strip()
        for key, value in mapping.items()
    ):
        raise ValueError("Dataset/species mapping keys and values must be non-empty strings.")
    mapping = {
        key: _canonicalize_whale_category(value)
        for key, value in mapping.items()
    }
    if any(value is None for value in mapping.values()):
        raise ValueError("Dataset/species mapping values must name a known category.")

    original_index = frame.index.copy()
    original_columns = list(frame.columns)
    original_non_species = frame.drop(columns=[species_column]).copy(deep=True)
    original_species = frame[species_column].map(_known_species)
    normalized_species = original_species.map(_canonicalize_whale_category)
    null_before = int(normalized_species.isna().sum())
    rows_by_dataset = {}
    happywhale_filled = 0

    happywhale_mask = frame[dataset_column].astype(str).str.strip().eq("HappyWhale")
    happywhale_missing = happywhale_mask & normalized_species.isna()
    if happywhale_mask.any() and happywhale_species_csv is not None:
        frame[species_column] = normalized_species
        normalized_species, happywhale_filled = fill_happywhale_species_from_official(
            frame,
            happywhale_species_csv,
            dataset_column=dataset_column,
            species_column=species_column,
            path_column=path_column,
            identity_column=identity_column,
        )
        rows_by_dataset["HappyWhale"] = happywhale_filled
    else:
        normalized_species.loc[happywhale_mask] = normalized_species.loc[happywhale_mask].map(
            canonicalize_happywhale_species
        )
        if happywhale_missing.any() and {identity_column, path_column} <= set(frame.columns):
            frozen = _frozen_happywhale_categories()
            requested = frame.loc[happywhale_missing, [identity_column, path_column]]
            labels = [
                frozen.get((str(identity).strip(), Path(str(path).replace("\\", "/")).name))
                for identity, path in requested.itertuples(index=False, name=None)
            ]
            normalized_species.loc[happywhale_missing] = labels
            rows_by_dataset["HappyWhale"] = sum(label is not None for label in labels)

    for dataset, canonical_species in mapping.items():
        canonical_species = str(canonical_species).strip()
        dataset_mask = frame[dataset_column].astype(str).str.strip().eq(str(dataset).strip())
        if not dataset_mask.any():
            rows_by_dataset[str(dataset)] = 0
            continue
        existing = sorted({
            str(value).strip()
            for value in normalized_species[dataset_mask].dropna().tolist()
        })
        conflicts = [
            value for value in existing
            if value.casefold() != canonical_species.casefold()
        ]
        if conflicts:
            raise ValueError(
                f"Dataset {dataset!r} conflicts with approved species "
                f"{canonical_species!r}; existing values: {existing}."
            )
        fill_mask = dataset_mask & normalized_species.isna()
        rows_by_dataset[str(dataset)] = int(fill_mask.sum())
        normalized_species.loc[fill_mask] = canonical_species

    frame[species_column] = normalized_species
    unresolved_mask = frame[species_column].isna()
    unresolved_datasets = sorted(
        frame.loc[unresolved_mask, dataset_column].astype(str).str.strip().unique().tolist()
    )
    null_after = int(unresolved_mask.sum())
    if require_complete and null_after:
        details = ", ".join(
            f"{name}={int(((frame[dataset_column].astype(str).str.strip() == name) & unresolved_mask).sum())}"
            for name in unresolved_datasets
        )
        raise ValueError(
            "BHCA requires complete explicit species metadata after approved enrichment. "
            f"Unresolved rows by dataset: {details}."
        )

    if len(frame) != len(original_non_species) or not frame.index.equals(original_index):
        raise RuntimeError("Species enrichment changed row membership or ordering.")
    if list(frame.columns) != original_columns:
        if species_column in original_columns:
            raise RuntimeError("Species enrichment changed the metadata schema.")
    pd.testing.assert_frame_equal(
        frame.drop(columns=[species_column]),
        original_non_species,
        check_dtype=True,
        check_like=False,
    )
    normalized_mask = original_species.notna() & original_species.ne(normalized_species)
    normalized_by_dataset = (
        frame.loc[normalized_mask, dataset_column]
        .astype(str).str.strip().value_counts().sort_index().to_dict()
    )
    report = SpeciesEnrichmentReport(
        num_rows=int(len(frame)),
        null_species_before=null_before,
        null_species_after=null_after,
        rows_filled=int(null_before - null_after),
        rows_by_dataset=rows_by_dataset,
        canonical_dataset_species=mapping,
        unresolved_datasets=unresolved_datasets,
        happywhale_rows_filled_from_official=happywhale_filled,
        rows_normalized=int(normalized_mask.sum()),
        rows_normalized_by_dataset=normalized_by_dataset,
    )
    return frame, report


def sqrt_inverse_identity_frequency(identity_counts):
    """Return parameter-free inverse-square-root weights normalized to mean 1."""
    counts = np.asarray(identity_counts, dtype=np.float64)
    if counts.ndim != 1 or counts.size == 0:
        raise ValueError("identity_counts must be a non-empty one-dimensional sequence.")
    if np.any(counts <= 0):
        raise ValueError("Every represented species must contain at least one identity.")
    raw = 1.0 / np.sqrt(counts)
    return (raw / raw.mean()).astype(np.float32)


@dataclass(frozen=True)
class TrainingHierarchyMetadata:
    """Deterministic hierarchy state derived from the training split only."""

    class_to_species: list
    species_to_index: dict
    species_names: list
    identities_per_species: list
    species_weights: list
    num_training_identities: int
    num_training_species: int
    known_parent_identities: int
    missing_parent_identities: int
    singleton_species: int
    min_identities_per_species: int
    median_identities_per_species: float
    max_identities_per_species: int
    min_species_weight: float
    mean_species_weight: float
    max_species_weight: float
    species_balance: str = "sqrt_inverse_identity_frequency"

    @property
    def known_parent_percentage(self):
        if self.num_training_identities == 0:
            return 0.0
        return 100.0 * self.known_parent_identities / self.num_training_identities

    def to_dict(self):
        result = asdict(self)
        result["known_parent_percentage"] = self.known_parent_percentage
        return result


def build_training_hierarchy(
    data,
    label_to_idx,
    *,
    identity_column="identity",
    species_column="species",
    missing_index=MISSING_SPECIES_INDEX,
):
    """Build an identity hierarchy exclusively from an explicit training table.

    Missing parents remain masked.  No identity-string or dataset-name inference
    is performed, and conflicting explicit parents for one identity are rejected.
    """
    frame = data.copy() if isinstance(data, pd.DataFrame) else pd.read_csv(data)
    if identity_column not in frame.columns:
        raise KeyError(f"Missing required training column: '{identity_column}'")
    num_classes = len(label_to_idx)
    class_to_name = {int(index): identity for identity, index in label_to_idx.items()}
    if set(class_to_name) != set(range(num_classes)):
        raise ValueError("label_to_idx values must be contiguous in [0, num_classes).")

    identity_parent = {}
    if species_column in frame.columns:
        for identity, group in frame.groupby(identity_column, dropna=False, sort=False):
            if identity not in label_to_idx:
                continue
            parents = sorted({
                parent for parent in (_known_species(value) for value in group[species_column])
                if parent is not None
            })
            if len(parents) > 1:
                raise ValueError(
                    f"Training identity {identity!r} has conflicting species/category values: {parents}."
                )
            identity_parent[identity] = parents[0] if parents else None

    species_names = sorted({parent for parent in identity_parent.values() if parent is not None})
    species_to_index = {name: index for index, name in enumerate(species_names)}
    class_to_species = [int(missing_index)] * num_classes
    for class_index, identity in class_to_name.items():
        parent = identity_parent.get(identity)
        if parent is not None:
            class_to_species[class_index] = species_to_index[parent]

    counts = np.bincount(
        np.asarray([value for value in class_to_species if value != missing_index], dtype=np.int64),
        minlength=len(species_names),
    ).astype(np.int64)
    if counts.size:
        weights = sqrt_inverse_identity_frequency(counts)
        min_count = int(counts.min())
        median_count = float(np.median(counts))
        max_count = int(counts.max())
        min_weight = float(weights.min())
        mean_weight = float(weights.mean())
        max_weight = float(weights.max())
    else:
        weights = np.empty(0, dtype=np.float32)
        min_count = max_count = 0
        median_count = 0.0
        min_weight = mean_weight = max_weight = 0.0

    known = sum(value != missing_index for value in class_to_species)
    metadata = TrainingHierarchyMetadata(
        class_to_species=class_to_species,
        species_to_index=species_to_index,
        species_names=species_names,
        identities_per_species=counts.tolist(),
        species_weights=weights.tolist(),
        num_training_identities=num_classes,
        num_training_species=len(species_names),
        known_parent_identities=int(known),
        missing_parent_identities=int(num_classes - known),
        singleton_species=int(np.sum(counts == 1)) if counts.size else 0,
        min_identities_per_species=min_count,
        median_identities_per_species=median_count,
        max_identities_per_species=max_count,
        min_species_weight=min_weight,
        mean_species_weight=mean_weight,
        max_species_weight=max_weight,
    )
    if metadata.num_training_species and not math.isclose(
        metadata.mean_species_weight, 1.0, rel_tol=1e-6, abs_tol=1e-6
    ):
        raise RuntimeError("Species balancing weights were not normalized to mean 1.")
    return metadata


__all__ = [
    "BHCA_DATASET_TO_SPECIES",
    "MISSING_SPECIES_INDEX",
    "SpeciesEnrichmentReport",
    "TrainingHierarchyMetadata",
    "build_training_hierarchy",
    "canonicalize_happywhale_species",
    "enrich_species_metadata",
    "fill_happywhale_species_from_official",
    "load_happywhale_official_species",
    "sqrt_inverse_identity_frequency",
]
