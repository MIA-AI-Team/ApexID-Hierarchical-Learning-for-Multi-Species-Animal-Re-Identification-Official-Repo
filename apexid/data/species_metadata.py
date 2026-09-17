"""Training-only identity-to-species metadata for hierarchical objectives."""

from dataclasses import asdict, dataclass
import math
from collections.abc import Mapping

import numpy as np
import pandas as pd


MISSING_SPECIES_INDEX = -100

# Explicitly approved dataset-level metadata. Values follow the lower-case
# canonical style already used by open_set_splits_v2/train.csv.
BHCA_DATASET_TO_SPECIES = {
    "CzechLynx": "lynx",
    "SalamanderID2025": "salamander",
    "humpback-whale-identification": "humpback-whale",
    "HappyWhale": "whale",
}


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

    def to_dict(self):
        return asdict(self)


def enrich_species_metadata(
    data,
    *,
    dataset_to_species=None,
    dataset_column="dataset",
    species_column="species",
    require_complete=True,
):
    """Fill missing species from approved dataset rules without changing rows.

    Existing non-null labels are preserved. For a dataset covered by an
    approved rule, any conflicting existing label is an error rather than being
    silently overwritten. When ``require_complete`` is true, unresolved nulls
    prevent BHCA training.
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
    if not mapping or any(
        not str(key).strip() or not str(value).strip()
        for key, value in mapping.items()
    ):
        raise ValueError("Dataset/species mapping keys and values must be non-empty strings.")

    original_index = frame.index.copy()
    original_columns = list(frame.columns)
    original_non_species = frame.drop(columns=[species_column]).copy(deep=True)
    normalized_species = frame[species_column].map(_known_species)
    null_before = int(normalized_species.isna().sum())
    rows_by_dataset = {}

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
        # A missing species column may be appended, but existing tables must
        # retain their exact column order and all non-species fields.
        if species_column in original_columns:
            raise RuntimeError("Species enrichment changed the metadata schema.")
    pd.testing.assert_frame_equal(
        frame.drop(columns=[species_column]),
        original_non_species,
        check_dtype=True,
        check_like=False,
    )
    report = SpeciesEnrichmentReport(
        num_rows=int(len(frame)),
        null_species_before=null_before,
        null_species_after=null_after,
        rows_filled=int(null_before - null_after),
        rows_by_dataset=rows_by_dataset,
        canonical_dataset_species=mapping,
        unresolved_datasets=unresolved_datasets,
    )
    return frame, report


def _known_species(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    if not value or value.lower() in {"nan", "none", "null", "unknown", "-100"}:
        return None
    return value


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
    "enrich_species_metadata",
    "sqrt_inverse_identity_frequency",
]
