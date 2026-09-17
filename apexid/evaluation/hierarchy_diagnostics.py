"""Validation-only representation diagnostics for species/identity hierarchy."""

import math

import numpy as np


def _as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _known_species_mask(species):
    normalized = np.asarray([str(value).strip().lower() for value in species], dtype=object)
    return ~np.isin(normalized, ["", "nan", "none", "null", "unknown", "-100"])


def _pair_cosines(embeddings, pairs):
    if not pairs:
        return np.empty(0, dtype=np.float64)
    pairs = np.asarray(pairs, dtype=np.int64)
    return np.sum(embeddings[pairs[:, 0]] * embeddings[pairs[:, 1]], axis=1)


def _sample_hierarchy_pairs(labels, species, max_pairs, rng):
    identity_groups = {
        label: np.flatnonzero(labels == label) for label in np.unique(labels)
    }
    species_groups = {
        parent: np.flatnonzero(species == parent) for parent in np.unique(species)
    }
    identities_by_species = {}
    for parent, indices in species_groups.items():
        identities_by_species[parent] = [
            identity_groups[label]
            for label in np.unique(labels[indices])
            if identity_groups[label].size
        ]

    same_id_groups = [indices for indices in identity_groups.values() if indices.size >= 2]
    sibling_species = [
        parent for parent, groups in identities_by_species.items() if len(groups) >= 2
    ]
    species_values = list(species_groups)
    same_id = []
    same_species = []
    different_species = []
    for _ in range(int(max_pairs)):
        if same_id_groups:
            group = same_id_groups[int(rng.integers(len(same_id_groups)))]
            pair = rng.choice(group, size=2, replace=False)
            same_id.append((int(pair[0]), int(pair[1])))
        if sibling_species:
            parent = sibling_species[int(rng.integers(len(sibling_species)))]
            groups = identities_by_species[parent]
            selected = rng.choice(len(groups), size=2, replace=False)
            same_species.append((
                int(groups[int(selected[0])][int(rng.integers(groups[int(selected[0])].size))]),
                int(groups[int(selected[1])][int(rng.integers(groups[int(selected[1])].size))]),
            ))
        if len(species_values) >= 2:
            selected = rng.choice(len(species_values), size=2, replace=False)
            first = species_groups[species_values[int(selected[0])]]
            second = species_groups[species_values[int(selected[1])]]
            different_species.append((
                int(first[int(rng.integers(first.size))]),
                int(second[int(rng.integers(second.size))]),
            ))
    return same_id, same_species, different_species


def _distribution_metrics(prefix, values):
    values = np.asarray(values, dtype=np.float64)
    result = {f"{prefix}_count": int(values.size)}
    for name in ("mean", "std", "p05", "median", "p95"):
        result[f"{prefix}_{name}"] = float("nan")
    if values.size:
        result.update({
            f"{prefix}_mean": float(np.mean(values)),
            f"{prefix}_std": float(np.std(values)),
            f"{prefix}_p05": float(np.quantile(values, 0.05)),
            f"{prefix}_median": float(np.median(values)),
            f"{prefix}_p95": float(np.quantile(values, 0.95)),
        })
    return result


def compute_hierarchical_similarity_diagnostics(
    embeddings,
    identity_labels,
    species_labels,
    *,
    max_pairs=100000,
    chunk_size=1024,
    random_state=42,
):
    """Measure same-ID, sibling-ID, and cross-species cosine distributions.

    This function is evaluation-only.  It never modifies a loss, proxy, sample
    weight, or training target.
    """
    embeddings = _as_numpy(embeddings).astype(np.float64, copy=False)
    identities = _as_numpy(identity_labels).reshape(-1)
    species = _as_numpy(species_labels).reshape(-1)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [num_samples, embedding_dim].")
    if identities.size != embeddings.shape[0] or species.size != embeddings.shape[0]:
        raise ValueError("identity_labels and species_labels must align with embeddings.")
    if int(max_pairs) <= 0 or int(chunk_size) <= 0:
        raise ValueError("max_pairs and chunk_size must be positive.")

    known = _known_species_mask(species)
    embeddings = embeddings[known]
    identities = identities[known]
    species = species[known].astype(str)
    if embeddings.shape[0]:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, 1e-12, None)

    rng = np.random.default_rng(int(random_state))
    pairs = _sample_hierarchy_pairs(identities, species, int(max_pairs), rng)
    same_id = _pair_cosines(embeddings, pairs[0])
    sibling = _pair_cosines(embeddings, pairs[1])
    cross_species = _pair_cosines(embeddings, pairs[2])
    result = {}
    result.update(_distribution_metrics("hierarchy_diag/same_identity", same_id))
    result.update(_distribution_metrics("hierarchy_diag/different_identity_same_species", sibling))
    result.update(_distribution_metrics("hierarchy_diag/different_species", cross_species))

    same_mean = result["hierarchy_diag/same_identity_mean"]
    sibling_mean = result["hierarchy_diag/different_identity_same_species_mean"]
    cross_mean = result["hierarchy_diag/different_species_mean"]
    result["hierarchy_diag/delta_identity"] = (
        float(same_mean - sibling_mean)
        if math.isfinite(same_mean) and math.isfinite(sibling_mean)
        else float("nan")
    )
    result["hierarchy_diag/delta_species"] = (
        float(sibling_mean - cross_mean)
        if math.isfinite(sibling_mean) and math.isfinite(cross_mean)
        else float("nan")
    )

    hardest_siblings = []
    hardest_positives = []
    hardest_gaps = []
    num_samples = embeddings.shape[0]
    for start in range(0, num_samples, int(chunk_size)):
        stop = min(start + int(chunk_size), num_samples)
        similarities = embeddings[start:stop] @ embeddings.T
        for local_index, global_index in enumerate(range(start, stop)):
            same_identity_mask = identities == identities[global_index]
            same_identity_mask[global_index] = False
            sibling_mask = (species == species[global_index]) & (
                identities != identities[global_index]
            )
            if not np.any(same_identity_mask) or not np.any(sibling_mask):
                continue
            positive = float(np.max(similarities[local_index, same_identity_mask]))
            negative = float(np.max(similarities[local_index, sibling_mask]))
            hardest_positives.append(positive)
            hardest_siblings.append(negative)
            hardest_gaps.append(positive - negative)

    result.update(_distribution_metrics(
        "hierarchy_diag/hardest_same_species_negative", hardest_siblings
    ))
    result.update(_distribution_metrics(
        "hierarchy_diag/hardest_positive", hardest_positives
    ))
    result.update(_distribution_metrics(
        "hierarchy_diag/hardest_same_species_gap", hardest_gaps
    ))
    result["hierarchy_diag/hardest_same_species_violation_rate"] = (
        float(np.mean(np.asarray(hardest_gaps) <= 0))
        if hardest_gaps else float("nan")
    )
    result["hierarchy_diag/known_species_samples"] = int(num_samples)
    return result


__all__ = ["compute_hierarchical_similarity_diagnostics"]
