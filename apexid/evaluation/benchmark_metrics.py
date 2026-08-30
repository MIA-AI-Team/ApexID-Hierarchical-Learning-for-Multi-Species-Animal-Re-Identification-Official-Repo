"""Label-invariant metrics used by the ApexID benchmark evaluator.

These helpers operate only on predictions and evaluator-owned ground truth;
they deliberately have no model- or threshold-selection dependencies.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Iterable, Mapping

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score


B3_METRIC_VERSION = "standard_item_averaged_v2"


def _finite(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else float("nan")


def _label_array(labels: Iterable[object], name: str) -> np.ndarray:
    """Return a one-dimensional object array of hashable cluster labels."""
    if isinstance(labels, np.ndarray) and labels.ndim != 1:
        raise ValueError(f"{name} must be a 1D sequence")

    values = list(labels)
    result = np.empty(len(values), dtype=object)
    result[:] = values
    for index, value in enumerate(values):
        try:
            hash(value)
        except TypeError as exc:
            raise TypeError(f"{name}[{index}] must be hashable, got {type(value).__name__}") from exc
    return result


def _validate(true_labels: Iterable[object], predicted_labels: Iterable[object]) -> tuple[np.ndarray, np.ndarray]:
    true = _label_array(true_labels, "true_labels")
    predicted = _label_array(predicted_labels, "predicted_labels")
    if true.shape != predicted.shape:
        raise ValueError("true_labels and predicted_labels must be same-length 1D sequences")
    return true, predicted


def b3_metrics(true_labels: Iterable[object], predicted_labels: Iterable[object]) -> dict[str, float]:
    """Return standard item-averaged B³ precision, recall, and F1.

    Empty inputs return NaN values because benchmark scenario aggregation uses
    empty subsets to represent unavailable metrics.
    """
    true, predicted = _validate(true_labels, predicted_labels)
    if true.size == 0:
        return {"b3_precision": float("nan"), "b3_recall": float("nan"), "b3_f1": float("nan")}
    contingency = Counter(zip(true.tolist(), predicted.tolist()))
    true_sizes = Counter(true.tolist())
    predicted_sizes = Counter(predicted.tolist())
    precision = sum(
        count * count / predicted_sizes[predicted_label]
        for (_true_label, predicted_label), count in contingency.items()
    ) / true.size
    recall = sum(
        count * count / true_sizes[true_label]
        for (true_label, _predicted_label), count in contingency.items()
    ) / true.size
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"b3_precision": float(precision), "b3_recall": float(recall), "b3_f1": float(f1)}


def pairwise_metrics(true_labels: Iterable[object], predicted_labels: Iterable[object]) -> dict[str, float]:
    """Return pairwise precision/recall/F1 using contingency counts."""
    true, predicted = _validate(true_labels, predicted_labels)
    if true.size == 0:
        return {"pairwise_precision": float("nan"), "pairwise_recall": float("nan"), "pairwise_f1": float("nan")}
    choose2 = lambda count: count * (count - 1) // 2
    contingency = Counter(zip(true.tolist(), predicted.tolist()))
    true_sizes = Counter(true.tolist())
    predicted_sizes = Counter(predicted.tolist())
    tp = sum(choose2(count) for count in contingency.values())
    predicted_pairs = sum(choose2(count) for count in predicted_sizes.values())
    true_pairs = sum(choose2(count) for count in true_sizes.values())
    precision = tp / predicted_pairs if predicted_pairs else 0.0
    recall = tp / true_pairs if true_pairs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"pairwise_precision": float(precision), "pairwise_recall": float(recall), "pairwise_f1": float(f1)}


def clustering_metrics(true_labels: Iterable[object], predicted_labels: Iterable[object]) -> dict[str, float]:
    """Return the common label-invariant clustering metrics."""
    true, predicted = _validate(true_labels, predicted_labels)
    if true.size == 0:
        base = {"ari": float("nan"), "ami": float("nan"), "nmi": float("nan")}
        return base | b3_metrics([], []) | pairwise_metrics([], [])
    return {
        "ari": _finite(adjusted_rand_score(true, predicted)),
        "ami": _finite(adjusted_mutual_info_score(true, predicted)),
        "nmi": _finite(normalized_mutual_info_score(true, predicted)),
        **b3_metrics(true, predicted),
        **pairwise_metrics(true, predicted),
    }


def fragmentation_metrics(true_labels: Iterable[object], predicted_labels: Iterable[object]) -> dict[str, float]:
    true, predicted = _validate(true_labels, predicted_labels)
    if true.size == 0:
        return {
            "mean_fragments_per_identity": float("nan"), "median_fragments_per_identity": float("nan"),
            "max_fragments_per_identity": float("nan"), "fraction_true_identities_fragmented": float("nan"),
        }
    groups: dict[object, set[object]] = defaultdict(set)
    for target, prediction in zip(true, predicted):
        groups[target].add(prediction)
    fragments = np.asarray([len(values) for values in groups.values()], dtype=float)
    return {
        "mean_fragments_per_identity": float(fragments.mean()),
        "median_fragments_per_identity": float(np.median(fragments)),
        "max_fragments_per_identity": float(fragments.max()),
        "fraction_true_identities_fragmented": float(np.mean(fragments > 1)),
    }


def merge_metrics(true_labels: Iterable[object], predicted_labels: Iterable[object]) -> dict[str, float]:
    true, predicted = _validate(true_labels, predicted_labels)
    if true.size == 0:
        return {
            "mean_true_ids_per_predicted_cluster": float("nan"), "max_true_ids_per_predicted_cluster": float("nan"),
            "fraction_predicted_clusters_merged": float("nan"),
        }
    groups: dict[object, set[object]] = defaultdict(set)
    for target, prediction in zip(true, predicted):
        groups[prediction].add(target)
    merged = np.asarray([len(values) for values in groups.values()], dtype=float)
    return {
        "mean_true_ids_per_predicted_cluster": float(merged.mean()),
        "max_true_ids_per_predicted_cluster": float(merged.max()),
        "fraction_predicted_clusters_merged": float(np.mean(merged > 1)),
    }


def cluster_count_error(true_labels: Iterable[object], predicted_labels: Iterable[object]) -> dict[str, float]:
    true, predicted = _validate(true_labels, predicted_labels)
    true_count = len(set(true.tolist()))
    predicted_count = len(set(predicted.tolist()))
    error = abs(predicted_count - true_count)
    return {
        "true_identity_count": float(true_count),
        "predicted_cluster_count": float(predicted_count),
        "absolute_cluster_count_error": float(error),
        "relative_cluster_count_error": float(error / true_count) if true_count else float("nan"),
    }


def discovery_metrics(true_labels: Iterable[object], predicted_labels: Iterable[object]) -> dict[str, float]:
    """Aggregate all cluster-structure diagnostics for an unknown subset."""
    return clustering_metrics(true_labels, predicted_labels) | fragmentation_metrics(true_labels, predicted_labels) | merge_metrics(true_labels, predicted_labels) | cluster_count_error(true_labels, predicted_labels)


__all__ = [
    "B3_METRIC_VERSION", "b3_metrics", "cluster_count_error", "clustering_metrics", "discovery_metrics",
    "fragmentation_metrics", "merge_metrics", "pairwise_metrics",
]
