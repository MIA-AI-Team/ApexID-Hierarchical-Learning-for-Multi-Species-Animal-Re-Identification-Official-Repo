import math

import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    auc,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    v_measure_score,
)

from .reid_metrics import _normalize_embeddings, _to_numpy


def _as_1d_array(values, name):
    values = _to_numpy(values)
    if values.ndim != 1:
        raise ValueError(f"{name} must have shape (num_samples,).")
    return values


def _as_bool_mask(values, name):
    values = np.asarray(values, dtype=bool)
    if values.ndim != 1:
        raise ValueError(f"{name} must have shape (num_samples,).")
    return values


def _validate_embedding_labels(embeddings, labels, name="labels"):
    embeddings = _normalize_embeddings(embeddings).astype(np.float32, copy=False)
    labels = _as_1d_array(labels, name)
    if labels.shape[0] != embeddings.shape[0]:
        raise ValueError(
            f"{name} length must match embeddings. Expected {embeddings.shape[0]}, got {labels.shape[0]}."
        )
    return embeddings, labels


def _metric_nan():
    return float("nan")


def _safe_mean(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return _metric_nan()
    return float(np.mean(values))


def _balanced_mean_by_label(labels, values):
    labels = _as_1d_array(labels, "labels")
    values = np.asarray(values, dtype=np.float64)
    if labels.shape[0] != values.shape[0]:
        raise ValueError("labels and values must have the same length.")
    if labels.size == 0:
        return _metric_nan()

    per_label = []
    for label in np.unique(labels):
        mask = labels == label
        if np.any(mask):
            per_label.append(float(np.mean(values[mask])))
    return _safe_mean(per_label)


def _chunk_size(chunk_size, num_samples):
    if chunk_size is None:
        return max(int(num_samples), 1)
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0.")
    return chunk_size


def _optional_sample_limit(value, name):
    if value in {None, "", "null"}:
        return None
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0 when provided.")
    return value


def _sample_limits_from_config(config):
    sample_cfg = dict(config.get("sample_limits", {}) or {})
    aliases = {
        "gallery": ("gallery", "max_gallery_samples"),
        "known": ("known", "max_known_samples"),
        "unknown": ("unknown", "max_unknown_samples"),
        "mixed": ("mixed", "max_mixed_samples"),
    }

    limits = {}
    for split_name, keys in aliases.items():
        value = None
        for key in keys:
            if key in sample_cfg:
                value = sample_cfg[key]
                break
            if key in config:
                value = config[key]
                break
        limits[split_name] = _optional_sample_limit(value, f"evaluation.sample_limits.{split_name}")
    return limits


def _balanced_sample_indices(labels, max_samples, rng):
    labels = _as_1d_array(labels, "labels")
    num_samples = int(labels.shape[0])
    if max_samples is None or max_samples >= num_samples:
        return np.arange(num_samples, dtype=np.int64)

    max_samples = int(max_samples)
    unique_labels = np.unique(labels)
    selected = []

    if unique_labels.size > max_samples:
        unique_labels = rng.choice(unique_labels, size=max_samples, replace=False)

    for label in unique_labels:
        label_indices = np.flatnonzero(labels == label)
        if label_indices.size:
            selected.append(int(rng.choice(label_indices)))

    selected = np.asarray(selected, dtype=np.int64)
    remaining_count = max_samples - int(selected.size)
    if remaining_count > 0:
        mask = np.ones(num_samples, dtype=bool)
        mask[selected] = False
        remaining = np.flatnonzero(mask)
        if remaining.size:
            fill = rng.choice(remaining, size=min(remaining_count, remaining.size), replace=False)
            selected = np.concatenate([selected, fill.astype(np.int64, copy=False)])

    selected = np.unique(selected.astype(np.int64, copy=False))
    if selected.size > max_samples:
        selected = rng.choice(selected, size=max_samples, replace=False)
    return np.sort(selected.astype(np.int64, copy=False))


def _gallery_sample_indices(labels, max_samples, required_labels, rng):
    labels = _as_1d_array(labels, "gallery_labels")
    num_samples = int(labels.shape[0])
    if max_samples is None or max_samples >= num_samples:
        return np.arange(num_samples, dtype=np.int64)

    max_samples = int(max_samples)
    selected = []
    required_labels = np.unique(np.asarray(required_labels, dtype=labels.dtype))
    for label in required_labels:
        label_indices = np.flatnonzero(labels == label)
        if label_indices.size:
            selected.append(int(rng.choice(label_indices)))

    selected = np.unique(np.asarray(selected, dtype=np.int64))
    if selected.size > max_samples:
        selected = rng.choice(selected, size=max_samples, replace=False)

    remaining_count = max_samples - int(selected.size)
    if remaining_count > 0:
        mask = np.ones(num_samples, dtype=bool)
        mask[selected] = False
        remaining = np.flatnonzero(mask)
        if remaining.size:
            extra = _balanced_sample_indices(labels[remaining], remaining_count, rng)
            selected = np.concatenate([selected, remaining[extra]])

    return np.sort(np.unique(selected.astype(np.int64, copy=False)))


def _take_optional_1d(values, indices, name):
    if values is None:
        return None
    values = _as_1d_array(values, name)
    return values[indices]


def _is_concatenated_split(
    mixed_embeddings,
    mixed_labels,
    known_embeddings,
    known_labels,
    unknown_embeddings,
    unknown_labels,
):
    if mixed_labels.shape[0] != known_labels.shape[0] + unknown_labels.shape[0]:
        return False
    if mixed_embeddings.shape[0] != known_embeddings.shape[0] + unknown_embeddings.shape[0]:
        return False
    expected_labels = np.concatenate([known_labels, unknown_labels])
    if not np.array_equal(mixed_labels, expected_labels):
        return False
    expected_embeddings = np.concatenate([known_embeddings, unknown_embeddings], axis=0)
    return np.array_equal(mixed_embeddings, expected_embeddings)


def best_gallery_matches(query_embeddings, gallery_embeddings, gallery_labels, chunk_size=1024):
    """Return best cosine score and gallery identity for each query embedding."""
    query_embeddings = _normalize_embeddings(query_embeddings).astype(np.float32, copy=False)
    gallery_embeddings, gallery_labels = _validate_embedding_labels(
        gallery_embeddings,
        gallery_labels,
        name="gallery_labels",
    )
    if query_embeddings.ndim != 2:
        raise ValueError("query_embeddings must have shape (num_samples, embedding_dim).")
    if gallery_embeddings.shape[0] == 0:
        return (
            np.full(query_embeddings.shape[0], -np.inf, dtype=np.float32),
            np.full(query_embeddings.shape[0], -1, dtype=np.int64),
        )
    if query_embeddings.shape[1] != gallery_embeddings.shape[1]:
        raise ValueError("query and gallery embeddings must have the same embedding dimension.")

    scores = np.empty(query_embeddings.shape[0], dtype=np.float32)
    labels = np.empty(query_embeddings.shape[0], dtype=gallery_labels.dtype)
    chunk_size = _chunk_size(chunk_size, query_embeddings.shape[0])
    for start in range(0, query_embeddings.shape[0], chunk_size):
        stop = min(start + chunk_size, query_embeddings.shape[0])
        similarity = query_embeddings[start:stop] @ gallery_embeddings.T
        best_indices = np.argmax(similarity, axis=1)
        scores[start:stop] = similarity[np.arange(stop - start), best_indices]
        labels[start:stop] = gallery_labels[best_indices]
    return scores, labels


def _average_precision_from_relevance(relevance):
    hits = np.flatnonzero(relevance)
    if hits.size == 0:
        return _metric_nan()
    precision_at_hits = np.arange(1, hits.size + 1, dtype=np.float64) / (hits.astype(np.float64) + 1.0)
    return float(np.mean(precision_at_hits))


def compute_known_retrieval_metrics(
    query_embeddings,
    query_labels,
    gallery_embeddings,
    gallery_labels,
    top_k=(1, 5, 10),
    cmc_ranks=(20, 50),
    chunk_size=1024,
    compute_map=True,
    return_matches=False,
):
    """Compute known-identity retrieval metrics using train embeddings as gallery."""
    query_embeddings, query_labels = _validate_embedding_labels(
        query_embeddings,
        query_labels,
        name="query_labels",
    )
    gallery_embeddings, gallery_labels = _validate_embedding_labels(
        gallery_embeddings,
        gallery_labels,
        name="gallery_labels",
    )
    if query_embeddings.shape[1:] != gallery_embeddings.shape[1:]:
        raise ValueError("query and gallery embeddings must have the same embedding dimension.")

    top_k = tuple(sorted({int(k) for k in top_k}))
    cmc_ranks = tuple(sorted({int(k) for k in cmc_ranks} - set(top_k)))
    ranks = tuple(sorted(set(top_k) | set(cmc_ranks)))
    if any(k <= 0 for k in ranks):
        raise ValueError("rank values must be > 0.")

    result = {
        "known/mAP": _metric_nan(),
        "known/BAKS": _metric_nan(),
    }
    for k in top_k:
        result[f"known/top{k}"] = _metric_nan()
    for k in cmc_ranks:
        result[f"known/cmc_rank{k}"] = _metric_nan()

    num_queries = int(query_labels.shape[0])
    num_gallery = int(gallery_labels.shape[0])
    if return_matches:
        best_scores = np.full(num_queries, -np.inf, dtype=np.float32)
        best_labels = np.full(num_queries, -1, dtype=gallery_labels.dtype)
    if num_queries == 0 or num_gallery == 0:
        if return_matches:
            return result, best_scores, best_labels
        return result

    chunk_size = _chunk_size(chunk_size, num_queries)
    max_rank = max(ranks) if ranks else 0
    top_hits = {k: 0 for k in ranks}
    valid_queries = 0
    ap_values = []
    top1_correct = np.zeros(num_queries, dtype=np.float64)

    gallery_positive_indices = {
        label: np.flatnonzero(gallery_labels == label)
        for label in np.unique(query_labels)
    }
    gallery_positive_counts = {label: int(indices.size) for label, indices in gallery_positive_indices.items()}

    for start in range(0, num_queries, chunk_size):
        stop = min(start + chunk_size, num_queries)
        labels_chunk = query_labels[start:stop]
        similarity = query_embeddings[start:stop] @ gallery_embeddings.T

        if return_matches:
            best_indices = np.argmax(similarity, axis=1)
            best_scores[start:stop] = similarity[np.arange(stop - start), best_indices]
            best_labels[start:stop] = gallery_labels[best_indices]

        has_positive = np.array(
            [gallery_positive_counts.get(label, 0) > 0 for label in labels_chunk],
            dtype=bool,
        )
        if not np.any(has_positive):
            continue

        valid_queries += int(np.sum(has_positive))

        if max_rank > 0:
            rank_limit = min(max_rank, num_gallery)
            if rank_limit == num_gallery:
                top_indices = np.argsort(-similarity, axis=1)[:, :rank_limit]
            else:
                top_indices = np.argpartition(
                    -similarity,
                    kth=rank_limit - 1,
                    axis=1,
                )[:, :rank_limit]
                top_scores = np.take_along_axis(similarity, top_indices, axis=1)
                top_order = np.argsort(-top_scores, axis=1)
                top_indices = np.take_along_axis(top_indices, top_order, axis=1)
            top_relevance = gallery_labels[top_indices] == labels_chunk[:, None]
            top1_correct[start:stop] = top_relevance[:, :1].any(axis=1).astype(np.float64)
            for k in ranks:
                k_eff = min(k, rank_limit)
                top_hits[k] += int(top_relevance[has_positive, :k_eff].any(axis=1).sum())

        if compute_map:
            valid_local_indices = np.flatnonzero(has_positive)
            sorted_scores = -np.sort(-similarity[valid_local_indices], axis=1)
            for sorted_row_index, row_offset in enumerate(valid_local_indices):
                positive_indices = gallery_positive_indices[labels_chunk[row_offset]]
                positive_scores = similarity[row_offset, positive_indices]
                positive_order = np.argsort(-positive_scores)
                positive_scores = positive_scores[positive_order]
                positive_ranks = np.searchsorted(
                    -sorted_scores[sorted_row_index],
                    -positive_scores,
                    side="left",
                )
                if positive_ranks.size > 1:
                    tie_offsets = np.zeros(positive_ranks.shape[0], dtype=np.int64)
                    tie_start = 0
                    while tie_start < positive_scores.size:
                        tie_stop = tie_start + 1
                        while (
                            tie_stop < positive_scores.size
                            and positive_scores[tie_stop] == positive_scores[tie_start]
                        ):
                            tie_stop += 1
                        tie_offsets[tie_start:tie_stop] = np.arange(tie_stop - tie_start, dtype=np.int64)
                        tie_start = tie_stop
                    positive_ranks = positive_ranks + tie_offsets
                positive_ranks = positive_ranks + 1
                positive_ranks.sort()
                precision_at_hits = (
                    np.arange(1, positive_ranks.size + 1, dtype=np.float64)
                    / positive_ranks.astype(np.float64)
                )
                ap_values.append(float(np.mean(precision_at_hits)))

    if valid_queries == 0:
        if return_matches:
            return result, best_scores, best_labels
        return result

    for k in top_k:
        result[f"known/top{k}"] = float(top_hits[k] / valid_queries)
    for k in cmc_ranks:
        result[f"known/cmc_rank{k}"] = float(top_hits[k] / valid_queries)
    if compute_map:
        result["known/mAP"] = _safe_mean(ap_values)
    result["known/BAKS"] = _balanced_mean_by_label(query_labels, top1_correct)
    if return_matches:
        return result, best_scores, best_labels
    return result


class _UnionFind:
    def __init__(self, size):
        self.parent = np.arange(int(size), dtype=np.int64)
        self.rank = np.zeros(int(size), dtype=np.int8)

    def find(self, x):
        x = int(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = int(self.parent[x])
        return x

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b
        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1


def threshold_connected_components(embeddings, threshold=0.5, chunk_size=1024):
    """Cluster embeddings by cosine-threshold connected components."""
    embeddings = _normalize_embeddings(embeddings).astype(np.float32, copy=False)
    num_samples = int(embeddings.shape[0])
    if num_samples == 0:
        return np.empty(0, dtype=np.int64)
    if num_samples == 1:
        return np.zeros(1, dtype=np.int64)

    threshold = float(threshold)
    chunk_size = _chunk_size(chunk_size, num_samples)
    union_find = _UnionFind(num_samples)

    for start in range(0, num_samples, chunk_size):
        stop = min(start + chunk_size, num_samples)
        similarity = embeddings[start:stop] @ embeddings.T
        for row_offset in range(stop - start):
            row_index = start + row_offset
            matches = np.flatnonzero(similarity[row_offset] >= threshold)
            matches = matches[matches > row_index]
            for match in matches:
                union_find.union(row_index, int(match))

    roots = np.array([union_find.find(i) for i in range(num_samples)], dtype=np.int64)
    _, cluster_ids = np.unique(roots, return_inverse=True)
    return cluster_ids.astype(np.int64, copy=False)


def cluster_embeddings(embeddings, method="threshold_connected_components", threshold=0.5, chunk_size=1024):
    method = str(method or "threshold_connected_components").lower()
    if method in {"threshold_connected_components", "connected_components", "threshold"}:
        return threshold_connected_components(embeddings, threshold=threshold, chunk_size=chunk_size)
    raise ValueError(
        "Unknown evaluation.clustering.method '{}'. Available: threshold_connected_components.".format(method)
    )


def compute_unknown_clustering_metrics(
    embeddings,
    labels,
    method="threshold_connected_components",
    threshold=0.5,
    chunk_size=1024,
    knownness_scores=None,
    known_unknown_threshold=None,
):
    """Compute clustering and balanced unknown-rejection metrics for unknown validation."""
    embeddings, labels = _validate_embedding_labels(embeddings, labels)
    result = {
        "unknown/ARI": _metric_nan(),
        "unknown/AMI": _metric_nan(),
        "unknown/v_measure": _metric_nan(),
        "unknown/BAUS": _metric_nan(),
    }
    if labels.size == 0:
        return result

    clusters = cluster_embeddings(
        embeddings,
        method=method,
        threshold=threshold,
        chunk_size=chunk_size,
    )
    result["unknown/ARI"] = float(adjusted_rand_score(labels, clusters))
    result["unknown/AMI"] = float(adjusted_mutual_info_score(labels, clusters))
    result["unknown/v_measure"] = float(v_measure_score(labels, clusters))

    if knownness_scores is None or known_unknown_threshold is None:
        rejected = np.ones(labels.shape[0], dtype=bool)
    else:
        knownness_scores = np.asarray(knownness_scores, dtype=np.float64)
        if knownness_scores.shape[0] != labels.shape[0]:
            raise ValueError("knownness_scores length must match labels.")
        rejected = knownness_scores < float(known_unknown_threshold)
    result["unknown/BAUS"] = _balanced_mean_by_label(labels, rejected.astype(np.float64))
    return result


def threshold_at_target_tpr(known_scores, target_tpr=0.95):
    known_scores = np.asarray(known_scores, dtype=np.float64)
    known_scores = known_scores[np.isfinite(known_scores)]
    if known_scores.size == 0:
        return _metric_nan(), _metric_nan()
    target_tpr = float(target_tpr)
    if not 0.0 < target_tpr <= 1.0:
        raise ValueError("target_tpr must be in the interval (0, 1].")

    sorted_scores = np.sort(known_scores)[::-1]
    index = min(max(int(math.ceil(target_tpr * sorted_scores.size)) - 1, 0), sorted_scores.size - 1)
    threshold = float(sorted_scores[index])
    achieved_tpr = float(np.mean(known_scores >= threshold))
    return threshold, achieved_tpr


def compute_rejection_metrics(
    known_scores,
    unknown_scores,
    threshold=None,
    target_tpr=0.95,
):
    """Compute known-vs-unknown separation and false-accept metrics."""
    known_scores = np.asarray(known_scores, dtype=np.float64)
    unknown_scores = np.asarray(unknown_scores, dtype=np.float64)
    result = {
        "rejection/AUROC": _metric_nan(),
        "rejection/FAR": _metric_nan(),
        "rejection/FPR95": _metric_nan(),
        "rejection/threshold_at_95_tpr": _metric_nan(),
        "rejection/known_tpr_at_threshold": _metric_nan(),
        "rejection/unknown_recall_at_threshold": _metric_nan(),
    }
    if known_scores.size == 0 or unknown_scores.size == 0:
        return result

    y_true = np.concatenate(
        [
            np.ones(known_scores.shape[0], dtype=np.int64),
            np.zeros(unknown_scores.shape[0], dtype=np.int64),
        ]
    )
    y_score = np.concatenate([known_scores, unknown_scores])
    try:
        result["rejection/AUROC"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        result["rejection/AUROC"] = _metric_nan()

    threshold_95, tpr_95 = threshold_at_target_tpr(known_scores, target_tpr=target_tpr)
    result["rejection/threshold_at_95_tpr"] = float(threshold_95)
    result["rejection/FPR95"] = float(np.mean(unknown_scores >= threshold_95)) if math.isfinite(threshold_95) else _metric_nan()

    if threshold is None:
        threshold = threshold_95
    threshold = float(threshold)
    result["rejection/FAR"] = float(np.mean(unknown_scores >= threshold))
    result["rejection/known_tpr_at_threshold"] = float(np.mean(known_scores >= threshold))
    result["rejection/unknown_recall_at_threshold"] = float(np.mean(unknown_scores < threshold))
    if target_tpr == 0.95:
        result["rejection/known_tpr_at_95_threshold"] = float(tpr_95)
    return result


def _balanced_known_accuracy(true_known_mask, true_labels, pred_labels, accept_mask):
    true_known_mask = _as_bool_mask(true_known_mask, "true_known_mask")
    accept_mask = _as_bool_mask(accept_mask, "accept_mask")
    true_labels = _as_1d_array(true_labels, "true_labels")
    pred_labels = _as_1d_array(pred_labels, "pred_labels")
    if not np.any(true_known_mask):
        return _metric_nan()
    correct = (pred_labels == true_labels) & accept_mask
    return _balanced_mean_by_label(true_labels[true_known_mask], correct[true_known_mask].astype(np.float64))


def _balanced_unknown_accuracy(true_known_mask, true_labels, accept_mask):
    true_known_mask = _as_bool_mask(true_known_mask, "true_known_mask")
    accept_mask = _as_bool_mask(accept_mask, "accept_mask")
    true_labels = _as_1d_array(true_labels, "true_labels")
    unknown_mask = ~true_known_mask
    if not np.any(unknown_mask):
        return _metric_nan()
    rejected = ~accept_mask
    return _balanced_mean_by_label(true_labels[unknown_mask], rejected[unknown_mask].astype(np.float64))


def norm_accuracy(baks, baus):
    if not math.isfinite(float(baks)) or not math.isfinite(float(baus)):
        return _metric_nan()
    return float(math.sqrt(max(float(baks), 0.0) * max(float(baus), 0.0)))


def _candidate_thresholds(scores):
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return np.array([], dtype=np.float64)
    unique = np.unique(scores)
    lower = np.nextafter(float(unique.min()), -np.inf)
    upper = np.nextafter(float(unique.max()), np.inf)
    return np.concatenate([[lower], unique, [upper]]).astype(np.float64)


def select_threshold_by_normacc(scores, true_known_mask, true_labels, pred_known_labels):
    """Select the highest threshold among those maximizing mixed NormAcc."""
    scores = np.asarray(scores, dtype=np.float64)
    true_known_mask = _as_bool_mask(true_known_mask, "true_known_mask")
    true_labels = _as_1d_array(true_labels, "true_labels")
    pred_known_labels = _as_1d_array(pred_known_labels, "pred_known_labels")
    if not (scores.shape[0] == true_known_mask.shape[0] == true_labels.shape[0] == pred_known_labels.shape[0]):
        raise ValueError("scores, masks, true labels, and predicted labels must have the same length.")

    thresholds = _candidate_thresholds(scores)
    if thresholds.size == 0:
        return _metric_nan()

    known_positions = np.flatnonzero(true_known_mask)
    unknown_positions = np.flatnonzero(~true_known_mask)
    known_inverse = np.empty(0, dtype=np.int64)
    unknown_inverse = np.empty(0, dtype=np.int64)
    known_counts = np.empty(0, dtype=np.float64)
    unknown_counts = np.empty(0, dtype=np.float64)
    known_inverse_by_position = np.full(scores.shape[0], -1, dtype=np.int64)
    unknown_inverse_by_position = np.full(scores.shape[0], -1, dtype=np.int64)

    if known_positions.size:
        _, known_inverse, known_counts = np.unique(
            true_labels[known_positions],
            return_inverse=True,
            return_counts=True,
        )
        known_counts = known_counts.astype(np.float64)
        known_inverse_by_position[known_positions] = known_inverse
    if unknown_positions.size:
        _, unknown_inverse, unknown_counts = np.unique(
            true_labels[unknown_positions],
            return_inverse=True,
            return_counts=True,
        )
        unknown_counts = unknown_counts.astype(np.float64)
        unknown_inverse_by_position[unknown_positions] = unknown_inverse

    known_correct_accepted = np.zeros(known_counts.shape[0], dtype=np.float64)
    unknown_accepted = np.zeros(unknown_counts.shape[0], dtype=np.float64)

    def score_from_counts():
        if known_counts.size == 0 or unknown_counts.size == 0:
            return _metric_nan()
        baks = float(np.mean(known_correct_accepted / known_counts))
        baus = float(np.mean((unknown_counts - unknown_accepted) / unknown_counts))
        return norm_accuracy(baks, baus)

    best_threshold = float(np.nextafter(np.max(scores[np.isfinite(scores)]), np.inf))
    best_score = -np.inf

    score = score_from_counts()
    score_for_selection = -np.inf if not math.isfinite(score) else score
    if score_for_selection > best_score:
        best_score = score_for_selection

    finite_indices = np.flatnonzero(np.isfinite(scores))
    order = finite_indices[np.argsort(-scores[finite_indices])]
    cursor = 0
    while cursor < order.size:
        threshold = scores[order[cursor]]
        next_cursor = cursor + 1
        while next_cursor < order.size and scores[order[next_cursor]] == threshold:
            next_cursor += 1
        indices = order[cursor:next_cursor]
        known_group = indices[true_known_mask[indices]]
        if known_group.size:
            correct_group = known_group[pred_known_labels[known_group] == true_labels[known_group]]
            if correct_group.size:
                np.add.at(known_correct_accepted, known_inverse_by_position[correct_group], 1.0)
        unknown_group = indices[~true_known_mask[indices]]
        if unknown_group.size:
            np.add.at(unknown_accepted, unknown_inverse_by_position[unknown_group], 1.0)

        score = score_from_counts()
        score_for_selection = -np.inf if not math.isfinite(score) else score
        if score_for_selection > best_score or (
            score_for_selection == best_score and float(threshold) > best_threshold
        ):
            best_score = score_for_selection
            best_threshold = float(threshold)
        cursor = next_cursor
    return best_threshold


def _known_unknown_acceptance_curve(known_scores, unknown_scores, known_correct):
    known_scores = np.asarray(known_scores, dtype=np.float64)
    unknown_scores = np.asarray(unknown_scores, dtype=np.float64)
    known_correct = np.asarray(known_correct, dtype=bool)
    if known_scores.size == 0 or unknown_scores.size == 0:
        return None, None
    if known_correct.shape[0] != known_scores.shape[0]:
        raise ValueError("known_correct length must match known_scores.")

    scores = np.concatenate([known_scores, unknown_scores])
    is_known = np.concatenate(
        [
            np.ones(known_scores.shape[0], dtype=bool),
            np.zeros(unknown_scores.shape[0], dtype=bool),
        ]
    )
    correct = np.concatenate([known_correct, np.zeros(unknown_scores.shape[0], dtype=bool)])
    finite_indices = np.flatnonzero(np.isfinite(scores))
    if finite_indices.size == 0:
        return None, None

    fpr_values = []
    ccr_values = []
    unknown_accepted = 0
    known_correct_accepted = 0
    fpr_values.append(0.0)
    ccr_values.append(0.0)

    order = finite_indices[np.argsort(-scores[finite_indices])]
    cursor = 0
    while cursor < order.size:
        threshold = scores[order[cursor]]
        next_cursor = cursor + 1
        while next_cursor < order.size and scores[order[next_cursor]] == threshold:
            next_cursor += 1
        indices = order[cursor:next_cursor]
        unknown_accepted += int(np.sum(~is_known[indices]))
        known_correct_accepted += int(np.sum(is_known[indices] & correct[indices]))
        fpr_values.append(float(unknown_accepted / unknown_scores.shape[0]))
        ccr_values.append(float(known_correct_accepted / known_scores.shape[0]))
        cursor = next_cursor

    return np.asarray(fpr_values, dtype=np.float64), np.asarray(ccr_values, dtype=np.float64)


def compute_oscr_auc(known_scores, unknown_scores, known_correct):
    curve = _known_unknown_acceptance_curve(known_scores, unknown_scores, known_correct)
    fpr_values, ccr_values = curve
    if fpr_values is None:
        return _metric_nan()

    order = np.argsort(fpr_values)
    fpr_values = fpr_values[order]
    ccr_values = ccr_values[order]
    unique_fpr = np.unique(fpr_values)
    max_ccr = np.array(
        [np.max(ccr_values[fpr_values == fpr]) for fpr in unique_fpr],
        dtype=np.float64,
    )
    return float(auc(unique_fpr, max_ccr))


def compute_dir_at_far(known_scores, unknown_scores, known_correct, far_level=0.01):
    curve = _known_unknown_acceptance_curve(known_scores, unknown_scores, known_correct)
    fpr_values, ccr_values = curve
    if fpr_values is None:
        return _metric_nan()
    valid = fpr_values <= float(far_level)
    if not np.any(valid):
        return _metric_nan()
    return float(np.max(ccr_values[valid]))


def compute_mixed_open_set_metrics(
    embeddings,
    labels,
    true_known_mask,
    gallery_embeddings,
    gallery_labels,
    known_unknown_threshold,
    clustering_method="threshold_connected_components",
    clustering_threshold=0.5,
    dir_far_levels=(0.01, 0.001),
    chunk_size=1024,
    knownness_scores=None,
    best_known_labels=None,
):
    """Compute full mixed open-set metrics on val.csv."""
    embeddings, labels = _validate_embedding_labels(embeddings, labels)
    gallery_embeddings, gallery_labels = _validate_embedding_labels(
        gallery_embeddings,
        gallery_labels,
        name="gallery_labels",
    )
    true_known_mask = _as_bool_mask(true_known_mask, "true_known_mask")
    if true_known_mask.shape[0] != labels.shape[0]:
        raise ValueError("true_known_mask length must match labels.")

    result = {
        "mixed/ARI": _metric_nan(),
        "mixed/F1": _metric_nan(),
        "mixed/NormAcc": _metric_nan(),
        "mixed/BAKS": _metric_nan(),
        "mixed/BAUS": _metric_nan(),
        "mixed/OSCR_AUC": _metric_nan(),
        "mixed/known_unknown_threshold": float(known_unknown_threshold),
    }
    for far_level in dir_far_levels:
        key = "mixed/DIR_at_0_1FAR" if float(far_level) == 0.001 else f"mixed/DIR_at_{float(far_level) * 100:g}FAR"
        result[key] = _metric_nan()

    if labels.size == 0:
        return result

    if knownness_scores is None or best_known_labels is None:
        knownness_scores, best_known_labels = best_gallery_matches(
            embeddings,
            gallery_embeddings,
            gallery_labels,
            chunk_size=chunk_size,
        )
    else:
        knownness_scores = np.asarray(knownness_scores, dtype=np.float64)
        best_known_labels = _as_1d_array(best_known_labels, "best_known_labels")

    accept_mask = knownness_scores >= float(known_unknown_threshold)
    rejected_mask = ~accept_mask
    pred_final = np.empty(labels.shape[0], dtype=np.int64)
    pred_final[accept_mask] = best_known_labels[accept_mask].astype(np.int64, copy=False)

    if np.any(rejected_mask):
        rejected_clusters = cluster_embeddings(
            embeddings[rejected_mask],
            method=clustering_method,
            threshold=clustering_threshold,
            chunk_size=chunk_size,
        )
        max_existing = 0
        if labels.size:
            max_existing = max(max_existing, int(np.max(labels.astype(np.int64, copy=False))))
        if gallery_labels.size:
            max_existing = max(max_existing, int(np.max(gallery_labels.astype(np.int64, copy=False))))
        pred_final[rejected_mask] = max_existing + 1 + rejected_clusters

    result["mixed/ARI"] = float(adjusted_rand_score(labels, pred_final))
    result["mixed/F1"] = float(
        f1_score(true_known_mask.astype(np.int64), accept_mask.astype(np.int64), zero_division=0)
    )
    precision, recall, _, _ = precision_recall_fscore_support(
        true_known_mask.astype(np.int64),
        accept_mask.astype(np.int64),
        labels=[1, 0],
        zero_division=0,
    )
    result["mixed/known_precision"] = float(precision[0])
    result["mixed/known_recall"] = float(recall[0])
    result["mixed/unknown_precision"] = float(precision[1])
    result["mixed/unknown_recall"] = float(recall[1])

    baks = _balanced_known_accuracy(true_known_mask, labels, best_known_labels, accept_mask)
    baus = _balanced_unknown_accuracy(true_known_mask, labels, accept_mask)
    result["mixed/BAKS"] = float(baks)
    result["mixed/BAUS"] = float(baus)
    result["mixed/NormAcc"] = norm_accuracy(baks, baus)

    known_scores = knownness_scores[true_known_mask]
    unknown_scores = knownness_scores[~true_known_mask]
    known_correct = best_known_labels[true_known_mask] == labels[true_known_mask]
    result["mixed/OSCR_AUC"] = compute_oscr_auc(known_scores, unknown_scores, known_correct)
    for far_level in dir_far_levels:
        key = "mixed/DIR_at_0_1FAR" if float(far_level) == 0.001 else f"mixed/DIR_at_{float(far_level) * 100:g}FAR"
        result[key] = compute_dir_at_far(known_scores, unknown_scores, known_correct, far_level=far_level)
    return result


def should_run_diagnostics(epoch_index, every_n_epochs=5):
    every_n_epochs = int(every_n_epochs)
    if every_n_epochs <= 0:
        raise ValueError("diagnostics_every_n_epochs must be > 0.")
    return ((int(epoch_index) + 1) % every_n_epochs) == 0


def _sample_positive_pairs(labels, max_pairs, rng):
    labels = _as_1d_array(labels, "labels")
    groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    groups = [group for group in groups if group.size >= 2]
    if not groups or max_pairs <= 0:
        return np.empty((0, 2), dtype=np.int64)

    pairs = np.empty((int(max_pairs), 2), dtype=np.int64)
    for idx in range(int(max_pairs)):
        group = groups[int(rng.integers(0, len(groups)))]
        selected = rng.choice(group, size=2, replace=False)
        pairs[idx] = selected
    return pairs


def _sample_negative_pairs(labels, max_pairs, rng):
    labels = _as_1d_array(labels, "labels")
    num_samples = labels.shape[0]
    if num_samples < 2 or np.unique(labels).size < 2 or max_pairs <= 0:
        return np.empty((0, 2), dtype=np.int64)

    pairs = []
    attempts = 0
    max_attempts = int(max_pairs) * 20
    while len(pairs) < int(max_pairs) and attempts < max_attempts:
        attempts += 1
        i = int(rng.integers(0, num_samples))
        j = int(rng.integers(0, num_samples - 1))
        if j >= i:
            j += 1
        if labels[i] != labels[j]:
            pairs.append((i, j))
    return np.asarray(pairs, dtype=np.int64) if pairs else np.empty((0, 2), dtype=np.int64)


def _pair_cosines(embeddings, pairs):
    if pairs.size == 0:
        return np.empty(0, dtype=np.float64)
    return np.sum(embeddings[pairs[:, 0]] * embeddings[pairs[:, 1]], axis=1).astype(np.float64, copy=False)


def compute_embedding_diagnostics(
    embeddings,
    labels,
    gallery_embeddings=None,
    gallery_labels=None,
    query_embeddings=None,
    query_labels=None,
    max_pairs=100000,
    chunk_size=1024,
    random_state=42,
):
    """Compute sampled embedding-space diagnostics."""
    raw_embeddings = _to_numpy(embeddings).astype(np.float64, copy=False)
    if raw_embeddings.ndim != 2:
        raise ValueError("embeddings must have shape (num_samples, embedding_dim).")
    labels = _as_1d_array(labels, "labels")
    if labels.shape[0] != raw_embeddings.shape[0]:
        raise ValueError("labels length must match embeddings.")

    result = {
        "diagnostics/intra_similarity": _metric_nan(),
        "diagnostics/inter_similarity": _metric_nan(),
        "diagnostics/intra_inter_gap": _metric_nan(),
        "diagnostics/embedding_norm_mean": _metric_nan(),
        "diagnostics/embedding_norm_std": _metric_nan(),
        "diagnostics/embedding_norm_min": _metric_nan(),
        "diagnostics/embedding_norm_max": _metric_nan(),
        "diagnostics/alignment": _metric_nan(),
        "diagnostics/uniformity": _metric_nan(),
        "diagnostics/knn_top1": _metric_nan(),
        "diagnostics/knn_top5": _metric_nan(),
    }
    if raw_embeddings.shape[0] == 0:
        return result

    norms = np.linalg.norm(raw_embeddings, axis=1)
    result["diagnostics/embedding_norm_mean"] = float(np.mean(norms))
    result["diagnostics/embedding_norm_std"] = float(np.std(norms))
    result["diagnostics/embedding_norm_min"] = float(np.min(norms))
    result["diagnostics/embedding_norm_max"] = float(np.max(norms))

    normalized = _normalize_embeddings(raw_embeddings).astype(np.float32, copy=False)
    rng = np.random.default_rng(int(random_state))
    half_pairs = max(int(max_pairs) // 2, 1)
    positive_pairs = _sample_positive_pairs(labels, half_pairs, rng)
    negative_pairs = _sample_negative_pairs(labels, half_pairs, rng)

    positive_cosines = _pair_cosines(normalized, positive_pairs)
    negative_cosines = _pair_cosines(normalized, negative_pairs)
    intra = _safe_mean(positive_cosines)
    inter = _safe_mean(negative_cosines)
    result["diagnostics/intra_similarity"] = intra
    result["diagnostics/inter_similarity"] = inter
    if math.isfinite(intra) and math.isfinite(inter):
        result["diagnostics/intra_inter_gap"] = float(intra - inter)
    if positive_cosines.size:
        result["diagnostics/alignment"] = float(np.mean(2.0 - (2.0 * positive_cosines)))

    uniform_pairs = np.vstack([positive_pairs, negative_pairs]) if positive_pairs.size or negative_pairs.size else np.empty((0, 2), dtype=np.int64)
    uniform_cosines = _pair_cosines(normalized, uniform_pairs)
    if uniform_cosines.size:
        distances_sq = 2.0 - (2.0 * uniform_cosines)
        result["diagnostics/uniformity"] = float(np.log(np.mean(np.exp(-2.0 * distances_sq))))

    if (
        gallery_embeddings is not None
        and gallery_labels is not None
        and query_embeddings is not None
        and query_labels is not None
    ):
        knn_metrics = compute_known_retrieval_metrics(
            query_embeddings,
            query_labels,
            gallery_embeddings,
            gallery_labels,
            top_k=(1, 5),
            cmc_ranks=(),
            chunk_size=chunk_size,
            compute_map=False,
        )
        result["diagnostics/knn_top1"] = float(knn_metrics.get("known/top1", _metric_nan()))
        result["diagnostics/knn_top5"] = float(knn_metrics.get("known/top5", _metric_nan()))
    return result


def _true_known_mask_from_cls(cls_labels, metric_labels, gallery_labels, unknown_label=-100):
    if cls_labels is not None:
        cls_labels = _as_1d_array(cls_labels, "cls_labels")
        return cls_labels != unknown_label
    metric_labels = _as_1d_array(metric_labels, "metric_labels")
    gallery_labels = _as_1d_array(gallery_labels, "gallery_labels")
    return np.isin(metric_labels, np.unique(gallery_labels))


def evaluate_all_validation_splits(
    gallery_embeddings,
    gallery_labels,
    known_embeddings,
    known_labels,
    unknown_embeddings,
    unknown_labels,
    mixed_embeddings,
    mixed_labels,
    known_cls_labels=None,
    unknown_cls_labels=None,
    mixed_cls_labels=None,
    config=None,
    epoch_index=None,
    classification_ignore_index=-100,
    chunk_size=1024,
    random_state=42,
):
    """Compute all configured ApexID open-set validation metric groups."""
    config = dict(config or {})
    diagnostics_cfg = dict(config.get("diagnostics", {}) or {})
    clustering_cfg = dict(config.get("clustering", {}) or {})
    rejection_cfg = dict(config.get("rejection", {}) or {})
    dir_cfg = dict(config.get("dir", {}) or {})

    top_k = tuple(config.get("top_k", (1, 5, 10)))
    cmc_ranks = tuple(config.get("cmc_ranks", (20, 50)))
    clustering_method = clustering_cfg.get("method", "threshold_connected_components")
    clustering_threshold = float(clustering_cfg.get("threshold", 0.5))
    target_tpr = float(rejection_cfg.get("fpr95_target_tpr", 0.95))
    dir_far_levels = tuple(dir_cfg.get("far_levels", (0.01, 0.001)))

    gallery_embeddings, gallery_labels = _validate_embedding_labels(
        gallery_embeddings,
        gallery_labels,
        name="gallery_labels",
    )
    known_embeddings, known_labels = _validate_embedding_labels(
        known_embeddings,
        known_labels,
        name="known_labels",
    )
    unknown_embeddings, unknown_labels = _validate_embedding_labels(
        unknown_embeddings,
        unknown_labels,
        name="unknown_labels",
    )
    mixed_embeddings, mixed_labels = _validate_embedding_labels(
        mixed_embeddings,
        mixed_labels,
        name="mixed_labels",
    )

    full_counts = {
        "gallery": int(gallery_labels.shape[0]),
        "known": int(known_labels.shape[0]),
        "unknown": int(unknown_labels.shape[0]),
        "mixed": int(mixed_labels.shape[0]),
    }
    sample_limits = _sample_limits_from_config(config)
    rng = np.random.default_rng(int(random_state))

    mixed_true_known_full = _true_known_mask_from_cls(
        mixed_cls_labels,
        mixed_labels,
        gallery_labels,
        unknown_label=classification_ignore_index,
    )
    known_indices = _balanced_sample_indices(known_labels, sample_limits["known"], rng)
    unknown_indices = _balanced_sample_indices(unknown_labels, sample_limits["unknown"], rng)
    mixed_indices = _balanced_sample_indices(mixed_labels, sample_limits["mixed"], rng)

    known_embeddings = known_embeddings[known_indices]
    known_labels = known_labels[known_indices]
    known_cls_labels = _take_optional_1d(known_cls_labels, known_indices, "known_cls_labels")
    unknown_embeddings = unknown_embeddings[unknown_indices]
    unknown_labels = unknown_labels[unknown_indices]
    unknown_cls_labels = _take_optional_1d(unknown_cls_labels, unknown_indices, "unknown_cls_labels")
    mixed_embeddings = mixed_embeddings[mixed_indices]
    mixed_labels = mixed_labels[mixed_indices]
    mixed_cls_labels = _take_optional_1d(mixed_cls_labels, mixed_indices, "mixed_cls_labels")
    mixed_true_known = mixed_true_known_full[mixed_indices]

    required_gallery_labels = known_labels
    if mixed_labels.size and np.any(mixed_true_known):
        required_gallery_labels = np.concatenate([required_gallery_labels, mixed_labels[mixed_true_known]])
    gallery_indices = _gallery_sample_indices(
        gallery_labels,
        sample_limits["gallery"],
        required_gallery_labels,
        rng,
    )
    gallery_embeddings = gallery_embeddings[gallery_indices]
    gallery_labels = gallery_labels[gallery_indices]

    used_counts = {
        "gallery": int(gallery_labels.shape[0]),
        "known": int(known_labels.shape[0]),
        "unknown": int(unknown_labels.shape[0]),
        "mixed": int(mixed_labels.shape[0]),
    }
    metrics_sampled = any(used_counts[name] < full_counts[name] for name in full_counts)

    metrics = {}
    if metrics_sampled:
        metrics.update(
            {
                "validation/metrics_sampled": 1,
                "validation/gallery_samples_used": used_counts["gallery"],
                "validation/gallery_samples_total": full_counts["gallery"],
                "validation/known_samples_used": used_counts["known"],
                "validation/known_samples_total": full_counts["known"],
                "validation/unknown_samples_used": used_counts["unknown"],
                "validation/unknown_samples_total": full_counts["unknown"],
                "validation/mixed_samples_used": used_counts["mixed"],
                "validation/mixed_samples_total": full_counts["mixed"],
            }
        )

    if bool(config.get("compute_known", True)):
        known_metrics, known_scores, known_best_labels = compute_known_retrieval_metrics(
            known_embeddings,
            known_labels,
            gallery_embeddings,
            gallery_labels,
            top_k=top_k,
            cmc_ranks=cmc_ranks,
            chunk_size=chunk_size,
            return_matches=True,
        )
        metrics.update(known_metrics)
    else:
        known_scores, known_best_labels = best_gallery_matches(
            known_embeddings,
            gallery_embeddings,
            gallery_labels,
            chunk_size=chunk_size,
        )

    unknown_scores, unknown_best_labels = best_gallery_matches(
        unknown_embeddings,
        gallery_embeddings,
        gallery_labels,
        chunk_size=chunk_size,
    )

    mixed_is_concatenated = _is_concatenated_split(
        mixed_embeddings,
        mixed_labels,
        known_embeddings,
        known_labels,
        unknown_embeddings,
        unknown_labels,
    )
    if mixed_is_concatenated:
        metrics["validation/mixed_scores_reused"] = 1
        mixed_scores = np.concatenate([known_scores, unknown_scores])
        mixed_best_labels = np.concatenate([known_best_labels, unknown_best_labels])
    else:
        metrics["validation/mixed_scores_reused"] = 0
        mixed_scores, mixed_best_labels = best_gallery_matches(
            mixed_embeddings,
            gallery_embeddings,
            gallery_labels,
            chunk_size=chunk_size,
        )

    threshold = config.get("known_unknown_threshold")
    if threshold in {None, "", "null"}:
        threshold = select_threshold_by_normacc(
            mixed_scores,
            mixed_true_known,
            mixed_labels,
            mixed_best_labels,
        )
    threshold = float(threshold)

    if bool(config.get("compute_unknown", True)):
        metrics.update(
            compute_unknown_clustering_metrics(
                unknown_embeddings,
                unknown_labels,
                method=clustering_method,
                threshold=clustering_threshold,
                chunk_size=chunk_size,
                knownness_scores=unknown_scores,
                known_unknown_threshold=threshold,
            )
        )
    if bool(config.get("compute_rejection", True)):
        metrics.update(
            compute_rejection_metrics(
                known_scores,
                unknown_scores,
                threshold=threshold,
                target_tpr=target_tpr,
            )
        )
    if bool(config.get("compute_mixed", True)):
        metrics.update(
            compute_mixed_open_set_metrics(
                mixed_embeddings,
                mixed_labels,
                mixed_true_known,
                gallery_embeddings,
                gallery_labels,
                known_unknown_threshold=threshold,
                clustering_method=clustering_method,
                clustering_threshold=clustering_threshold,
                dir_far_levels=dir_far_levels,
                chunk_size=chunk_size,
                knownness_scores=mixed_scores,
                best_known_labels=mixed_best_labels,
            )
        )

    diagnostics_enabled = bool(diagnostics_cfg.get("enabled", True))
    diagnostics_every = int(config.get("diagnostics_every_n_epochs", 5))
    run_diagnostics = (
        diagnostics_enabled
        and epoch_index is not None
        and should_run_diagnostics(epoch_index, every_n_epochs=diagnostics_every)
    )
    metrics["validation/diagnostics_ran"] = int(bool(run_diagnostics))
    if run_diagnostics:
        metrics.update(
            compute_embedding_diagnostics(
                mixed_embeddings,
                mixed_labels,
                gallery_embeddings=gallery_embeddings,
                gallery_labels=gallery_labels,
                query_embeddings=known_embeddings,
                query_labels=known_labels,
                max_pairs=int(diagnostics_cfg.get("max_pairs", 100000)),
                chunk_size=chunk_size,
                random_state=random_state,
            )
        )
    return metrics


__all__ = [
    "best_gallery_matches",
    "cluster_embeddings",
    "compute_dir_at_far",
    "compute_embedding_diagnostics",
    "compute_known_retrieval_metrics",
    "compute_mixed_open_set_metrics",
    "compute_oscr_auc",
    "compute_rejection_metrics",
    "compute_unknown_clustering_metrics",
    "evaluate_all_validation_splits",
    "norm_accuracy",
    "select_threshold_by_normacc",
    "should_run_diagnostics",
    "threshold_at_target_tpr",
    "threshold_connected_components",
]
