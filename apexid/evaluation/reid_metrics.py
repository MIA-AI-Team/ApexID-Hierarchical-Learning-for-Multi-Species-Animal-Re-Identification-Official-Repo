import numpy as np
import torch


def _to_numpy(array):
    """Convert supported tensor/array inputs to NumPy.

    Args:
        array (torch.Tensor or numpy.ndarray or Sequence): Input values. Tensors
            may be on CPU or GPU.

    Returns:
        numpy.ndarray: CPU NumPy array detached from autograd when the input is a
        tensor.
    """
    if torch.is_tensor(array):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _normalize_embeddings(embeddings):
    """L2-normalize an embedding matrix row-wise.

    Args:
        embeddings (torch.Tensor or numpy.ndarray): Embedding matrix with shape
            ``(num_samples, embedding_dim)``.

    Returns:
        numpy.ndarray: Float64 normalized embedding matrix with the same shape.

    Raises:
        ValueError: If ``embeddings`` is not a 2D matrix.
    """
    embeddings = _to_numpy(embeddings).astype(np.float64, copy=False)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape (num_samples, embedding_dim).")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return embeddings / norms


def _validate_labels(labels, num_samples, name="labels"):
    """Validate a one-dimensional label vector.

    Args:
        labels (torch.Tensor or numpy.ndarray or Sequence): Label vector with
            shape ``(num_samples,)``.
        num_samples (int): Expected number of labels.
        name (str): Name used in error messages.

    Returns:
        numpy.ndarray: One-dimensional label array.

    Raises:
        ValueError: If labels are not one-dimensional or the length does not
        match ``num_samples``.
    """
    labels = _to_numpy(labels)
    if labels.ndim != 1:
        raise ValueError(f"{name} must have shape (num_samples,).")
    if labels.shape[0] != num_samples:
        raise ValueError(
            f"{name} length must match embeddings. Expected {num_samples}, got {labels.shape[0]}."
        )
    return labels


def _valid_mask(labels, ignore_index=None):
    """Build the evaluation mask after applying an optional ignore label.

    Args:
        labels (numpy.ndarray): Label vector with shape ``(num_samples,)``.
        ignore_index (int or None): Label value to exclude from evaluation.

    Returns:
        numpy.ndarray: Boolean mask with shape ``(num_samples,)``.
    """
    if ignore_index is None:
        return np.ones(labels.shape[0], dtype=bool)
    return labels != ignore_index


def _ranking_from_similarity(similarity):
    """Return descending nearest-neighbor indices excluding self matches.

    Args:
        similarity (numpy.ndarray): Square cosine-similarity matrix with shape
            ``(num_samples, num_samples)``.

    Returns:
        numpy.ndarray: Ranked gallery indices with shape
        ``(num_samples, num_samples - 1)``.
    """
    similarity = similarity.copy()
    np.fill_diagonal(similarity, -np.inf)
    ranked = np.argsort(-similarity, axis=1)
    self_indices = np.arange(ranked.shape[0])[:, None]
    return ranked[ranked != self_indices].reshape(ranked.shape[0], ranked.shape[1] - 1)


def _relevance_from_ranked_labels(labels, ranked):
    """Build a binary relevance matrix from ranked gallery indices.

    Args:
        labels (numpy.ndarray): Identity labels with shape ``(num_samples,)``.
        ranked (numpy.ndarray): Ranked gallery indices with shape
            ``(num_samples, num_gallery)``.

    Returns:
        numpy.ndarray: Integer relevance matrix with shape
        ``(num_samples, num_gallery)`` where ``1`` means the ranked gallery item
        has the same identity as the query.
    """
    return (labels[ranked] == labels[:, None]).astype(np.int32)


def topk_accuracy(embeddings, labels, k=1, ignore_index=None):
    """Compute leave-one-out re-identification top-k accuracy.

    Args:
        embeddings (torch.Tensor or numpy.ndarray): Embedding matrix with shape
            ``(num_samples, embedding_dim)``.
        labels (torch.Tensor or numpy.ndarray or Sequence): Identity labels with
            shape ``(num_samples,)``.
        k (int): Rank cutoff. ``k=1`` is top-1 accuracy and ``k=5`` is top-5
            accuracy.
        ignore_index (int or None): Optional label value excluded from
            evaluation before ranking.

    Returns:
        float: Fraction of valid queries whose top-k retrieved gallery contains
        at least one sample with the same identity. Returns ``nan`` when no query
        has a positive match in the gallery.

    Functionality:
        L2-normalizes embeddings, computes cosine similarity, removes self
        matches, ranks all other samples for each query, and checks whether any
        same-identity sample appears in the top-k positions.
    """
    if k <= 0:
        raise ValueError("k must be > 0.")

    emb = _normalize_embeddings(embeddings)
    labels = _validate_labels(labels, emb.shape[0])
    mask = _valid_mask(labels, ignore_index=ignore_index)
    emb = emb[mask]
    labels = labels[mask]

    if labels.shape[0] < 2:
        return float("nan")

    similarity = emb @ emb.T
    ranked = _ranking_from_similarity(similarity)
    relevance = _relevance_from_ranked_labels(labels, ranked)

    positive_counts = relevance.sum(axis=1)
    valid_queries = positive_counts > 0
    if not np.any(valid_queries):
        return float("nan")

    k = min(int(k), relevance.shape[1])
    hits = relevance[:, :k].any(axis=1)
    return float(hits[valid_queries].mean())


def top1_accuracy(embeddings, labels, ignore_index=None):
    """Compute leave-one-out re-identification top-1 accuracy.

    Args:
        embeddings (torch.Tensor or numpy.ndarray): Embedding matrix with shape
            ``(num_samples, embedding_dim)``.
        labels (torch.Tensor or numpy.ndarray or Sequence): Identity labels with
            shape ``(num_samples,)``.
        ignore_index (int or None): Optional label value excluded from
            evaluation.

    Returns:
        float: Top-1 retrieval accuracy.
    """
    return topk_accuracy(embeddings, labels, k=1, ignore_index=ignore_index)


def top5_accuracy(embeddings, labels, ignore_index=None):
    """Compute leave-one-out re-identification top-5 accuracy.

    Args:
        embeddings (torch.Tensor or numpy.ndarray): Embedding matrix with shape
            ``(num_samples, embedding_dim)``.
        labels (torch.Tensor or numpy.ndarray or Sequence): Identity labels with
            shape ``(num_samples,)``.
        ignore_index (int or None): Optional label value excluded from
            evaluation.

    Returns:
        float: Top-5 retrieval accuracy.
    """
    return topk_accuracy(embeddings, labels, k=5, ignore_index=ignore_index)


def mean_average_precision(embeddings, labels, ignore_index=None):
    """Compute leave-one-out mean Average Precision (mAP).

    Args:
        embeddings (torch.Tensor or numpy.ndarray): Embedding matrix with shape
            ``(num_samples, embedding_dim)``.
        labels (torch.Tensor or numpy.ndarray or Sequence): Identity labels with
            shape ``(num_samples,)``.
        ignore_index (int or None): Optional label value excluded from
            evaluation before ranking.

    Returns:
        float: Mean Average Precision over valid queries. A query is valid when
        it has at least one same-identity match in the gallery. Returns ``nan``
        when there are no valid queries.

    Functionality:
        For each query, ranks all other samples by cosine similarity, computes
        precision at every correct-match rank, averages those precisions to get
        AP, then averages AP across valid queries.
    """
    emb = _normalize_embeddings(embeddings)
    labels = _validate_labels(labels, emb.shape[0])
    mask = _valid_mask(labels, ignore_index=ignore_index)
    emb = emb[mask]
    labels = labels[mask]

    if labels.shape[0] < 2:
        return float("nan")

    similarity = emb @ emb.T
    ranked = _ranking_from_similarity(similarity)
    relevance = _relevance_from_ranked_labels(labels, ranked)

    ap_values = []
    for rel in relevance:
        positive_ranks = np.flatnonzero(rel)
        if positive_ranks.size == 0:
            continue
        precision_at_hits = [
            (hit_index + 1) / float(rank + 1)
            for hit_index, rank in enumerate(positive_ranks)
        ]
        ap_values.append(float(np.mean(precision_at_hits)))

    if not ap_values:
        return float("nan")
    return float(np.mean(ap_values))


def micro_f1_score(y_true, y_pred, ignore_index=None):
    """Compute micro-averaged F1 score for identity predictions.

    Args:
        y_true (torch.Tensor or numpy.ndarray or Sequence): Ground-truth identity
            labels with shape ``(num_samples,)``.
        y_pred (torch.Tensor or numpy.ndarray or Sequence): Predicted identity
            labels with shape ``(num_samples,)``. These can come from a
            classifier head or from nearest-neighbor retrieval predictions.
        ignore_index (int or None): Optional ground-truth label value excluded
            from the calculation.

    Returns:
        float: Micro F1 score. For single-label multi-class predictions this is
        equivalent to accuracy over the non-ignored samples. Returns ``nan`` when
        no samples remain after masking.

    Functionality:
        Counts global true positives, false positives, and false negatives across
        all identities, then computes ``2TP / (2TP + FP + FN)``.
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must both have shape (num_samples,).")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"y_true and y_pred must have the same length. Got {y_true.shape[0]} and {y_pred.shape[0]}."
        )

    mask = _valid_mask(y_true, ignore_index=ignore_index)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.shape[0] == 0:
        return float("nan")

    labels = np.union1d(y_true, y_pred)
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for label in labels:
        true_positive += int(np.logical_and(y_true == label, y_pred == label).sum())
        false_positive += int(np.logical_and(y_true != label, y_pred == label).sum())
        false_negative += int(np.logical_and(y_true == label, y_pred != label).sum())

    denominator = (2 * true_positive) + false_positive + false_negative
    if denominator == 0:
        return float("nan")
    return float((2 * true_positive) / denominator)


def nearest_neighbor_predictions(embeddings, labels, ignore_index=None):
    """Predict each sample identity using its nearest non-self neighbor.

    Args:
        embeddings (torch.Tensor or numpy.ndarray): Embedding matrix with shape
            ``(num_samples, embedding_dim)``.
        labels (torch.Tensor or numpy.ndarray or Sequence): Identity labels with
            shape ``(num_samples,)``.
        ignore_index (int or None): Optional label value excluded before ranking.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: ``(y_true, y_pred)`` label arrays for
        valid queries. Each prediction is the identity label of the closest
        non-self sample by cosine similarity.

    Functionality:
        Provides retrieval-based predictions that can be passed directly to
        ``micro_f1_score``.
    """
    emb = _normalize_embeddings(embeddings)
    labels = _validate_labels(labels, emb.shape[0])
    mask = _valid_mask(labels, ignore_index=ignore_index)
    emb = emb[mask]
    labels = labels[mask]

    if labels.shape[0] < 2:
        empty = np.array([], dtype=labels.dtype)
        return empty, empty

    similarity = emb @ emb.T
    ranked = _ranking_from_similarity(similarity)
    y_pred = labels[ranked[:, 0]]
    return labels.copy(), y_pred


def compute_reid_metrics(embeddings, labels, ignore_index=None):
    """Compute the standard ApexID re-identification metric bundle.

    Args:
        embeddings (torch.Tensor or numpy.ndarray): Embedding matrix with shape
            ``(num_samples, embedding_dim)``.
        labels (torch.Tensor or numpy.ndarray or Sequence): Identity labels with
            shape ``(num_samples,)``.
        ignore_index (int or None): Optional label value excluded from all
            metrics.

    Returns:
        dict: Metric dictionary with keys ``top1``, ``top5``, ``mAP``, and
        ``micro_f1``.

    Functionality:
        Computes leave-one-out retrieval top-1/top-5 accuracy, mAP, and micro-F1
        from nearest-neighbor identity predictions using the same embedding and
        label inputs.
    """
    y_true, y_pred = nearest_neighbor_predictions(
        embeddings,
        labels,
        ignore_index=ignore_index,
    )
    return {
        "top1": top1_accuracy(embeddings, labels, ignore_index=ignore_index),
        "top5": top5_accuracy(embeddings, labels, ignore_index=ignore_index),
        "mAP": mean_average_precision(embeddings, labels, ignore_index=ignore_index),
        "micro_f1": micro_f1_score(y_true, y_pred),
    }


__all__ = [
    "compute_reid_metrics",
    "mean_average_precision",
    "micro_f1_score",
    "nearest_neighbor_predictions",
    "top1_accuracy",
    "top5_accuracy",
    "topk_accuracy",
]
