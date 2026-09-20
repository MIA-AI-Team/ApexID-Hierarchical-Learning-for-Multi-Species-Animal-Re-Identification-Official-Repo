"""AnimalCLEF-style graph clustering for cached similarity matrices.

The implementation is adapted from MIA-AI-Team/AnimalCLEF26.  It is kept
independent of feature extraction so the same frozen WildFusion matrix is used
for Optuna validation and final Stage 00 evaluation.
"""

from __future__ import annotations

from collections import defaultdict
import random
from typing import Sequence

import networkx as nx
import numpy as np


def compute_identity_scores(
    sim_query_db: np.ndarray,
    db_ids: Sequence[str],
    top_r: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Average each query's strongest ``top_r`` matches per known identity."""
    labels = np.asarray(db_ids, dtype=str)
    identity_names = np.asarray(list(dict.fromkeys(labels.tolist())), dtype=object)
    if identity_names.size == 0:
        raise ValueError("Known-identity scoring requires at least one database identity")
    scores = np.zeros((sim_query_db.shape[0], identity_names.size), dtype=np.float32)
    for column, identity in enumerate(identity_names):
        db_columns = np.flatnonzero(labels == identity)
        similarities = sim_query_db[:, db_columns]
        count = min(int(top_r), similarities.shape[1])
        if count <= 0:
            raise ValueError("top_r must be positive")
        if count == similarities.shape[1]:
            scores[:, column] = similarities.mean(axis=1)
        else:
            scores[:, column] = np.partition(similarities, -count, axis=1)[:, -count:].mean(axis=1)
    return identity_names, scores


def summarize_best_identities(
    identity_names: np.ndarray,
    identity_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return best identity, best score, runner-up score, and score margin."""
    if identity_scores.ndim != 2 or identity_scores.shape[1] == 0:
        raise ValueError("identity_scores must contain at least one identity column")
    best_indices = identity_scores.argmax(axis=1)
    best_scores = identity_scores[np.arange(identity_scores.shape[0]), best_indices]
    best_names = np.asarray(identity_names, dtype=object)[best_indices]
    if identity_scores.shape[1] >= 2:
        top_two = np.partition(identity_scores, -2, axis=1)[:, -2:]
        second_scores = np.min(top_two, axis=1)
    else:
        second_scores = np.zeros(identity_scores.shape[0], dtype=np.float32)
    return best_names, best_scores, second_scores, best_scores - second_scores


def _topk_neighbors(sim_matrix: np.ndarray, top_k: int) -> np.ndarray:
    sample_count = sim_matrix.shape[0]
    count = min(int(top_k), sample_count - 1)
    if count <= 0:
        return np.empty((sample_count, 0), dtype=np.int64)
    similarities = sim_matrix.copy()
    np.fill_diagonal(similarities, -np.inf)
    return np.argpartition(similarities, -count, axis=1)[:, -count:]


def chinese_whispers(
    sim_matrix: np.ndarray,
    threshold: float,
    iterations: int,
    top_k: int,
    k_core_degree: int,
    seed: int = 42,
    skip_mutual_knn: bool = False,
) -> np.ndarray:
    """Run the repository's k-core and mutual-kNN Chinese Whispers variant."""
    random.seed(seed)
    np.random.seed(seed)
    similarities = np.asarray(sim_matrix, dtype=np.float32)
    sample_count = similarities.shape[0]
    if similarities.shape != (sample_count, sample_count):
        raise ValueError("Chinese Whispers requires a square similarity matrix")

    adjacency = similarities.copy()
    np.fill_diagonal(adjacency, 0)
    adjacency[adjacency < threshold] = 0
    graph = nx.from_numpy_array(adjacency)
    if graph.number_of_edges():
        try:
            backbone = nx.k_core(graph, k=int(k_core_degree))
            if backbone.number_of_nodes() == 0:
                backbone = graph
        except nx.NetworkXError:
            backbone = graph
    else:
        backbone = graph
    backbone_nodes = list(backbone.nodes()) or list(range(sample_count))

    in_backbone = np.zeros(sample_count, dtype=bool)
    in_backbone[backbone_nodes] = True
    valid_edges = (
        (similarities >= threshold)
        & in_backbone[:, None]
        & in_backbone[None, :]
    )
    np.fill_diagonal(valid_edges, False)
    if not skip_mutual_knn:
        topk = _topk_neighbors(similarities, top_k)
        nearest = np.zeros((sample_count, sample_count), dtype=bool)
        if topk.shape[1]:
            nearest[np.arange(sample_count)[:, None], topk] = True
        valid_edges &= nearest & nearest.T

    sources, targets = np.nonzero(valid_edges)
    graph_list: list[list[tuple[int, float]]] = [[] for _ in range(sample_count)]
    for source, target in zip(sources, targets):
        graph_list[int(source)].append((int(target), float(similarities[source, target])))

    labels = list(range(sample_count))
    nodes = backbone_nodes.copy()
    for _ in range(int(iterations)):
        random.shuffle(nodes)
        changes = 0
        for node in nodes:
            label_weights: defaultdict[int, float] = defaultdict(float)
            for neighbor, weight in graph_list[node]:
                label_weights[labels[neighbor]] += weight
            if not label_weights:
                continue
            new_label = max(label_weights.items(), key=lambda item: (item[1], -item[0]))[0]
            changes += int(new_label != labels[node])
            labels[node] = new_label
        if changes == 0:
            break
    return np.unique(labels, return_inverse=True)[1].astype(np.int64)


def re_ranking(
    sim_matrix: np.ndarray,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
) -> np.ndarray:
    """Apply the AnimalCLEF k-reciprocal re-ranking implementation."""
    similarities = np.asarray(sim_matrix, dtype=np.float32)
    sample_count = similarities.shape[0]
    original_dist = np.clip(1.0 - similarities, 0.0, None)
    column_max = np.max(original_dist, axis=0)
    column_max[column_max == 0] = 1.0
    original_dist = (original_dist / column_max).T.astype(np.float32)
    values = np.zeros((sample_count, sample_count), dtype=np.float32)
    initial_rank = np.argsort(original_dist, axis=1).astype(np.int32)
    half_k1 = int(np.around(k1 / 2.0)) + 1
    for index in range(sample_count):
        forward = initial_rank[index, : k1 + 1]
        backward = initial_rank[forward, : k1 + 1]
        reciprocal = forward[np.where(backward == index)[0]]
        expanded = reciprocal.copy()
        for neighbor in reciprocal:
            neighbor_forward = initial_rank[neighbor, :half_k1]
            neighbor_backward = initial_rank[neighbor_forward, :half_k1]
            neighbor_reciprocal = neighbor_forward[np.where(neighbor_backward == neighbor)[0]]
            if (
                len(neighbor_reciprocal)
                and len(np.intersect1d(neighbor_reciprocal, reciprocal))
                > (2.0 / 3.0) * len(neighbor_reciprocal)
            ):
                expanded = np.append(expanded, neighbor_reciprocal)
        expanded = np.unique(expanded)
        weight = np.exp(-original_dist[index, expanded])
        if weight.sum() > 0:
            values[index, expanded] = weight / weight.sum()
    if k2 > 1:
        expanded_values = np.zeros_like(values)
        for index in range(sample_count):
            expanded_values[index] = np.mean(values[initial_rank[index, :k2]], axis=0)
        values = expanded_values
    jaccard = np.zeros((sample_count, sample_count), dtype=np.float32)
    inverse_index = [np.where(values[:, column] != 0)[0] for column in range(sample_count)]
    for index in range(sample_count):
        temp_min = np.zeros(sample_count, dtype=np.float32)
        for column in np.where(values[index] != 0)[0]:
            rows = inverse_index[column]
            temp_min[rows] += np.minimum(values[index, column], values[rows, column])
        jaccard[index] = 1.0 - temp_min / (2.0 - temp_min + 1e-12)
    final_dist = np.clip(
        jaccard * (1.0 - lambda_value) + original_dist * lambda_value,
        0.0,
        1.0,
    )
    np.fill_diagonal(final_dist, 0.0)
    return (1.0 - final_dist).astype(np.float32)


def resolve_cluster_labels(
    cluster_ids: np.ndarray,
    best_names: np.ndarray,
    best_scores: np.ndarray,
    margins: np.ndarray,
    tau_attach: float,
    tau_margin: float,
) -> np.ndarray:
    """Attach whole clusters to known IDs by accumulated confident votes."""
    votes: defaultdict[int, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    for index, cluster in enumerate(cluster_ids):
        if best_scores[index] >= tau_attach and margins[index] >= tau_margin:
            votes[int(cluster)][str(best_names[index])] += float(best_scores[index])
    known_labels = {
        cluster: max(identity_votes.items(), key=lambda item: item[1])[0]
        for cluster, identity_votes in votes.items()
        if identity_votes
    }
    return np.asarray(
        [
            known_labels.get(int(cluster), f"__new_cluster_{int(cluster)}")
            for cluster in cluster_ids
        ],
        dtype=str,
    )

