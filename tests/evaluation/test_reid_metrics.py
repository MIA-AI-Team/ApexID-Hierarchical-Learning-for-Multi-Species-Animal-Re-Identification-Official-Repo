import math

import numpy as np
import torch

from apexid.evaluation import (
    compute_reid_metrics,
    mean_average_precision,
    micro_f1_score,
    nearest_neighbor_predictions,
    top1_accuracy,
    top5_accuracy,
    topk_accuracy,
)


def test_reid_retrieval_metrics_match_hand_checked_values():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.0, 1.0],
            [0.1, 0.9],
            [0.2, 0.8],
            [-1.0, 0.0],
            [-0.9, 0.1],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1, 1, 0, 2, 2])

    assert top1_accuracy(embeddings, labels) == 0.75
    assert top5_accuracy(embeddings, labels) == 1.0
    assert topk_accuracy(embeddings, labels, k=2) == 0.75
    np.testing.assert_allclose(mean_average_precision(embeddings, labels), 359 / 480)

    y_true, y_pred = nearest_neighbor_predictions(embeddings, labels)
    np.testing.assert_array_equal(y_true, labels)
    np.testing.assert_array_equal(y_pred, np.array([0, 0, 0, 1, 1, 1, 2, 2]))
    assert micro_f1_score(y_true, y_pred) == 0.75

    chunked = compute_reid_metrics(embeddings, labels, chunk_size=2)
    np.testing.assert_allclose(chunked["top1"], 0.75)
    np.testing.assert_allclose(chunked["top5"], 1.0)
    np.testing.assert_allclose(chunked["mAP"], 359 / 480)
    np.testing.assert_allclose(chunked["micro_f1"], 0.75)


def test_compute_reid_metrics_accepts_torch_inputs_and_ignore_index():
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
            [-1.0, 0.0],
        ]
    )
    labels = torch.tensor([0, 0, 1, 1, -100])

    metrics = compute_reid_metrics(embeddings, labels, ignore_index=-100, chunk_size=2)

    assert metrics == {
        "top1": 1.0,
        "top5": 1.0,
        "mAP": 1.0,
        "micro_f1": 1.0,
    }


def test_metrics_return_nan_when_no_query_has_positive_gallery_match():
    embeddings = np.eye(3, dtype=np.float32)
    labels = np.array([0, 1, 2])

    assert math.isnan(top1_accuracy(embeddings, labels, chunk_size=1))
    assert math.isnan(top5_accuracy(embeddings, labels, chunk_size=1))
    assert math.isnan(mean_average_precision(embeddings, labels, chunk_size=1))


def test_reid_metrics_match_across_chunk_sizes():
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 0.9],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1, 2, 2])

    full_chunk = compute_reid_metrics(embeddings, labels, chunk_size=len(labels))
    for chunk_size in [1, 2, 4]:
        chunked = compute_reid_metrics(embeddings, labels, chunk_size=chunk_size)
        assert chunked == full_chunk


def test_micro_f1_score_handles_ignore_index_and_shape_errors():
    y_true = np.array([0, 1, 1, -100])
    y_pred = np.array([0, 0, 1, 1])

    assert micro_f1_score(y_true, y_pred, ignore_index=-100) == 2 / 3

    with np.testing.assert_raises(ValueError):
        micro_f1_score(np.array([[0, 1]]), np.array([0, 1]))

    with np.testing.assert_raises(ValueError):
        micro_f1_score(np.array([0, 1]), np.array([0]))
