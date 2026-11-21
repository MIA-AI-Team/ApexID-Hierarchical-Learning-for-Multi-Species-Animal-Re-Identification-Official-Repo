import math

import numpy as np
import pytest

from apexid.evaluation import B3_METRIC_VERSION
from apexid.evaluation.benchmark_metrics import b3_metrics


def _reference_b3(true_labels, predicted_labels):
    """Independent direct per-item implementation used only by tests."""
    true_labels = list(true_labels)
    predicted_labels = list(predicted_labels)
    if len(true_labels) != len(predicted_labels):
        raise ValueError
    if not true_labels:
        return float("nan"), float("nan"), float("nan")

    precision_values = []
    recall_values = []
    for index, (true_label, predicted_label) in enumerate(zip(true_labels, predicted_labels)):
        true_members = {
            member_index
            for member_index, member_label in enumerate(true_labels)
            if member_label == true_label
        }
        predicted_members = {
            member_index
            for member_index, member_label in enumerate(predicted_labels)
            if member_label == predicted_label
        }
        intersection = len(true_members.intersection(predicted_members))
        precision_values.append(intersection / len(predicted_members))
        recall_values.append(intersection / len(true_members))

    precision = float(np.mean(precision_values))
    recall = float(np.mean(recall_values))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


@pytest.mark.parametrize(
    ("true_labels", "predicted_labels", "expected"),
    [
        (["A", "A", "B", "B"], ["X", "X", "Y", "Y"], (1.0, 1.0, 1.0)),
        (["A", "A", "A"], ["X", "Y", "Z"], (1.0, 1.0 / 3.0, 0.5)),
        (["A", "B", "C"], ["X", "X", "X"], (1.0 / 3.0, 1.0, 0.5)),
        (["A", "A", "B"], ["X", "Y", "Y"], (2.0 / 3.0,) * 3),
        (["A"], ["X"], (1.0, 1.0, 1.0)),
    ],
)
def test_b3_analytical_cases(true_labels, predicted_labels, expected):
    result = b3_metrics(true_labels, predicted_labels)

    assert result["b3_precision"] == pytest.approx(expected[0])
    assert result["b3_recall"] == pytest.approx(expected[1])
    assert result["b3_f1"] == pytest.approx(expected[2])


def test_b3_is_label_invariant_and_accepts_arbitrary_hashable_labels():
    true = [("animal", 1), ("animal", 1), ("animal", 2), ("animal", 2)]
    first = b3_metrics(true, ["X", "X", "Y", "Y"])
    second = b3_metrics(true, [91, 91, -7, -7])

    assert first == second == {"b3_precision": 1.0, "b3_recall": 1.0, "b3_f1": 1.0}


def test_b3_empty_input_uses_benchmark_nan_convention():
    result = b3_metrics([], [])

    assert all(math.isnan(value) for value in result.values())


def test_b3_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="same-length"):
        b3_metrics(["A"], ["X", "Y"])
    with pytest.raises(ValueError, match="1D"):
        b3_metrics(np.asarray([["A", "B"]]), ["X", "Y"])
    with pytest.raises(TypeError, match="hashable"):
        b3_metrics([["A"], ["B"]], ["X", "Y"])


def test_b3_matches_independent_itemwise_reference_on_random_partitions():
    rng = np.random.default_rng(20260830)
    for sample_count in range(1, 40):
        for _ in range(8):
            true = rng.integers(-3, 8, size=sample_count).tolist()
            predicted = rng.integers(20, 31, size=sample_count).tolist()
            expected = _reference_b3(true, predicted)
            result = b3_metrics(true, predicted)

            assert result["b3_precision"] == pytest.approx(expected[0])
            assert result["b3_recall"] == pytest.approx(expected[1])
            assert result["b3_f1"] == pytest.approx(expected[2])


def test_b3_metric_version_is_public_and_specific():
    assert B3_METRIC_VERSION == "standard_item_averaged_v2"
