import numpy as np
import pandas as pd
import pytest

from apexid.evaluation.benchmark_evaluator import (
    BOOTSTRAP_VERSION,
    ONLINE_PROTOTYPE_UPDATE_VERSION,
    DiscoveredPrototypeState,
    EvaluationConfig,
    _online_assign,
    bootstrap_headlines,
    infer_online,
    occurrence_layer_sample,
)


def _bootstrap_frame():
    return pd.DataFrame(
        {
            "image_id": ["a-k", "a-n", "b-k", "b-n"],
            "true_identity": ["A", "A", "B", "B"],
            "predicted_identity": ["A", "A", "B", "B"],
            "scenario": ["known_identity", "novel_identity_known_species"] * 2,
            "novelty_score": [0.0, 1.0, 0.0, 1.0],
            "predicted_status": ["known", "novel"] * 2,
            "known_database_id": ["A", np.nan, "B", np.nan],
            "_known_r1": [1.0, np.nan, 1.0, np.nan],
            "_known_ap": [1.0, np.nan, 1.0, np.nan],
        }
    )


def test_discovered_prototype_uses_float64_unnormalized_sum():
    state = DiscoveredPrototypeState.from_embedding(np.asarray([3.0, 4.0], dtype=np.float32))
    state.update(np.asarray([0.0, 2.0], dtype=np.float32))
    state.update(np.asarray([-1.0, 0.0], dtype=np.float32))
    assert state.vector_sum.dtype == np.float64
    assert state.count == 3
    np.testing.assert_array_equal(state.vector_sum, np.asarray([2.0, 6.0]))
    np.testing.assert_allclose(state.scoring_view, np.asarray([2.0, 6.0]) / np.sqrt(40.0))
    assert ONLINE_PROTOTYPE_UPDATE_VERSION == "unnormalized_vector_sum_v2"


def test_incremental_prototype_sum_is_order_invariant():
    rng = np.random.default_rng(17)
    vectors = rng.normal(size=(20, 8))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    expected = vectors.sum(axis=0, dtype=np.float64)
    for order in (np.arange(len(vectors)), rng.permutation(len(vectors))):
        state = DiscoveredPrototypeState.from_embedding(vectors[order[0]])
        for index in order[1:]:
            state.update(vectors[index])
        np.testing.assert_allclose(state.vector_sum, expected, rtol=0, atol=1e-14)


def test_online_assignment_is_causal_and_names_are_deterministic():
    embeddings = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]])
    labels, scores, created = _online_assign(embeddings, threshold=0.9)
    assert labels == ["NEW_000001", "NEW_000001", "NEW_000002", "NEW_000002"]
    assert created == [True, False, True, False]
    assert np.isnan(scores[0])
    assert scores[2] < 0.9
    assert _online_assign(embeddings, threshold=0.9)[0] == labels


def test_future_queries_do_not_change_prior_online_assignments():
    embeddings = np.asarray([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [-1.0, 0.0]])
    for prefix_length in range(1, len(embeddings) + 1):
        prefix = _online_assign(embeddings[:prefix_length], threshold=0.7)
        complete = _online_assign(embeddings, threshold=0.7)
        assert prefix[0] == complete[0][:prefix_length]
        assert prefix[2] == complete[2][:prefix_length]


def test_known_prototype_updates_are_rejected_before_online_inference(tmp_path):
    config = EvaluationConfig(
        weights=tmp_path / "weights.pt",
        benchmark_dir=tmp_path,
        data_root=tmp_path,
        output_dir=tmp_path,
        update_known_prototypes=True,
    )
    with pytest.raises(ValueError, match="known gallery prototypes are static"):
        infer_online(pd.DataFrame(), {}, None, {}, config)


def test_occurrence_layers_prevent_self_copy_interaction_but_preserve_merges():
    frame = _bootstrap_frame().iloc[[0, 2]].copy()
    frame["predicted_identity"] = "MERGED"
    sample = occurrence_layer_sample(frame, ["A", "A", "B"])
    first_a, second_a, first_b = (sample.iloc[index] for index in range(3))
    assert first_a["_bootstrap_true_label"] != second_a["_bootstrap_true_label"]
    assert first_a["_bootstrap_predicted_label"] != second_a["_bootstrap_predicted_label"]
    assert first_a["_bootstrap_predicted_label"] == first_b["_bootstrap_predicted_label"]


def test_occurrence_layers_retain_original_fragmentation():
    frame = pd.concat([_bootstrap_frame().iloc[[0]], _bootstrap_frame().iloc[[0]]], ignore_index=True)
    frame["image_id"] = ["a1", "a2"]
    frame["predicted_identity"] = ["X", "Y"]
    sample = occurrence_layer_sample(frame, ["A", "A"])
    assert sample.groupby("_bootstrap_true_label")["_bootstrap_predicted_label"].nunique().tolist() == [2, 2]
    first_codes = set(sample.loc[sample._bootstrap_true_label.eq(0), "_bootstrap_predicted_label"])
    second_codes = set(sample.loc[sample._bootstrap_true_label.eq(1), "_bootstrap_predicted_label"])
    assert first_codes.isdisjoint(second_codes)


def test_occurrence_layer_relations_match_independent_pair_reference():
    frame = pd.DataFrame({
        "image_id": ["a1", "a2", "b1", "b2"],
        "true_identity": ["A", "A", "B", "B"],
        "predicted_identity": ["X", "Y", "X", "Z"],
    })
    chosen = ["A", "A", "B"]
    sample = occurrence_layer_sample(frame, chosen)
    source_rows = [(identity, rank, predicted) for identity, rank in [("A", 0), ("A", 1), ("B", 0)] for predicted in frame.loc[frame.true_identity.eq(identity), "predicted_identity"]]
    for left in range(len(sample)):
        for right in range(len(sample)):
            expected_predicted_same = (
                source_rows[left][1] == source_rows[right][1]
                and source_rows[left][2] == source_rows[right][2]
            )
            observed_predicted_same = (
                sample.iloc[left]["_bootstrap_predicted_label"]
                == sample.iloc[right]["_bootstrap_predicted_label"]
            )
            assert observed_predicted_same == expected_predicted_same


def test_bootstrap_is_seeded_complete_and_perfect_intervals_are_one():
    first = bootstrap_headlines(_bootstrap_frame(), 40, 123, minimum_valid_fraction=0.9)
    second = bootstrap_headlines(_bootstrap_frame(), 40, 123, minimum_valid_fraction=0.9)
    pd.testing.assert_frame_equal(first, second)
    assert set(first.metric) == {
        "Overall_ARI", "Overall_B3_F1", "Known_R1", "Known_mAP",
        "Known_E2E_Accuracy", "Novelty_AUROC",
    }
    assert first[["estimate", "ci_lower", "ci_upper"]].eq(1.0).all().all()
    assert first.valid_replicates.eq(40).all()
    assert first.reliable.all()
    assert first.bootstrap_version.eq(BOOTSTRAP_VERSION).all()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"confidence_level": 0.0}, "confidence level"),
        ({"confidence_level": 1.0}, "confidence level"),
        ({"method": "basic"}, "method"),
        ({"minimum_valid_fraction": -0.1}, "minimum-valid"),
        ({"minimum_valid_fraction": 1.1}, "minimum-valid"),
    ],
)
def test_bootstrap_configuration_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        bootstrap_headlines(_bootstrap_frame(), 10, 1, **kwargs)


def test_bootstrap_reports_unreliable_single_class_auroc():
    frame = _bootstrap_frame().loc[lambda value: value.scenario.eq("known_identity")].copy()
    result = bootstrap_headlines(frame, 10, 2)
    row = result.set_index("metric").loc["Novelty_AUROC"]
    assert row.valid_replicates == 0
    assert row.invalid_replicates == 10
    assert not row.reliable
    assert np.isnan(row.ci_lower) and np.isnan(row.ci_upper)
