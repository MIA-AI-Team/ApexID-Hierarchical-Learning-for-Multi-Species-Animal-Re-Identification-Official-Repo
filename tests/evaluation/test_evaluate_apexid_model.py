import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from apexid.evaluation.benchmark_evaluator import (
    EvaluationConfig,
    assert_prediction_integrity,
    build_gallery,
    cluster_rejected,
    evaluate_final,
    infer_batch,
    infer_online,
    l2_normalize,
    score_gallery,
    tune_discovery_threshold,
    tune_known_threshold,
)
from apexid.evaluation.benchmark_metrics import (
    b3_metrics,
    cluster_count_error,
    clustering_metrics,
    fragmentation_metrics,
    merge_metrics,
    pairwise_metrics,
)


def _gallery():
    database = pd.DataFrame(
        {
            "path": ["a", "b", "c", "d"],
            "database_id": ["DB_A", "DB_A", "DB_B", "DB_B"],
            "species": ["animal"] * 4,
            "dataset": ["set"] * 4,
            "encounter_id": ["a1", "a2", "b1", "b2"],
        }
    )
    embeddings = np.asarray([[1, 0], [.98, .02], [0, 1], [.02, .98]], dtype=np.float32)
    return build_gallery(database, embeddings)


def _config(tmp_path):
    return EvaluationConfig(
        weights=tmp_path / "weights.pt", benchmark_dir=tmp_path / "benchmark", data_root=tmp_path,
        output_dir=tmp_path / "output", bootstrap_samples=0,
    )


def test_l2_normalization_and_centroid_retrieval(tmp_path):
    gallery = _gallery()
    normalized = l2_normalize(np.asarray([[3.0, 4.0]], dtype=np.float32))
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1.0)
    scores, ranks, ranked_scores = score_gallery(np.asarray([[1.0, 0.0]], dtype=np.float32), gallery)
    assert gallery.identity_ids[ranks[0, 0]] == "DB_A"
    assert ranked_scores[0, 0] > ranked_scores[0, 1]


def test_known_threshold_uses_development_labels_and_score_direction():
    scores = np.asarray([.95, .85, .2, .1])
    scenarios = ["known_identity", "known_identity", "novel_identity_known_species", "novel_identity_known_species"]
    threshold, curve = tune_known_threshold(scores, scenarios, "f1")
    assert .2 < threshold <= .85
    assert curve.loc[curve.threshold.eq(threshold), "FPR"].iloc[0] == 0.0


def test_label_invariant_cluster_metrics_and_diagnostics():
    true = ["a", "a", "b", "b"]
    predicted = ["NEW_7", "NEW_7", "NEW_1", "NEW_1"]
    assert clustering_metrics(true, predicted)["ari"] == 1.0
    assert b3_metrics(true, predicted)["b3_f1"] == 1.0
    assert pairwise_metrics(true, predicted)["pairwise_f1"] == 1.0
    assert fragmentation_metrics(true, ["x", "y", "z", "z"])["fraction_true_identities_fragmented"] == 0.5
    assert merge_metrics(true, ["x", "x", "x", "x"])["fraction_predicted_clusters_merged"] == 1.0
    assert cluster_count_error(true, ["x", "x", "x", "x"])["absolute_cluster_count_error"] == 1.0


def test_batch_inference_and_missing_prediction_detection(tmp_path):
    gallery = _gallery()
    public = pd.DataFrame(
        {
            "image_id": ["q1", "q2", "q3"], "arrival_index": [1, 2, 3],
            "path": ["q1", "q2", "q3"], "species": ["animal"] * 3, "dataset": ["set"] * 3,
        }
    )
    embeddings = np.asarray([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)
    predictions, retrieval, _ = infer_batch(public, embeddings, gallery, {"known_similarity_threshold": .8, "clustering_threshold": .7}, _config(tmp_path))
    assert predictions.loc[0, "predicted_identity"] == "DB_A"
    assert predictions.loc[2, "predicted_status"] == "novel"
    assert retrieval.groupby("image_id").size().eq(2).all()
    with pytest.raises(ValueError, match="exactly once"):
        assert_prediction_integrity(predictions.iloc[:2], public, gallery)


def test_online_discovery_reassociates_and_resets_between_streams(tmp_path):
    gallery = _gallery()
    stream = pd.DataFrame(
        {"image_id": ["q1", "q2", "q3"], "arrival_index": [1, 2, 3], "path": ["q1", "q2", "q3"], "species": ["x"] * 3, "dataset": ["x"] * 3}
    )
    vectors = {"q1": np.asarray([-1.0, 0.0]), "q2": np.asarray([-.99, .01]), "q3": np.asarray([-1.0, .02])}
    thresholds = {"known_similarity_threshold": .8, "discovered_identity_threshold": .8}
    first, _ = infer_online(stream, vectors, gallery, thresholds, _config(tmp_path))
    second, _ = infer_online(stream, vectors, gallery, thresholds, _config(tmp_path))
    assert first.predicted_identity.tolist() == second.predicted_identity.tolist()
    assert first.created_new_identity.tolist() == [True, False, False]


def test_final_ground_truth_is_outside_inference_interface(tmp_path):
    assert "ground_truth" not in inspect.signature(infer_batch).parameters
    gallery = _gallery()
    predictions = pd.DataFrame(
        {
            "image_id": ["q1", "q2"], "arrival_index": [1, 2], "predicted_identity": ["DB_A", "NEW_000001"],
            "predicted_status": ["known", "novel"], "top1_database_id": ["DB_A", "DB_A"],
            "top1_similarity": [.99, .1], "top2_database_id": ["DB_B", "DB_B"], "top2_similarity": [.1, .0],
            "novelty_score": [.01, .9], "matched_discovered_id": ["", "NEW_000001"],
            "discovered_similarity": [np.nan, np.nan], "created_new_identity": [False, True],
        }
    )
    gt_path = tmp_path / "final_test_ground_truth.csv"
    pd.DataFrame(
        {
            "image_id": ["q1", "q2"], "true_identity": ["DB_A", "GT_NOVEL_1"], "original_identity": ["a", "n"],
            "scenario": ["known_identity", "novel_identity_known_species"], "species": ["x", "x"], "dataset": ["x", "x"],
            "known_database_id": ["DB_A", np.nan], "encounter_id": ["1", "2"],
        }
    ).to_csv(gt_path, index=False)
    embeddings = {"q1": np.asarray([1.0, 0.0]), "q2": np.asarray([-1.0, 0.0])}
    result, merged = evaluate_final(predictions, gt_path, embeddings, gallery)
    assert result["overall"]["ari"] == 1.0
    assert result["novelty_detection"]["Novelty_AUROC"] == 1.0
    assert len(merged) == 2


def test_discovery_threshold_and_clustering_are_deterministic():
    embeddings = np.asarray([[-1, 0], [-.98, .02], [0, -1], [.02, -.98]], dtype=np.float32)
    labels = ["a", "a", "b", "b"]
    threshold_a, _ = tune_discovery_threshold(embeddings, labels, "ari", 42, permutations=2)
    threshold_b, _ = tune_discovery_threshold(embeddings, labels, "ari", 42, permutations=2)
    assert threshold_a == threshold_b
    clusters = cluster_rejected(embeddings, threshold_a)
    assert len(clusters) == len(labels)
