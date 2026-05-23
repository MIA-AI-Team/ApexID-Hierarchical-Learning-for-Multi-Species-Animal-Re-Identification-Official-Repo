import math

import numpy as np

from apexid.evaluation import (
    compute_dir_at_far,
    compute_embedding_diagnostics,
    compute_known_retrieval_metrics,
    compute_mixed_open_set_metrics,
    compute_oscr_auc,
    compute_rejection_metrics,
    compute_unknown_clustering_metrics,
    evaluate_all_validation_splits,
    norm_accuracy,
    should_run_diagnostics,
)


def _toy_gallery():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.0, 1.0],
            [0.05, 0.95],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1])
    return embeddings, labels


def _toy_known():
    embeddings = np.array(
        [
            [0.98, 0.02],
            [0.02, 0.98],
            [0.90, 0.10],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 1, 0])
    cls_labels = np.array([0, 1, 0])
    return embeddings, labels, cls_labels


def _toy_unknown():
    embeddings = np.array(
        [
            [-1.0, 0.0],
            [-0.95, 0.05],
            [0.0, -1.0],
            [0.05, -0.95],
        ],
        dtype=np.float32,
    )
    labels = np.array([2, 2, 3, 3])
    cls_labels = np.array([-100, -100, -100, -100])
    return embeddings, labels, cls_labels


def _toy_mixed():
    known_embeddings, known_labels, known_cls = _toy_known()
    unknown_embeddings, unknown_labels, unknown_cls = _toy_unknown()
    return (
        np.vstack([known_embeddings, unknown_embeddings]),
        np.concatenate([known_labels, unknown_labels]),
        np.concatenate([known_cls, unknown_cls]),
    )


def test_known_retrieval_metrics_use_train_gallery():
    gallery_embeddings, gallery_labels = _toy_gallery()
    known_embeddings, known_labels, _ = _toy_known()

    metrics = compute_known_retrieval_metrics(
        known_embeddings,
        known_labels,
        gallery_embeddings,
        gallery_labels,
        top_k=(1, 5, 10),
        cmc_ranks=(1, 5, 10, 20, 50),
        chunk_size=2,
    )

    assert metrics["known/top1"] == 1.0
    assert metrics["known/top5"] == 1.0
    assert metrics["known/top10"] == 1.0
    assert metrics["known/mAP"] == 1.0
    assert metrics["known/BAKS"] == 1.0
    assert metrics["known/cmc_rank1"] == 1.0
    assert metrics["known/cmc_rank50"] == 1.0


def test_unknown_clustering_metrics_and_baus_use_true_unknown_labels_only_for_eval():
    unknown_embeddings, unknown_labels, _ = _toy_unknown()

    metrics = compute_unknown_clustering_metrics(
        unknown_embeddings,
        unknown_labels,
        threshold=0.8,
        knownness_scores=np.array([0.1, 0.2, 0.15, 0.05]),
        known_unknown_threshold=0.5,
        chunk_size=2,
    )

    assert metrics["unknown/ARI"] == 1.0
    assert metrics["unknown/AMI"] == 1.0
    assert metrics["unknown/v_measure"] == 1.0
    assert metrics["unknown/BAUS"] == 1.0


def test_rejection_metrics_far_auroc_and_fpr95():
    metrics = compute_rejection_metrics(
        known_scores=np.array([0.95, 0.90, 0.80]),
        unknown_scores=np.array([0.10, 0.20, 0.30]),
        threshold=0.5,
        target_tpr=0.95,
    )

    assert metrics["rejection/AUROC"] == 1.0
    assert metrics["rejection/FAR"] == 0.0
    assert metrics["rejection/FPR95"] == 0.0
    assert metrics["rejection/threshold_at_95_tpr"] == 0.80


def test_mixed_normacc_oscr_and_dir_metrics_are_correct_on_separable_toy_data():
    gallery_embeddings, gallery_labels = _toy_gallery()
    mixed_embeddings, mixed_labels, mixed_cls = _toy_mixed()
    true_known_mask = mixed_cls != -100

    metrics = compute_mixed_open_set_metrics(
        mixed_embeddings,
        mixed_labels,
        true_known_mask,
        gallery_embeddings,
        gallery_labels,
        known_unknown_threshold=0.5,
        clustering_threshold=0.8,
        dir_far_levels=(0.01, 0.001),
        chunk_size=2,
    )

    assert metrics["mixed/ARI"] == 1.0
    assert metrics["mixed/F1"] == 1.0
    assert metrics["mixed/BAKS"] == 1.0
    assert metrics["mixed/BAUS"] == 1.0
    assert metrics["mixed/NormAcc"] == 1.0
    assert metrics["mixed/OSCR_AUC"] == 1.0
    assert metrics["mixed/DIR_at_1FAR"] == 1.0
    assert metrics["mixed/DIR_at_0_1FAR"] == 1.0
    assert norm_accuracy(0.25, 1.0) == 0.5


def test_oscr_and_dir_direct_helpers():
    known_scores = np.array([0.9, 0.8, 0.7])
    unknown_scores = np.array([0.1, 0.2])
    known_correct = np.array([True, True, False])

    assert compute_oscr_auc(known_scores, unknown_scores, known_correct) == 2 / 3
    assert compute_dir_at_far(known_scores, unknown_scores, known_correct, far_level=0.0) == 2 / 3


def test_evaluate_all_validation_splits_skips_and_runs_diagnostics_by_one_based_epoch():
    gallery_embeddings, gallery_labels = _toy_gallery()
    known_embeddings, known_labels, known_cls = _toy_known()
    unknown_embeddings, unknown_labels, unknown_cls = _toy_unknown()
    mixed_embeddings, mixed_labels, mixed_cls = _toy_mixed()
    cfg = {
        "known_unknown_threshold": 0.5,
        "diagnostics_every_n_epochs": 5,
        "diagnostics": {"enabled": True, "max_pairs": 20},
        "clustering": {"method": "threshold_connected_components", "threshold": 0.8},
    }

    skipped = evaluate_all_validation_splits(
        gallery_embeddings,
        gallery_labels,
        known_embeddings,
        known_labels,
        unknown_embeddings,
        unknown_labels,
        mixed_embeddings,
        mixed_labels,
        known_cls_labels=known_cls,
        unknown_cls_labels=unknown_cls,
        mixed_cls_labels=mixed_cls,
        config=cfg,
        epoch_index=3,
        chunk_size=2,
    )
    ran = evaluate_all_validation_splits(
        gallery_embeddings,
        gallery_labels,
        known_embeddings,
        known_labels,
        unknown_embeddings,
        unknown_labels,
        mixed_embeddings,
        mixed_labels,
        known_cls_labels=known_cls,
        unknown_cls_labels=unknown_cls,
        mixed_cls_labels=mixed_cls,
        config=cfg,
        epoch_index=4,
        chunk_size=2,
    )

    assert should_run_diagnostics(3, 5) is False
    assert should_run_diagnostics(4, 5) is True
    assert skipped["validation/diagnostics_ran"] == 0
    assert "diagnostics/intra_similarity" not in skipped
    assert ran["validation/diagnostics_ran"] == 1
    assert math.isfinite(ran["diagnostics/embedding_norm_mean"])
    assert math.isfinite(ran["diagnostics/knn_top1"])


def test_diagnostics_and_metrics_do_not_crash_with_singleton_identities():
    embeddings = np.eye(3, dtype=np.float32)
    labels = np.array([0, 1, 2])

    diagnostics = compute_embedding_diagnostics(
        embeddings,
        labels,
        max_pairs=10,
        random_state=42,
    )
    known = compute_known_retrieval_metrics(
        embeddings[:1],
        labels[:1],
        embeddings[1:],
        labels[1:],
        top_k=(1, 5, 10),
        cmc_ranks=(1,),
    )

    assert math.isnan(diagnostics["diagnostics/intra_similarity"])
    assert math.isnan(diagnostics["diagnostics/alignment"])
    assert math.isnan(known["known/mAP"])
