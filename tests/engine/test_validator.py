import math

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from apexid.engine.validator import Validator, _compute_loss_and_logits
from apexid.models.heads import SphereFace2Head


class LogitHead(nn.Module):
    def forward(self, embeddings, labels):
        return embeddings[:, :2]

    def predict_logits(self, embeddings):
        return embeddings[:, :2]


def test_validator_computes_known_loss_accuracy_and_reid_metrics():
    images = torch.tensor(
        [
            [5.0, 0.0],
            [4.0, 0.1],
            [0.0, 5.0],
            [0.1, 4.0],
        ]
    )
    labels = torch.tensor([0, 0, 1, 1])
    loader = DataLoader(TensorDataset(images, labels, labels), batch_size=2)

    validator = Validator(
        nn.Identity(),
        LogitHead(),
        loss=nn.CrossEntropyLoss(),
        device="cpu",
        metric_chunk_size=2,
        progress=False,
    )
    result = validator.validate(loader, return_embeddings=True)

    assert result["num_known"] == 4
    assert result["num_samples"] == 4
    assert result["acc"] == 1.0
    assert result["top1"] == 1.0
    assert result["top5"] == 1.0
    assert result["mAP"] == 1.0
    assert result["micro_f1"] == 1.0
    assert tuple(result["embeddings"].shape) == (4, 2)
    assert tuple(result["metric_labels"].shape) == (4,)
    assert result["loss"] > 0.0


def test_validator_ignores_unknown_classification_labels_for_loss_and_acc():
    images = torch.tensor(
        [
            [5.0, 0.0],
            [4.0, 0.1],
            [0.0, 5.0],
            [0.1, 4.0],
            [-5.0, 0.0],
        ]
    )
    cls_labels = torch.tensor([0, 0, 1, 1, -100])
    metric_labels = torch.tensor([0, 0, 1, 1, 2])
    loader = DataLoader(TensorDataset(images, cls_labels, metric_labels), batch_size=5)

    validator = Validator(
        nn.Identity(),
        LogitHead(),
        loss=nn.CrossEntropyLoss(),
        device="cpu",
        classification_ignore_index=-100,
        metric_ignore_index=None,
        progress=False,
    )
    result = validator.validate(loader)

    assert result["num_known"] == 4
    assert result["num_samples"] == 5
    assert result["acc"] == 1.0
    assert result["top1"] == 1.0
    assert result["top5"] == 1.0
    assert result["mAP"] == 1.0
    assert not math.isnan(result["micro_f1"])


def test_validator_returns_nan_loss_and_acc_when_no_known_samples():
    images = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    cls_labels = torch.tensor([-100, -100])
    metric_labels = torch.tensor([0, 0])
    loader = DataLoader(TensorDataset(images, cls_labels, metric_labels), batch_size=2)

    validator = Validator(
        nn.Identity(),
        LogitHead(),
        loss=nn.CrossEntropyLoss(),
        device="cpu",
        classification_ignore_index=-100,
        progress=False,
    )
    result = validator.validate(loader)

    assert result["num_known"] == 0
    assert result["num_samples"] == 2
    assert math.isnan(result["loss"])
    assert math.isnan(result["acc"])
    assert result["top1"] == 1.0


def test_validator_open_set_uses_gallery_known_unknown_and_mixed_loaders():
    gallery_images = torch.tensor(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.0, 1.0],
            [0.05, 0.95],
        ]
    )
    gallery_labels = torch.tensor([0, 0, 1, 1])
    known_images = torch.tensor([[0.98, 0.02], [0.02, 0.98]])
    known_cls = torch.tensor([0, 1])
    known_metric = torch.tensor([0, 1])
    unknown_images = torch.tensor([[-1.0, 0.0], [0.0, -1.0]])
    unknown_cls = torch.tensor([-100, -100])
    unknown_metric = torch.tensor([2, 3])
    mixed_images = torch.cat([known_images, unknown_images], dim=0)
    mixed_cls = torch.cat([known_cls, unknown_cls], dim=0)
    mixed_metric = torch.cat([known_metric, unknown_metric], dim=0)

    validator = Validator(
        nn.Identity(),
        LogitHead(),
        loss=nn.CrossEntropyLoss(),
        device="cpu",
        classification_ignore_index=-100,
        metric_chunk_size=2,
        progress=False,
    )
    result = validator.validate_open_set(
        DataLoader(TensorDataset(gallery_images, gallery_labels, gallery_labels), batch_size=2),
        DataLoader(TensorDataset(known_images, known_cls, known_metric), batch_size=2),
        DataLoader(TensorDataset(unknown_images, unknown_cls, unknown_metric), batch_size=2),
        DataLoader(TensorDataset(mixed_images, mixed_cls, mixed_metric), batch_size=2),
        evaluation_config={
            "known_unknown_threshold": 0.5,
            "diagnostics_every_n_epochs": 5,
            "diagnostics": {"enabled": True, "max_pairs": 20},
            "clustering": {"method": "threshold_connected_components", "threshold": 0.8},
        },
        epoch=4,
        epochs=5,
        return_embeddings=True,
    )

    assert result["num_known"] == 2
    assert result["num_samples"] == 4
    assert result["known/mAP"] == 1.0
    assert result["unknown/ARI"] == 1.0
    assert result["mixed/NormAcc"] == 1.0
    assert result["rejection/AUROC"] == 1.0
    assert result["validation/diagnostics_ran"] == 1
    assert result["diagnostics/knn_top1"] == 1.0
    assert result["mAP"] == result["known/mAP"]
    assert result["micro_f1"] == result["mixed/F1"]
    assert tuple(result["embeddings"].shape) == (4, 2)


def test_compute_loss_and_logits_supports_sphereface2_internal_loss():
    torch.manual_seed(42)
    head = SphereFace2Head(
        in_features=2,
        out_features=2,
        alpha=0.7,
        r=16.0,
        m=0.2,
        t=2.0,
        lw=20.0,
    )
    embeddings = torch.randn(4, 2)
    labels = torch.tensor([0, 0, 1, 1])

    loss, logits = _compute_loss_and_logits(head, None, embeddings, labels)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert tuple(logits.shape) == (4, 2)
