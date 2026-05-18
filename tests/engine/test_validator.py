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
