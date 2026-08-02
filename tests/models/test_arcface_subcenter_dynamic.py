import torch

from apexid.models import build_head
from apexid.models.heads.arcface_subcenter_dynamic import (
    ArcFaceLossAdaptiveMargin,
    ArcFaceSubCenterDynamicHead,
    ArcMarginProductSubcenter,
)


def test_subcenter_projection_uses_max_over_subcenters():
    layer = ArcMarginProductSubcenter(in_features=2, out_features=2, k=2)
    with torch.no_grad():
        layer.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],   # class 0, subcenter 0
                    [0.0, 1.0],   # class 0, subcenter 1
                    [-1.0, 0.0],  # class 1, subcenter 0
                    [0.0, -1.0],  # class 1, subcenter 1
                ]
            )
        )

    features = torch.tensor([[1.0, 0.0]])
    logits = layer(features)

    assert tuple(logits.shape) == (1, 2)
    assert torch.allclose(logits[0], torch.tensor([1.0, 0.0]), atol=1e-6)


def test_adaptive_margin_applies_per_class_target_margin():
    loss_head = ArcFaceLossAdaptiveMargin(margins=[0.2, 0.5], out_dim=2, s=1.0)
    # Both rows have the same target cosine (0.8), but class 1 has larger margin.
    logits = torch.tensor([[0.8, 0.1], [0.1, 0.8]], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)
    adjusted = loss_head(logits, labels)

    assert tuple(adjusted.shape) == (2, 2)
    assert adjusted[1, 1] < adjusted[0, 0]


def test_subcenter_dynamic_head_factory_alias_builds_and_predicts():
    head = build_head(
        "arcface_subcenter_dynamic",
        embedding_dim=4,
        num_classes=3,
        s=16.0,
        k=2,
        margin_default=0.3,
    )
    embeddings = torch.randn(5, 4)
    labels = torch.tensor([0, 1, 2, 1, 0], dtype=torch.long)

    logits = head(embeddings, labels)
    pred_logits = head.predict_logits(embeddings)

    assert isinstance(head, ArcFaceSubCenterDynamicHead)
    assert tuple(logits.shape) == (5, 3)
    assert tuple(pred_logits.shape) == (5, 3)


def test_subcenter_dynamic_rejects_unsupported_margin_source_without_margins():
    try:
        ArcFaceSubCenterDynamicHead(
            in_features=4,
            out_features=3,
            margins_source="frequency_bins",
            margin_default=0.3,
        )
    except ValueError as exc:
        assert "Unsupported margins_source" in str(exc)
    else:
        raise AssertionError("Expected unsupported margins_source to raise ValueError.")
