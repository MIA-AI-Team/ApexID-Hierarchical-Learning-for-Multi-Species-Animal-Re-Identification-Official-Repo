import pytest
import torch

from apexid.models import build_head
from apexid.models.heads import SphereFace2FocalHead, SphereFace2Head
from apexid.losses.sphereface2 import SphereFace2


def test_sphereface2_factory_builds_adapter_and_predicts_logits():
    head = build_head(
        "sphereface2",
        embedding_dim=3,
        num_classes=4,
        alpha=0.6,
    )
    embeddings = torch.randn(5, 3)
    labels = torch.tensor([0, 1, 2, 3, 1])

    loss = head(embeddings, labels)
    logits = head.predict_logits(embeddings)

    assert isinstance(head, SphereFace2Head)
    assert isinstance(head, SphereFace2)
    assert head.alpha == 0.6
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert tuple(logits.shape) == (5, 4)


@pytest.mark.parametrize("magn_type", ["C", "A", "M"])
def test_sphereface2_focal_head_extends_base_head_and_predicts_logits(magn_type):
    head = SphereFace2FocalHead(
        in_features=3,
        out_features=4,
        magn_type=magn_type,
        r=16.0,
        m=0.2,
        t=2.0,
        alpha=0.7,
        lw=20.0,
        focal_gamma=2.0,
    )
    embeddings = torch.randn(5, 3)
    labels = torch.tensor([0, 1, 2, 3, 1])

    loss = head(embeddings, labels)
    logits = head.predict_logits(embeddings)

    assert isinstance(head, SphereFace2Head)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert tuple(logits.shape) == (5, 4)


@pytest.mark.parametrize("magn_type", ["C", "A", "M"])
def test_sphereface2_focal_gamma_zero_reuses_base_loss(magn_type):
    torch.manual_seed(42)
    base = SphereFace2Head(
        in_features=3,
        out_features=4,
        magn_type=magn_type,
        r=16.0,
        m=0.2,
        t=2.0,
        alpha=0.7,
        lw=20.0,
    )
    focal = SphereFace2FocalHead(
        in_features=3,
        out_features=4,
        magn_type=magn_type,
        r=16.0,
        m=0.2,
        t=2.0,
        alpha=0.7,
        lw=20.0,
        focal_gamma=0.0,
    )
    focal.load_state_dict(base.state_dict())
    embeddings = torch.randn(5, 3)
    labels = torch.tensor([0, 1, 2, 3, 1])

    assert torch.allclose(
        focal(embeddings, labels),
        base(embeddings, labels),
        atol=1e-7,
    )


def test_sphereface2_focal_factory_aliases_use_alpha_only():
    head = build_head(
        "sphereface2_focal_head",
        embedding_dim=3,
        num_classes=4,
        alpha=0.6,
        focal_gamma=2.0,
    )

    assert isinstance(head, SphereFace2FocalHead)
    assert head.alpha == 0.6


def test_sphereface2_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="in_features"):
        SphereFace2Head(in_features=0, out_features=4)

    with pytest.raises(ValueError, match="out_features"):
        SphereFace2Head(in_features=3, out_features=1)

    with pytest.raises(ValueError, match="magn_type"):
        SphereFace2FocalHead(in_features=3, out_features=4, magn_type="bad")

    with pytest.raises(ValueError, match="focal_gamma"):
        SphereFace2FocalHead(in_features=3, out_features=4, focal_gamma=-1.0)
