import pytest
import torch

from apexid.models import build_head
from apexid.models.heads.sphereface2 import SphereFace2Head
from apexid.models.heads.sphereface2_focal import SphereFace2FocalHead


def test_sphereface2_focal_head_extends_base_head_and_predicts_logits():
    head = SphereFace2FocalHead(
        in_features=3,
        out_features=4,
        r=16.0,
        m=0.2,
        t=2,
        lam=0.7,
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


def test_sphereface2_focal_gamma_zero_reuses_base_loss():
    torch.manual_seed(42)
    base = SphereFace2Head(
        in_features=3,
        out_features=4,
        r=16.0,
        m=0.2,
        t=2,
        lam=0.7,
    )
    focal = SphereFace2FocalHead(
        in_features=3,
        out_features=4,
        r=16.0,
        m=0.2,
        t=2,
        lam=0.7,
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


def test_sphereface2_focal_factory_aliases_and_alpha_alias():
    head = build_head(
        "sphereface2_focal_head",
        embedding_dim=3,
        num_classes=4,
        alpha=0.6,
        focal_gamma=2.0,
    )

    assert isinstance(head, SphereFace2FocalHead)
    assert head.lam == 0.6
    assert head.alpha == 0.6


def test_sphereface2_focal_rejects_non_base_margin_type():
    with pytest.raises(ValueError, match="magn_type='C'"):
        SphereFace2FocalHead(
            in_features=3,
            out_features=4,
            magn_type="A",
        )
