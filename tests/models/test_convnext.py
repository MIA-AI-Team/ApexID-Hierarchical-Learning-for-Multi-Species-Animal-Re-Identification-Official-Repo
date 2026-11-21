import torch
from torchvision import models

from apexid.models import ApexID, build_model
from apexid.models.backbones import available_backbones, build_backbone
from apexid.models.backbones.convnext import ConvNeXtBackbone


def test_convnext_backbones_are_registered():
    names = set(available_backbones())

    assert {"convnext_small", "convnext_small_pt"}.issubset(names)


def test_convnext_random_forward_returns_embedding_shape():
    model = build_backbone("convnext_small", embedding_dim=16, input_channels=3)
    model.eval()

    with torch.no_grad():
        embeddings = model(torch.randn(2, 3, 64, 64))

    assert isinstance(model, ConvNeXtBackbone)
    assert tuple(embeddings.shape) == (2, 16)
    assert model.embedding_dim == 16


def test_apexid_normalizes_convnext_embeddings():
    model = ApexID(
        backbone_name="convnext_small",
        embedding_dim=16,
        input_channels=3,
    )
    model.eval()

    with torch.no_grad():
        embeddings = model(torch.randn(2, 3, 64, 64))

    assert tuple(embeddings.shape) == (2, 16)
    assert torch.allclose(embeddings.norm(dim=1), torch.ones(2), atol=1e-5)


def test_convnext_pt_alias_uses_pretrained_by_default(monkeypatch):
    original_builder = models.convnext_small
    calls = {}

    def fake_builder(*, weights=None, progress=True, **kwargs):
        calls["weights"] = weights
        return original_builder(weights=None)

    monkeypatch.setattr("apexid.models.backbones.convnext.models.convnext_small", fake_builder)

    model = build_backbone("convnext_small_pt", embedding_dim=16, input_channels=3)

    assert calls["weights"] is models.ConvNeXt_Small_Weights.IMAGENET1K_V1
    assert model.model.features[0][0].in_channels == 3


def test_build_model_accepts_convnext_backbone_from_config():
    cfg = {
        "seed": 42,
        "data": {"wavelet": False},
        "model": {
            "name": "apexid",
            "embedding_dim": 16,
            "input_channels": 3,
            "backbone": {"name": "convnext_small", "kwargs": {}},
        },
    }

    model = build_model(cfg)

    assert model.backbone_name == "convnext_small"
    assert isinstance(model.backbone, ConvNeXtBackbone)
