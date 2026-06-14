import torch
from torchvision import models

from apexid.models import ApexID, build_model
from apexid.models.backbones import available_backbones, build_backbone
from apexid.models.backbones.convnext import ConvNeXtBackbone
from apexid.models.backbones.osnet_pt import WaveletStem


def test_convnext_backbones_are_registered():
    names = set(available_backbones())

    assert {
        "convnext_tiny",
        "convnext_small",
        "convnext_base",
        "convnext_large",
        "convnext_tiny_pt",
        "convnext_small_pt",
        "convnext_base_pt",
        "convnext_large_pt",
    }.issubset(names)


def test_convnext_random_forward_returns_embedding_shape():
    model = build_backbone("convnext_tiny", embedding_dim=16, input_channels=3)
    model.eval()

    with torch.no_grad():
        embeddings = model(torch.randn(2, 3, 64, 64))

    assert isinstance(model, ConvNeXtBackbone)
    assert tuple(embeddings.shape) == (2, 16)
    assert model.embedding_dim == 16


def test_apexid_normalizes_convnext_embeddings():
    model = ApexID(
        backbone_name="convnext_tiny",
        embedding_dim=16,
        input_channels=3,
    )
    model.eval()

    with torch.no_grad():
        embeddings = model(torch.randn(2, 3, 64, 64))

    assert tuple(embeddings.shape) == (2, 16)
    assert torch.allclose(embeddings.norm(dim=1), torch.ones(2), atol=1e-5)


def test_convnext_random_replaces_first_conv_for_six_channel_input():
    model = build_backbone("convnext_tiny", embedding_dim=16, input_channels=6)
    model.eval()

    with torch.no_grad():
        embeddings = model(torch.randn(2, 6, 64, 64))

    assert model.input_channels == 6
    assert isinstance(model.stem, torch.nn.Identity)
    assert model.model.features[0][0].in_channels == 6
    assert tuple(embeddings.shape) == (2, 16)


def test_convnext_pretrained_wavelet_input_uses_adapter(monkeypatch):
    original_builder = models.convnext_tiny
    calls = {}

    def fake_builder(*, weights=None, progress=True, **kwargs):
        calls["weights"] = weights
        calls["progress"] = progress
        calls["kwargs"] = kwargs
        return original_builder(weights=None)

    monkeypatch.setattr("apexid.models.backbones.convnext.models.convnext_tiny", fake_builder)

    model = build_backbone(
        "convnext_tiny",
        embedding_dim=16,
        input_channels=6,
        pretrained=True,
        progress=False,
    )
    model.eval()

    with torch.no_grad():
        embeddings = model(torch.randn(2, 6, 64, 64))

    assert calls["weights"] is models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    assert calls["progress"] is False
    assert isinstance(model.stem, WaveletStem)
    assert model.model.features[0][0].in_channels == 3
    assert tuple(embeddings.shape) == (2, 16)


def test_convnext_pt_alias_uses_pretrained_by_default(monkeypatch):
    original_builder = models.convnext_tiny
    calls = {}

    def fake_builder(*, weights=None, progress=True, **kwargs):
        calls["weights"] = weights
        return original_builder(weights=None)

    monkeypatch.setattr("apexid.models.backbones.convnext.models.convnext_tiny", fake_builder)

    model = build_backbone("convnext_tiny_pt", embedding_dim=16, input_channels=3)

    assert calls["weights"] is models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    assert model.model.features[0][0].in_channels == 3


def test_build_model_accepts_convnext_backbone_from_config():
    cfg = {
        "seed": 42,
        "data": {"wavelet": False},
        "model": {
            "name": "apexid",
            "embedding_dim": 16,
            "input_channels": 3,
            "backbone": {"name": "convnext_tiny", "kwargs": {}},
        },
    }

    model = build_model(cfg)

    assert model.backbone_name == "convnext_tiny"
    assert isinstance(model.backbone, ConvNeXtBackbone)
