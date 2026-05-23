import torch

from apexid.models import ApexID, build_model
from apexid.models.backbones.osnet import osnet_x0_25


def test_apexid_accepts_explicit_six_channel_inputs():
    model = ApexID(
        backbone_name="osnet_x0_25",
        embedding_dim=16,
        input_channels=6,
    )
    model.eval()

    with torch.no_grad():
        embeddings = model(torch.randn(2, 6, 64, 64))

    assert model.input_channels == 6
    assert model.backbone.conv1.conv.in_channels == 6
    assert tuple(embeddings.shape) == (2, 16)
    assert torch.allclose(embeddings.norm(dim=1), torch.ones(2), atol=1e-5)


def test_build_model_auto_uses_wavelet_channel_count():
    cfg = {
        "seed": 42,
        "data": {"wavelet": True},
        "model": {
            "name": "apexid",
            "embedding_dim": 16,
            "input_channels": None,
            "backbone": {"name": "osnet_x0_25", "kwargs": {}},
        },
    }
    model = build_model(cfg)

    assert model.input_channels == 6
    assert model.backbone.conv1.conv.in_channels == 6


def test_build_model_defaults_to_rgb_channels_without_wavelet():
    cfg = {
        "seed": 42,
        "data": {"wavelet": False},
        "model": {
            "name": "apexid",
            "embedding_dim": 16,
            "backbone": {"name": "osnet_x0_25", "kwargs": {}},
        },
    }
    model = build_model(cfg)

    assert model.input_channels == 3
    assert model.backbone.conv1.conv.in_channels == 3


def test_osnet_embedding_projection_keeps_signed_features():
    model = osnet_x0_25(embedding_dim=16, input_channels=3)

    assert model.fc is not None
    assert not any(isinstance(layer, torch.nn.ReLU) for layer in model.fc)
