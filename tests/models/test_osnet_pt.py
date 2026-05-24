import torch

from apexid.models import ApexID, build_model
from apexid.models.backbones import build_backbone
from apexid.models.backbones import osnet_pt
from apexid.models.backbones.osnet_pt import WaveletStem, osnet_x0_25_pt


def _fake_pretrained_state_dict(model):
    state_dict = model.state_dict()
    state_dict["classifier.weight"] = torch.empty(1000, model.embedding_dim)
    state_dict["classifier.bias"] = torch.empty(1000)
    return state_dict


def _mock_init_pretrained_weights(model, key):
    reference = build_backbone(key, embedding_dim=model.embedding_dim, input_channels=3)
    state_dict = _fake_pretrained_state_dict(reference)
    model_dict = model.state_dict()
    filtered = {
        key_name: value
        for key_name, value in state_dict.items()
        if key_name in model_dict and model_dict[key_name].shape == value.shape
    }
    model.load_state_dict(filtered, strict=False)
    return {
        "cached_file": f"mock/{key}_imagenet.pth",
        "matched_layers": sorted(filtered.keys()),
        "discarded_layers": sorted(set(state_dict) - set(filtered)),
    }


def test_wavelet_stem_initializes_rgb_and_wavelet_passthrough():
    stem = WaveletStem()
    weight = stem.proj.weight.detach()
    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        ]
    )

    assert tuple(weight.shape) == (3, 6, 1, 1)
    assert torch.allclose(weight[:, :, 0, 0], expected)


def test_pretrained_osnet_keeps_three_channel_conv1_for_wavelet_input():
    model = osnet_x0_25_pt(embedding_dim=16, input_channels=6, pretrained=False)
    model.eval()

    with torch.no_grad():
        embeddings = model(torch.randn(2, 6, 64, 64))

    assert model.input_channels == 6
    assert isinstance(model.stem, WaveletStem)
    assert model.backbone.conv1.conv.in_channels == 3
    assert tuple(embeddings.shape) == (2, 16)


def test_pretrained_osnet_rgb_path_has_no_stem():
    model = osnet_x0_25_pt(embedding_dim=16, input_channels=3, pretrained=False)

    assert model.input_channels == 3
    assert isinstance(model.stem, torch.nn.Identity)
    assert model.backbone.conv1.conv.in_channels == 3


def test_init_pretrained_weights_loads_full_backbone_for_three_channel_input(monkeypatch):
    monkeypatch.setattr(osnet_pt, "init_pretrained_weights", _mock_init_pretrained_weights)

    model = osnet_x0_25_pt(embedding_dim=512, input_channels=3, pretrained=True)
    reference = build_backbone("osnet_x0_25", embedding_dim=512, input_channels=3)
    expected = _fake_pretrained_state_dict(reference)

    for key, value in model.backbone.state_dict().items():
        assert key in expected
        assert tuple(value.shape) == tuple(expected[key].shape)


def test_init_pretrained_weights_skips_fc_when_embedding_dim_differs(monkeypatch):
    monkeypatch.setattr(osnet_pt, "init_pretrained_weights", _mock_init_pretrained_weights)

    model = osnet_x0_25_pt(embedding_dim=256, input_channels=3, pretrained=True)
    loaded = model.backbone.state_dict()

    assert loaded["conv1.conv.weight"].shape == (16, 3, 7, 7)
    assert loaded["fc.0.weight"].shape == (256, 128)


def test_apexid_accepts_pretrained_backbone_with_wavelet_channels():
    model = ApexID(
        backbone_name="osnet_x0_25_pt",
        embedding_dim=16,
        input_channels=6,
        backbone_kwargs={"pretrained": False},
    )
    model.eval()

    with torch.no_grad():
        embeddings = model(torch.randn(2, 6, 64, 64))

    assert model.input_channels == 6
    assert model.backbone.backbone.conv1.conv.in_channels == 3
    assert tuple(embeddings.shape) == (2, 16)
    assert torch.allclose(embeddings.norm(dim=1), torch.ones(2), atol=1e-5)


def test_build_model_auto_uses_pretrained_backbone_with_wavelet_channels():
    cfg = {
        "seed": 42,
        "data": {"wavelet": True},
        "model": {
            "name": "apexid",
            "embedding_dim": 16,
            "input_channels": None,
            "backbone": {
                "name": "osnet_x0_25_pt",
                "kwargs": {"pretrained": False},
            },
        },
    }
    model = build_model(cfg)

    assert model.input_channels == 6
    assert model.backbone.backbone.conv1.conv.in_channels == 3
