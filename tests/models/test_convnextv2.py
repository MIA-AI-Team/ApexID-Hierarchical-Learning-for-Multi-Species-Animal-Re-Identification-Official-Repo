from types import SimpleNamespace
import hashlib

import pytest
import torch
from torch import nn

from apexid.models import ApexID
from apexid.models.backbones import available_backbones, build_backbone
from apexid.models.backbones.convnextv2 import ConvNeXtV2Backbone


class _FakeConvNeXtV2(nn.Module):
    def __init__(self, hidden_size=12):
        super().__init__()
        self.config = SimpleNamespace(hidden_sizes=[4, 8, hidden_size])
        self.encoder = nn.Conv2d(3, hidden_size, kernel_size=1)

    def forward(self, *, pixel_values):
        features = self.encoder(pixel_values).mean(dim=(-2, -1))
        return SimpleNamespace(pooler_output=features)


def _fake_transformers(calls):
    class FakeAutoConfig:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append(("config", model_name, kwargs))
            return SimpleNamespace(hidden_sizes=[4, 8, 12])

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append(("pretrained", model_name, kwargs))
            return _FakeConvNeXtV2()

        @classmethod
        def from_config(cls, config):
            calls.append(("random", config, {}))
            return _FakeConvNeXtV2(hidden_size=config.hidden_sizes[-1])

    return FakeAutoConfig, FakeAutoModel


def test_convnextv2_backbones_are_registered():
    names = set(available_backbones())

    assert {
        "convnextv2_tiny",
        "convnextv2_base",
        "convnextv2_large",
        "convnextv2_tiny_pt",
        "convnextv2_base_pt",
        "convnextv2_large_pt",
    }.issubset(names)


def test_convnextv2_base_pt_loads_hf_weights_and_uses_identity(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "apexid.models.backbones.convnextv2._transformers_classes",
        lambda: _fake_transformers(calls),
    )

    model = build_backbone(
        "convnextv2_base_pt",
        embedding_dim=12,
        input_channels=3,
        local_files_only=True,
    )
    model.eval()

    with torch.no_grad():
        embeddings = model(torch.randn(2, 3, 32, 32))

    assert isinstance(model, ConvNeXtV2Backbone)
    assert isinstance(model.projection, nn.Identity)
    assert tuple(embeddings.shape) == (2, 12)
    assert calls == [
        (
            "pretrained",
            "facebook/convnextv2-base-22k-384",
            {"local_files_only": True},
        )
    ]


def test_convnextv2_projects_when_embedding_dim_differs(monkeypatch):
    monkeypatch.setattr(
        "apexid.models.backbones.convnextv2._transformers_classes",
        lambda: _fake_transformers([]),
    )

    model = build_backbone(
        "convnextv2_base_pt",
        embedding_dim=16,
        input_channels=3,
    )
    model.eval()

    with torch.no_grad():
        embeddings = model(torch.randn(2, 3, 32, 32))

    assert isinstance(model.projection, nn.Linear)
    assert model.projection.in_features == 12
    assert model.projection.out_features == 16
    assert tuple(embeddings.shape) == (2, 16)


def test_convnextv2_model_name_override(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "apexid.models.backbones.convnextv2._transformers_classes",
        lambda: _fake_transformers(calls),
    )

    model = build_backbone(
        "convnextv2_tiny_pt",
        embedding_dim=12,
        model_name="facebook/convnextv2-tiny-1k-224",
    )

    assert model.model_name == "facebook/convnextv2-tiny-1k-224"
    assert calls[0][1] == "facebook/convnextv2-tiny-1k-224"


def test_apexid_normalizes_convnextv2_embeddings(monkeypatch):
    monkeypatch.setattr(
        "apexid.models.backbones.convnextv2._transformers_classes",
        lambda: _fake_transformers([]),
    )
    model = ApexID(
        backbone_name="convnextv2_base_pt",
        embedding_dim=12,
        input_channels=3,
    )
    model.eval()

    with torch.no_grad():
        embeddings = model(torch.randn(2, 3, 32, 32))

    assert tuple(embeddings.shape) == (2, 12)
    assert torch.allclose(embeddings.norm(dim=1), torch.ones(2), atol=1e-5)


def test_convnextv2_pinned_revision_and_hash_are_verified(tmp_path, monkeypatch):
    calls = []
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"verified-convnextv2-weight")
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    monkeypatch.setattr(
        "apexid.models.backbones.convnextv2._transformers_classes",
        lambda: _fake_transformers(calls),
    )
    monkeypatch.setattr(
        "apexid.models.backbones.convnextv2._hf_hub_download",
        lambda **kwargs: weight,
    )
    model = build_backbone(
        "convnextv2_base_pt",
        embedding_dim=12,
        revision="immutable",
        expected_sha256=digest,
    )
    assert calls[0][2]["revision"] == "immutable"
    assert model.pretrained_provenance["sha256"] == digest


def test_convnextv2_rejects_weight_hash_mismatch(tmp_path, monkeypatch):
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"wrong")
    monkeypatch.setattr(
        "apexid.models.backbones.convnextv2._transformers_classes",
        lambda: _fake_transformers([]),
    )
    monkeypatch.setattr(
        "apexid.models.backbones.convnextv2._hf_hub_download",
        lambda **kwargs: weight,
    )
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        build_backbone(
            "convnextv2_base_pt",
            embedding_dim=12,
            revision="immutable",
            expected_sha256="0" * 64,
        )
