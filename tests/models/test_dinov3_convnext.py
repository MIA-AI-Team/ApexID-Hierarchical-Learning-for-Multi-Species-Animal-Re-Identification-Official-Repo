from types import SimpleNamespace
import hashlib

import pytest
import torch
from torch import nn

from apexid.models import ApexID
from apexid.models.backbones import available_backbones, build_backbone
from apexid.models.backbones.dinov3_convnext import DINOv3ConvNeXtBackbone


class _FakeDINOv3Model(nn.Module):
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
            return _FakeDINOv3Model()

        @classmethod
        def from_config(cls, config):
            calls.append(("random", config, {}))
            return _FakeDINOv3Model(hidden_size=config.hidden_sizes[-1])

    return FakeAutoConfig, FakeAutoModel


def test_dinov3_convnext_backbones_are_registered():
    names = set(available_backbones())

    assert {
        "dinov3_convnext_tiny",
        "dinov3_convnext_small",
        "dinov3_convnext_base",
        "dinov3_convnext_large",
        "dinov3_convnext_tiny_pt",
        "dinov3_convnext_small_pt",
        "dinov3_convnext_base_pt",
        "dinov3_convnext_large_pt",
    }.issubset(names)


def test_dinov3_convnext_small_pt_loads_hf_weights_and_projects(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "apexid.models.backbones.dinov3_convnext._transformers_classes",
        lambda: _fake_transformers(calls),
    )

    model = build_backbone(
        "dinov3_convnext_small_pt",
        embedding_dim=16,
        input_channels=3,
        local_files_only=True,
    )
    model.eval()

    with torch.no_grad():
        embeddings = model(torch.randn(2, 3, 32, 32))

    assert isinstance(model, DINOv3ConvNeXtBackbone)
    assert tuple(embeddings.shape) == (2, 16)
    assert calls == [
        (
            "pretrained",
            "facebook/dinov3-convnext-small-pretrain-lvd1689m",
            {"local_files_only": True},
        )
    ]


def test_dinov3_convnext_model_name_override(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "apexid.models.backbones.dinov3_convnext._transformers_classes",
        lambda: _fake_transformers(calls),
    )

    model = build_backbone(
        "dinov3_convnext_tiny_pt",
        embedding_dim=7,
        model_name="owner/custom-dinov3",
    )

    assert model.model_name == "owner/custom-dinov3"
    assert calls[0][1] == "owner/custom-dinov3"


def test_apexid_normalizes_dinov3_convnext_embeddings(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "apexid.models.backbones.dinov3_convnext._transformers_classes",
        lambda: _fake_transformers(calls),
    )
    model = ApexID(
        backbone_name="dinov3_convnext_small_pt",
        embedding_dim=16,
        input_channels=3,
    )
    model.eval()

    with torch.no_grad():
        embeddings = model(torch.randn(2, 3, 32, 32))

    assert tuple(embeddings.shape) == (2, 16)
    assert torch.allclose(embeddings.norm(dim=1), torch.ones(2), atol=1e-5)


def test_dinov3_pinned_revision_and_hash_are_verified(tmp_path, monkeypatch):
    calls = []
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"verified-dino-weight")
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    monkeypatch.setattr(
        "apexid.models.backbones.dinov3_convnext._transformers_classes",
        lambda: _fake_transformers(calls),
    )
    monkeypatch.setattr(
        "apexid.models.backbones.dinov3_convnext._hf_hub_download",
        lambda **kwargs: weight,
    )
    model = build_backbone(
        "dinov3_convnext_small_pt", embedding_dim=4,
        revision="immutable", expected_sha256=digest,
    )
    assert calls[0][2]["revision"] == "immutable"
    assert model.pretrained_provenance["sha256"] == digest


def test_dinov3_rejects_weight_hash_mismatch(tmp_path, monkeypatch):
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"wrong")
    monkeypatch.setattr(
        "apexid.models.backbones.dinov3_convnext._transformers_classes",
        lambda: _fake_transformers([]),
    )
    monkeypatch.setattr(
        "apexid.models.backbones.dinov3_convnext._hf_hub_download",
        lambda **kwargs: weight,
    )
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        build_backbone(
            "dinov3_convnext_small_pt", embedding_dim=4,
            revision="immutable", expected_sha256="0" * 64,
        )
