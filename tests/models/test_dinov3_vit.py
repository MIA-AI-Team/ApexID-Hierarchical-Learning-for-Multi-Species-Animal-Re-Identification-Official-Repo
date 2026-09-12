from types import SimpleNamespace
import hashlib

import pytest
import torch
from torch import nn

from apexid.models import ApexID
from apexid.models.backbones import available_backbones, build_backbone
from apexid.models.backbones.dinov3_vit import DINOv3ViTBackbone


class _FakeDINOv3ViT(nn.Module):
    def __init__(self, hidden_size=12, patch_size=16, num_register_tokens=4):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            patch_size=patch_size,
            num_register_tokens=num_register_tokens,
        )

    def forward(self, *, pixel_values):
        batch_size = pixel_values.shape[0]
        hidden_size = self.config.hidden_size
        cls = pixel_values.new_ones(batch_size, hidden_size)
        registers = pixel_values.new_zeros(batch_size, self.config.num_register_tokens, hidden_size)
        patches = pixel_values.new_full((batch_size, 4, hidden_size), 3.0)
        last_hidden_state = torch.cat([cls.unsqueeze(1), registers, patches], dim=1)
        pooler_output = pixel_values.new_full((batch_size, hidden_size), 2.0)
        return SimpleNamespace(last_hidden_state=last_hidden_state, pooler_output=pooler_output)


def _fake_transformers(calls):
    class FakeAutoConfig:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append(("config", model_name, kwargs))
            return SimpleNamespace(hidden_size=12, patch_size=16, num_register_tokens=4)

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append(("pretrained", model_name, kwargs))
            return _FakeDINOv3ViT()

        @classmethod
        def from_config(cls, config):
            calls.append(("random", config, {}))
            return _FakeDINOv3ViT(hidden_size=config.hidden_size)

    return FakeAutoConfig, FakeAutoModel


def test_dinov3_vit_backbones_are_registered():
    names = set(available_backbones())

    assert {
        "dinov3_vits16",
        "dinov3_vits16plus",
        "dinov3_vitb16",
        "dinov3_vitl16",
        "dinov3_vits16_pt",
        "dinov3_vits16plus_pt",
        "dinov3_vitb16_pt",
        "dinov3_vitl16_pt",
    }.issubset(names)


def test_dinov3_vitb16_pt_loads_hf_weights_and_uses_identity(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "apexid.models.backbones.dinov3_vit._transformers_classes",
        lambda: _fake_transformers(calls),
    )

    model = build_backbone(
        "dinov3_vitb16_pt",
        embedding_dim=12,
        input_channels=3,
        local_files_only=True,
    )
    model.eval()

    with torch.no_grad():
        embeddings = model(torch.randn(2, 3, 32, 32))

    assert isinstance(model, DINOv3ViTBackbone)
    assert isinstance(model.projection, nn.Identity)
    assert tuple(embeddings.shape) == (2, 12)
    assert torch.allclose(embeddings, torch.ones(2, 12))
    assert calls == [
        (
            "pretrained",
            "facebook/dinov3-vitb16-pretrain-lvd1689m",
            {"local_files_only": True, "attn_implementation": "sdpa"},
        )
    ]


def test_dinov3_vit_projects_when_embedding_dim_differs(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "apexid.models.backbones.dinov3_vit._transformers_classes",
        lambda: _fake_transformers(calls),
    )

    model = build_backbone(
        "dinov3_vitb16_pt",
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
    assert not torch.allclose(embeddings, torch.full((2, 16), 2.0))


def test_dinov3_vit_uses_cls_not_pooler(monkeypatch):
    monkeypatch.setattr(
        "apexid.models.backbones.dinov3_vit._transformers_classes",
        lambda: _fake_transformers([]),
    )
    model = build_backbone("dinov3_vitb16_pt", embedding_dim=12, input_channels=3)
    model.eval()

    with torch.no_grad():
        embeddings = model(torch.randn(1, 3, 32, 32))

    assert torch.allclose(embeddings, torch.ones(1, 12))
    assert not torch.allclose(embeddings, torch.full((1, 12), 2.0))
    assert not torch.allclose(embeddings, torch.full((1, 12), 3.0))


def test_dinov3_vit_model_name_override(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "apexid.models.backbones.dinov3_vit._transformers_classes",
        lambda: _fake_transformers(calls),
    )

    model = build_backbone(
        "dinov3_vits16_pt",
        embedding_dim=12,
        model_name="owner/custom-dinov3-vit",
    )

    assert model.model_name == "owner/custom-dinov3-vit"
    assert calls[0][1] == "owner/custom-dinov3-vit"


def test_apexid_normalizes_dinov3_vit_embeddings(monkeypatch):
    monkeypatch.setattr(
        "apexid.models.backbones.dinov3_vit._transformers_classes",
        lambda: _fake_transformers([]),
    )
    model = ApexID(
        backbone_name="dinov3_vitb16_pt",
        embedding_dim=12,
        input_channels=3,
    )
    model.eval()

    with torch.no_grad():
        embeddings = model(torch.randn(2, 3, 32, 32))

    assert tuple(embeddings.shape) == (2, 12)
    assert torch.allclose(embeddings.norm(dim=1), torch.ones(2), atol=1e-5)


def test_dinov3_vit_rejects_non_multiple_image_size(monkeypatch):
    monkeypatch.setattr(
        "apexid.models.backbones.dinov3_vit._transformers_classes",
        lambda: _fake_transformers([]),
    )
    model = build_backbone("dinov3_vitb16_pt", embedding_dim=12, input_channels=3)

    with pytest.raises(ValueError, match="multiple of patch_size=16"):
        model(torch.randn(1, 3, 440, 440))


def test_dinov3_vit_pinned_revision_and_hash_are_verified(tmp_path, monkeypatch):
    calls = []
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"verified-dino-vit-weight")
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    monkeypatch.setattr(
        "apexid.models.backbones.dinov3_vit._transformers_classes",
        lambda: _fake_transformers(calls),
    )
    monkeypatch.setattr(
        "apexid.models.backbones.dinov3_vit._hf_hub_download",
        lambda **kwargs: weight,
    )
    model = build_backbone(
        "dinov3_vitb16_pt",
        embedding_dim=12,
        revision="immutable",
        expected_sha256=digest,
    )
    assert calls[0][2]["revision"] == "immutable"
    assert calls[0][2]["attn_implementation"] == "sdpa"
    assert model.pretrained_provenance["sha256"] == digest


def test_dinov3_vit_rejects_weight_hash_mismatch(tmp_path, monkeypatch):
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"wrong")
    monkeypatch.setattr(
        "apexid.models.backbones.dinov3_vit._transformers_classes",
        lambda: _fake_transformers([]),
    )
    monkeypatch.setattr(
        "apexid.models.backbones.dinov3_vit._hf_hub_download",
        lambda **kwargs: weight,
    )
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        build_backbone(
            "dinov3_vitb16_pt",
            embedding_dim=12,
            revision="immutable",
            expected_sha256="0" * 64,
        )
