from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from apexid.data.transforms import build_eval_transforms
from apexid.models import ApexID
from apexid.models.backbones import available_backbones, build_backbone
from apexid.models.backbones import megadescriptor, miewid
from apexid.models.model_factory import build_components
from apexid.training.finetune import apply_finetune_phase, resolve_training_profile
from apexid.utils.config import load_config


class TinyEncoder(nn.Module):
    num_features = 8

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, self.num_features, 1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.checkpointing = False

    def set_grad_checkpointing(self, enabled):
        self.checkpointing = enabled

    def forward(self, x):
        return self.global_pool(self.conv(x)).flatten(1)


@pytest.fixture
def fake_timm(monkeypatch):
    calls = []

    def create(name, **kwargs):
        calls.append((name, kwargs))
        return TinyEncoder()

    monkeypatch.setattr(miewid, "_create_model", create)
    monkeypatch.setattr(megadescriptor, "_create_model", create)
    return calls


@pytest.fixture
def miewid_weights(tmp_path, fake_timm, monkeypatch):
    original = miewid._MiewIDEmbedding()
    with torch.no_grad():
        original.backbone.global_pool.p.fill_(2.5)
        original.bn.running_mean.fill_(0.3)
    state = original.state_dict()
    state.update({"final.weight": torch.randn(10, 8), "final.bias": torch.randn(10)})
    path = tmp_path / "model.safetensors"
    save_file(state, str(path))
    monkeypatch.setattr(miewid, "_download_weights", lambda **kwargs: path)
    return original, path


def test_miewid_restores_complete_embedding_and_trains(miewid_weights):
    original, path = miewid_weights
    backbone = build_backbone(
        "miewid_msv3_pt", embedding_dim=8,
        expected_sha256=miewid.sha256_file(path),
    )
    assert isinstance(backbone.projection, nn.Identity)
    x = torch.randn(2, 3, 12, 12)
    original.eval()
    backbone.eval()
    torch.testing.assert_close(backbone(x), original(x))
    backbone.train()
    backbone(x).square().sum().backward()
    assert backbone.model.backbone.global_pool.p.grad is not None
    assert backbone.model.bn.weight.grad is not None
    assert all(p.grad is not None for p in backbone.parameters())


def test_miewid_rejects_corrupt_weights(miewid_weights):
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        build_backbone("miewid_msv3_pt", expected_sha256="0" * 64)


def test_miewid_rejects_missing_embedding_weights(miewid_weights):
    _, path = miewid_weights
    state = miewid._MiewIDEmbedding().state_dict()
    del state["bn.running_mean"]
    save_file(state, str(path))
    with pytest.raises(RuntimeError, match="bn.running_mean"):
        build_backbone("miewid_msv3_pt", expected_sha256=None)


def test_megadescriptor_uses_pinned_wildlife_weights(fake_timm):
    model = build_backbone("megadescriptor_l_384_pt", embedding_dim=8, grad_checkpointing=True)
    assert fake_timm == [(
        f"hf-hub:BVRA/MegaDescriptor-L-384@{megadescriptor.MEGADESCRIPTOR_REVISION}",
        {"pretrained": True, "num_classes": 0, "cache_dir": None},
    )]
    assert isinstance(model.projection, nn.Identity)
    assert model.model.checkpointing
    with pytest.raises(ValueError, match="384"):
        model(torch.randn(2, 3, 440, 440))


@pytest.mark.parametrize("name", ["miewid_msv3", "megadescriptor_l_384"])
def test_offline_restore_and_normalized_projected_embeddings(name, fake_timm, monkeypatch):
    def no_download(**kwargs):
        pytest.fail("Scratch/checkpoint restoration must not download weights")

    monkeypatch.setattr(miewid, "_download_weights", no_download)
    assert {name, name + "_pt"}.issubset(available_backbones())
    model = ApexID(backbone_name=name + "_pt", embedding_dim=5, backbone_kwargs={"pretrained": False})
    model.eval()
    x = torch.randn(2, 3, 384, 384)
    with torch.no_grad():
        expected = model(x)
    restored = ApexID(backbone_name=name, embedding_dim=5)
    restored.load_state_dict(model.state_dict(), strict=True)
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(x), expected)
    torch.testing.assert_close(expected.norm(dim=1), torch.ones(2))
    assert all(not kwargs["pretrained"] for _, kwargs in fake_timm)
    assert all(not source.startswith("hf-hub:") for source, _ in fake_timm)


@pytest.mark.parametrize("name", ["miewid_msv3", "megadescriptor_l_384"])
def test_invalid_input_channels_fail_before_loading(name, fake_timm):
    with pytest.raises(ValueError, match="input_channels=3"):
        build_backbone(name + "_pt", input_channels=6)
    with pytest.raises(ValueError, match="positive"):
        build_backbone(name + "_pt", embedding_dim=0)
    assert not fake_timm


@pytest.mark.parametrize("name,size,mean,std", [
    ("miewid_msv3", 440, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ("megadescriptor_l_384", 384, [0.5] * 3, [0.5] * 3),
])
def test_finetune_recipe_backward_and_preprocessing(name, size, mean, std, fake_timm):
    path = Path(__file__).resolve().parents[2] / "configs" / f"finetune_{name}.yaml"
    cfg = resolve_training_profile(load_config(config_path=path))
    assert cfg["_finetune_backbone_lr"] == pytest.approx(1e-5)
    assert cfg["_finetune_head_lr"] == pytest.approx(1e-4)
    assert cfg["dataloader"]["batch_size"] == (
        cfg["dataloader"]["sampler"]["num_identities"] * cfg["dataloader"]["sampler"]["num_instances"]
    )
    # Exercise the complete factory/head path without fetching large weights.
    cfg["model"]["backbone"]["kwargs"]["pretrained"] = False
    model, head, criterion = build_components(cfg, num_classes=3)
    assert criterion is None
    assert apply_finetune_phase(model, head, cfg, epoch=0)["backbone_trainable"]
    transform = build_eval_transforms(config=cfg)
    image = transform(image=np.full((16, 20, 3), 255, dtype=np.uint8))["image"]
    assert image.shape == (3, size, size)
    torch.testing.assert_close(image[:, 0, 0], (1 - torch.tensor(mean)) / torch.tensor(std))
    x = torch.stack([image, image * 0.5])
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=1e-4)
    before = model.backbone.model.conv.weight.clone() if name.startswith("mega") else model.backbone.model.backbone.conv.weight.clone()
    loss = head(model(x), torch.tensor([0, 1]))
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    after = model.backbone.model.conv.weight if name.startswith("mega") else model.backbone.model.backbone.conv.weight
    assert not torch.equal(before, after)
