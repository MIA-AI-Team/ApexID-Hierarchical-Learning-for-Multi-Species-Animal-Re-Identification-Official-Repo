import pytest
import torch
from torch import nn

from apexid.models.apexid_model import ApexID
from apexid.training.finetune import (
    apply_finetune_phase,
    is_finetune_active,
    is_pretrained_backbone,
    resolve_training_profile,
    should_run_lr_range_test,
)


def _base_cfg():
    return {
        "model": {
            "backbone": {"name": "osnet_x0_25", "kwargs": {}},
        },
        "training": {"epochs": 50},
        "optimizer": {"name": "adamw", "kwargs": {"lr": 3e-4}},
        "scheduler": {"name": "cyclic_lr", "kwargs": {}},
        "lr_range_test": {"enabled": True},
        "finetune": {"enabled": "auto"},
    }


def test_is_pretrained_backbone_detects_pt_alias():
    cfg = _base_cfg()
    cfg["model"]["backbone"]["name"] = "osnet_x1_0_pt"
    assert is_pretrained_backbone(cfg) is True


def test_is_pretrained_backbone_detects_pretrained_kwarg():
    cfg = _base_cfg()
    cfg["model"]["backbone"]["kwargs"] = {"pretrained": True}
    assert is_pretrained_backbone(cfg) is True


def test_is_finetune_active_false_for_scratch_backbone():
    cfg = _base_cfg()
    assert is_finetune_active(cfg) is False


def test_resolve_training_profile_switches_scheduler_for_pretrained():
    cfg = _base_cfg()
    cfg["model"]["backbone"]["name"] = "convnext_base_pt"
    resolved = resolve_training_profile(cfg)

    assert resolved["scheduler"]["name"] == "cosine_annealing_lr"
    assert resolved["scheduler"]["kwargs"]["T_max"] == 50
    assert resolved["lr_range_test"]["enabled"] is False
    assert resolved["_finetune_backbone_lr"] == pytest.approx(1e-5)
    assert resolved["_finetune_head_lr"] == pytest.approx(1e-4)


def test_resolve_training_profile_keeps_scratch_settings():
    cfg = _base_cfg()
    resolved = resolve_training_profile(cfg)

    assert resolved["scheduler"]["name"] == "cyclic_lr"
    assert resolved["lr_range_test"]["enabled"] is True
    assert "_finetune_backbone_lr" not in resolved


def test_should_run_lr_range_test_respects_finetune_defaults():
    cfg = _base_cfg()
    cfg["model"]["backbone"]["name"] = "osnet_x1_0_pt"
    resolved = resolve_training_profile(cfg)
    assert should_run_lr_range_test(resolved) is False


def test_apply_finetune_phase_head_first_freezes_backbone():
    model = ApexID(backbone_name="osnet_x0_25", embedding_dim=8, input_channels=3)
    head = nn.Linear(8, 4)
    cfg = {
        "finetune": {
            "enabled": True,
            "strategy": "head_first",
            "head_only_epochs": 2,
            "unfreeze_top_fraction": 0.0,
        }
    }

    phase = apply_finetune_phase(model, head, cfg, epoch=0)
    assert phase["phase"] == "head_first"
    assert not any(param.requires_grad for param in model.backbone.parameters())
    assert all(param.requires_grad for param in head.parameters())

    phase = apply_finetune_phase(model, head, cfg, epoch=2)
    assert phase["backbone_trainable"] is True
    assert all(param.requires_grad for param in model.backbone.parameters())


def test_apply_finetune_phase_full_keeps_backbone_trainable():
    model = ApexID(backbone_name="osnet_x0_25", embedding_dim=8, input_channels=3)
    head = nn.Linear(8, 4)
    cfg = {"finetune": {"enabled": True, "strategy": "full"}}

    phase = apply_finetune_phase(model, head, cfg, epoch=0)
    assert phase["phase"] == "full"
    assert all(param.requires_grad for param in model.backbone.parameters())
