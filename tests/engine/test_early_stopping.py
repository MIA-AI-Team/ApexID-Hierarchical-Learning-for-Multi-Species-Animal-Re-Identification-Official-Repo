import json

import torch
from torch import nn

from apexid.engine.early_stopping import EarlyStopping


def test_early_stopping_saves_full_model_and_backbone_weights(tmp_path):
    backbone = nn.Linear(4, 3)
    head = nn.Linear(3, 2)
    model = nn.ModuleDict({"backbone": backbone, "head": head})

    stopper = EarlyStopping(
        monitor="val_top1",
        mode="max",
        patience=2,
        output_dir=tmp_path,
        verbose=False,
    )

    is_best = stopper({"val_top1": 0.5}, model=model, epoch=3)

    assert is_best is True
    assert stopper.best_metric == 0.5
    assert stopper.best_epoch == 3
    assert stopper.full_model_path.exists()
    assert stopper.backbone_path.exists()
    assert stopper.metadata_path.exists()

    full_state = torch.load(stopper.full_model_path, map_location="cpu")
    backbone_state = torch.load(stopper.backbone_path, map_location="cpu")

    assert set(full_state.keys()) == {
        "backbone.weight",
        "backbone.bias",
        "head.weight",
        "head.bias",
    }
    assert set(backbone_state.keys()) == {"weight", "bias"}
    torch.testing.assert_close(full_state["backbone.weight"], backbone_state["weight"])
    torch.testing.assert_close(full_state["backbone.bias"], backbone_state["bias"])

    metadata = json.loads(stopper.metadata_path.read_text())
    assert metadata["monitor"] == "val_top1"
    assert metadata["mode"] == "max"
    assert metadata["best_metric"] == 0.5
    assert metadata["best_epoch"] == 3


def test_early_stopping_patience_and_min_delta(tmp_path):
    model = nn.ModuleDict({"backbone": nn.Linear(2, 2), "head": nn.Linear(2, 2)})
    stopper = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=2,
        min_delta=0.1,
        output_dir=tmp_path,
    )

    assert stopper({"val_loss": 1.0}, model=model, epoch=0) is True
    assert stopper({"val_loss": 0.95}, model=model, epoch=1) is False
    assert stopper.early_stop is False
    assert stopper({"val_loss": 0.94}, model=model, epoch=2) is False
    assert stopper.early_stop is True
    assert stopper.best_metric == 1.0
    assert stopper.best_epoch == 0


def test_early_stopping_can_save_from_separate_backbone_and_head(tmp_path):
    backbone = nn.Sequential(nn.Linear(5, 4), nn.ReLU(), nn.Linear(4, 3))
    head = nn.Linear(3, 2)
    stopper = EarlyStopping(
        monitor="mAP",
        mode="max",
        output_dir=tmp_path,
    )

    assert stopper({"mAP": 0.25}, backbone=backbone, head=head, epoch=1) is True

    full_state = torch.load(stopper.full_model_path, map_location="cpu")
    backbone_state = torch.load(stopper.backbone_path, map_location="cpu")

    assert all(key.startswith(("backbone.", "head.")) for key in full_state)
    assert not any(key.startswith("head.") for key in backbone_state)
    assert "0.weight" in backbone_state
