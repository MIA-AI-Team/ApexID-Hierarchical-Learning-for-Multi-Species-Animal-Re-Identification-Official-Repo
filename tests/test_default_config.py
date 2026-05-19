from apexid.models import build_components
from apexid.models.heads import SphereFace2Head
from apexid.utils.config import load_config


def test_default_config_contains_experiment_training_sections():
    cfg = load_config()

    for section in [
        "experiment",
        "seed",
        "device",
        "data",
        "dataloader",
        "model",
        "head",
        "loss",
        "optimizer",
        "lr_range_test",
        "scheduler",
        "training",
        "validation",
        "checkpoint",
        "early_stopping",
        "tracking",
    ]:
        assert section in cfg

    assert cfg["seed"] == 42
    assert "train_path" in cfg["data"]
    assert "val_path" in cfg["data"]
    assert cfg["model"]["name"] == "apexid"
    assert "input_channels" in cfg["model"]
    assert cfg["validation"]["metrics"] == ["top1", "top5", "mAP", "micro_f1"]
    assert cfg["early_stopping"]["monitor"] == "val_mAP"
    assert cfg["early_stopping"]["mode"] == "max"
    assert cfg["checkpoint"]["delete_previous"] is True
    assert cfg["training"]["verbose"] is True
    assert cfg["training"]["classification_ignore_index"] == -100
    assert cfg["head"]["name"] == "sphereface2"
    assert cfg["head"]["kwargs"]["alpha"] == 0.9989
    assert cfg["loss"]["name"] is None
    assert cfg["lr_range_test"]["enabled"] is True
    assert cfg["lr_range_test"]["mode"] == "auto_apply"
    assert cfg["scheduler"]["name"] == "cyclic_lr"
    assert cfg["scheduler"]["kwargs"]["base_lr"] == 1e-5
    assert cfg["scheduler"]["kwargs"]["max_lr"] == 3e-4
    assert cfg["scheduler"]["kwargs"]["cycle_momentum"] is False
    assert cfg["tracking"]["enabled"] is False
    assert cfg["tracking"]["provider"] == "wandb"
    assert cfg["tracking"]["log_artifacts"] == "best"


def test_default_config_builds_internal_loss_sphereface2_head():
    cfg = load_config()
    model, head, criterion = build_components(cfg, num_classes=1000)

    assert model.embedding_dim == cfg["model"]["embedding_dim"]
    assert isinstance(head, SphereFace2Head)
    assert head.alpha == cfg["head"]["kwargs"]["alpha"]
    assert criterion is None
