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
        "scheduler",
        "training",
        "validation",
        "checkpoint",
        "early_stopping",
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
    assert cfg["head"]["kwargs"]["alpha"] == 0.7
    assert cfg["loss"]["name"] is None


def test_default_config_builds_internal_loss_sphereface2_head():
    cfg = load_config()
    model, head, criterion = build_components(cfg, num_classes=4)

    assert model.embedding_dim == cfg["model"]["embedding_dim"]
    assert isinstance(head, SphereFace2Head)
    assert head.alpha == cfg["head"]["kwargs"]["alpha"]
    assert criterion is None
