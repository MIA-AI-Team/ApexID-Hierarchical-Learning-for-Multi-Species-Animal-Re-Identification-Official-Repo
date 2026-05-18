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
