from apexid.models import build_components
from apexid.models.heads import ArcFaceHead
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
        "embedding_plot",
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
    assert cfg["experiment"]["name"] == "apexid_osnet_arcface"
    assert cfg["head"]["name"] == "arcface"
    assert cfg["head"]["kwargs"]["s"] == 64.0
    assert cfg["head"]["kwargs"]["m"] == 0.5
    assert cfg["loss"]["name"] == "cross_entropy"
    assert cfg["lr_range_test"]["enabled"] is True
    assert cfg["lr_range_test"]["mode"] == "auto_apply"
    assert cfg["scheduler"]["name"] == "cyclic_lr"
    assert cfg["scheduler"]["kwargs"]["base_lr"] == 1e-5
    assert cfg["scheduler"]["kwargs"]["max_lr"] == 3e-4
    assert cfg["scheduler"]["kwargs"]["cycle_momentum"] is False
    assert cfg["embedding_plot"]["enabled"] is True
    assert cfg["embedding_plot"]["split"] == "val"
    assert cfg["embedding_plot"]["output_dir"] is None
    assert cfg["embedding_plot"]["best_output_dir"] is None
    assert cfg["embedding_plot"]["filename_template"] == "epoch_{epoch_number:03d}_umap.png"
    assert cfg["embedding_plot"]["best_filename"] == "best_embedding_space.png"
    assert cfg["tracking"]["enabled"] is False
    assert cfg["tracking"]["provider"] == "wandb"
    assert cfg["tracking"]["log_artifacts"] == "best"


def test_default_config_builds_arcface_head_with_cross_entropy_loss():
    cfg = load_config()
    model, head, criterion = build_components(cfg, num_classes=1000)

    assert model.embedding_dim == cfg["model"]["embedding_dim"]
    assert isinstance(head, ArcFaceHead)
    assert head.s == cfg["head"]["kwargs"]["s"]
    assert head.m == cfg["head"]["kwargs"]["m"]
    assert criterion is not None
