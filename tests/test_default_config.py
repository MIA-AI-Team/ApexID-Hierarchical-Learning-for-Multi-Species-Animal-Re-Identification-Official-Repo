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
        "finetune",
        "training",
        "validation",
        "evaluation",
        "embedding_plot",
        "checkpoint",
        "early_stopping",
        "tracking",
    ]:
        assert section in cfg

    assert cfg["seed"] == 42
    assert "train_path" in cfg["data"]
    assert "val_known_path" in cfg["data"]
    assert "val_unknown_path" in cfg["data"]
    assert "val_path" in cfg["data"]
    assert cfg["model"]["name"] == "apexid"
    assert "input_channels" in cfg["model"]
    assert cfg["validation"]["metric_chunk_size"] == 1024
    assert cfg["validation"]["metrics"] == ["top1", "top5", "mAP", "micro_f1"]
    assert cfg["evaluation"]["diagnostics_every_n_epochs"] == 5
    assert cfg["evaluation"]["diagnostics"]["max_pairs"] == 100000
    assert cfg["evaluation"]["clustering"]["method"] == "threshold_connected_components"
    assert cfg["evaluation"]["sample_limits"]["gallery"] is None
    assert cfg["evaluation"]["sample_limits"]["known"] is None
    assert cfg["evaluation"]["sample_limits"]["unknown"] is None
    assert cfg["evaluation"]["sample_limits"]["mixed"] is None
    assert cfg["evaluation"]["cmc_ranks"] == [20, 50]
    assert cfg["evaluation"]["top_k"] == [1, 5, 10]
    assert cfg["evaluation"]["dir"]["far_levels"] == [0.01, 0.001]
    assert cfg["early_stopping"]["monitor"] == "known/mAP"
    assert cfg["early_stopping"]["mode"] == "max"
    assert cfg["checkpoint"]["delete_previous"] is True
    assert cfg["training"]["verbose"] is True
    assert cfg["training"]["classification_ignore_index"] == -100
    assert cfg["experiment"]["name"] == "apexid_osnet_arcface_64_0.5"
    assert cfg["head"]["name"] == "arcface"
    assert cfg["head"]["kwargs"]["s"] == 40.0
    assert cfg["head"]["kwargs"]["m"] == 0.3
    assert cfg["finetune"]["enabled"] == "auto"
    assert cfg["finetune"]["strategy"] == "full"
    assert cfg["loss"]["name"] == "cross_entropy"
    assert cfg["lr_range_test"]["enabled"] is True
    assert cfg["lr_range_test"]["mode"] == "auto_apply"
    assert cfg["scheduler"]["name"] == "cyclic_lr"
    assert cfg["scheduler"]["kwargs"]["base_lr"] == 1e-5
    assert cfg["scheduler"]["kwargs"]["max_lr"] == 3e-4
    assert cfg["scheduler"]["kwargs"]["cycle_momentum"] is False
    assert cfg["embedding_plot"]["enabled"] is False
    assert cfg["embedding_plot"]["split"] == "val"
    assert cfg["embedding_plot"]["output_dir"] is None
    assert cfg["embedding_plot"]["best_output_dir"] is None
    assert cfg["embedding_plot"]["filename_template"] == "epoch_{epoch_number:03d}_umap.png"
    assert cfg["embedding_plot"]["best_filename"] == "best_embedding_space.png"
    assert cfg["tracking"]["enabled"] is True
    assert cfg["tracking"]["provider"] == "wandb"
    assert cfg["tracking"]["log_artifacts"] == "none"
    assert cfg["tracking"]["block_on_artifact_upload"] is False
    assert cfg["tracking"]["artifact_finish_timeout"] == 300


def test_default_config_builds_arcface_head_with_cross_entropy_loss():
    cfg = load_config()
    model, head, criterion = build_components(cfg, num_classes=1000)

    assert model.embedding_dim == cfg["model"]["embedding_dim"]
    assert isinstance(head, ArcFaceHead)
    assert head.s == cfg["head"]["kwargs"]["s"]
    assert head.m == cfg["head"]["kwargs"]["m"]
    assert criterion is not None
