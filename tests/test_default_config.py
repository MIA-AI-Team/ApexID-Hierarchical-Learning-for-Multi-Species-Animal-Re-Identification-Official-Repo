import pytest

from apexid.models import build_head
from apexid.models.heads import ApexIDHierarchicalHead
from apexid.utils.config import DEFAULT_CONFIG_PATH, load_config


CONFIG_DIR = DEFAULT_CONFIG_PATH.parent


def test_default_config_matches_convnext_apexid_paper_preset():
    cfg = load_config()
    assert cfg["model"]["backbone"]["name"] == "convnext_small_pt"
    assert cfg["model"]["embedding_dim"] == 2048
    assert cfg["model"]["input_channels"] == 3
    assert cfg["data"]["image_size"] == [440, 440]
    assert cfg["data"]["transforms"]["mean"] == [0.485, 0.456, 0.406]
    assert cfg["dataloader"]["batch_size"] == 64
    assert cfg["dataloader"]["sampler"]["num_identities"] == 16
    assert cfg["dataloader"]["sampler"]["num_instances"] == 4
    assert cfg["optimizer"]["backbone_lr"] == 1e-5
    assert cfg["optimizer"]["head_lr"] == 1e-4
    assert cfg["training"]["epochs"] == 150
    assert cfg["training"]["gradient_clip_norm"] == 5.0
    assert cfg["loss"]["name"] is None
    head_cfg = cfg["head"]
    head = build_head(head_cfg["name"], 3, 4, class_to_species=[0, 0, 1, 1], **head_cfg["kwargs"])
    assert isinstance(head, ApexIDHierarchicalHead)
    assert head.species_weight == 0.5
    assert head.num_subcenters == 2


@pytest.mark.parametrize(("family", "size", "dim", "batch_size"), [
    ("convnext_small", 440, 2048, 64),
    ("miewid_v3", 440, 2152, 32),
    ("megadescriptor_l", 384, 1536, 16),
])
@pytest.mark.parametrize("objective", ["arcface", "dynamic_arcface", "sphereface2", "apexid_hierarchical"])
def test_paper_training_matrix_builds_heads(family, size, dim, batch_size, objective):
    path = CONFIG_DIR / family / f"{objective}.yaml"
    if family == "convnext_small" and objective == "apexid_hierarchical":
        path = DEFAULT_CONFIG_PATH
    cfg = load_config(config_path=path)
    assert cfg["model"]["embedding_dim"] == dim
    assert cfg["data"]["image_size"] == [size, size]
    assert cfg["dataloader"]["batch_size"] == batch_size
    head_cfg = cfg["head"]
    assert head_cfg["name"] == objective
    kwargs = dict(head_cfg["kwargs"])
    if objective == "apexid_hierarchical":
        kwargs["class_to_species"] = [0, 0, 1, 1]
    assert build_head(objective, 3, 4, **kwargs) is not None


@pytest.mark.parametrize("weight", ["0.025", "0.05", "0.075", "0.10", "0.15", "0.25"])
def test_category_ablation_changes_only_weight_and_destination(weight):
    full = load_config()
    ablation = load_config(config_path=CONFIG_DIR / "ablations" / f"category_weight_{weight}.yaml")
    assert ablation.pop("experiment") != full.pop("experiment")
    assert ablation["head"]["kwargs"].pop("species_weight") == float(weight)
    assert full["head"]["kwargs"].pop("species_weight") == 0.5
    assert ablation == full
