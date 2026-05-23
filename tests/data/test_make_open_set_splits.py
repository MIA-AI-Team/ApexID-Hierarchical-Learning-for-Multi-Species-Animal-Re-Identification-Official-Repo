import csv
import importlib.util
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "data" / "make_open_set_splits.py"


def _load_split_module():
    spec = importlib.util.spec_from_file_location("make_open_set_splits", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=(64, 96, 128)).save(path)


def _row(dataset, identity, path):
    return {
        "identity": identity,
        "path": path,
        "date": "",
        "orientation": "",
        "species": "",
        "dataset": dataset,
    }


def _identity_rows(root, dataset, identity, count):
    rows = []
    for idx in range(count):
        rel_path = f"images/{dataset}/{identity}_{idx}.png"
        _write_image(root / rel_path)
        rows.append(_row(dataset, identity, rel_path))
    return rows


def _write_metadata(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _paths(rows):
    return {row["path"] for row in rows}


def test_make_splits_routes_singletons_and_new_whale_to_unlabeled_ood(tmp_path):
    mod = _load_split_module()
    rows = []
    for idx in range(6):
        rows.extend(_identity_rows(tmp_path, "DatasetA", f"id_{idx}", 4))
    rows.extend(_identity_rows(tmp_path, "DatasetA", "singleton", 1))
    rows.extend(
        _identity_rows(
            tmp_path,
            "humpback-whale-identification",
            "new_whale",
            3,
        )
    )

    source = tmp_path / "metadata.csv"
    _write_metadata(source, rows, mod.FIELDNAMES)
    rows = mod.read_rows(source)
    splits = mod.make_splits(
        rows,
        seed=7,
        unknown_identity_ratio=0.5,
        known_image_ratio=0.5,
        max_unknown_images_per_identity=2,
    )

    output_dir = tmp_path / "splits"
    stale_file = output_dir / "old_split.csv"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text("old,data\n", encoding="utf-8")

    mod.write_splits(output_dir, splits, clear_existing=True)
    validation = mod.validate_outputs(
        output_dir,
        source_rows=rows,
        root_dir=tmp_path,
        check_images=True,
    )

    assert not stale_file.exists()
    assert validation["new_whale_counts"] == {
        "train": 0,
        "val_known": 0,
        "val_unknown": 0,
        "val_unknown_extra": 0,
        "unlabeled_ood": 3,
    }

    singleton_paths = {
        row["path"]
        for row in rows
        if row["dataset"] == "DatasetA" and row["identity"] == "singleton"
    }
    new_whale_paths = {
        row["path"]
        for row in rows
        if row["dataset"] == "humpback-whale-identification"
        and row["identity"] == "new_whale"
    }
    unlabeled_paths = _paths(splits["unlabeled_ood"])
    assert singleton_paths <= unlabeled_paths
    assert new_whale_paths <= unlabeled_paths

    train_ids = mod.identity_keys(splits["train"])
    val_known_ids = mod.identity_keys(splits["val_known"])
    val_unknown_ids = mod.identity_keys(splits["val_unknown"])
    val_unknown_extra_ids = mod.identity_keys(splits["val_unknown_extra"])

    assert not train_ids & val_unknown_ids
    assert val_known_ids <= train_ids
    assert not train_ids & val_unknown_extra_ids
    assert val_unknown_extra_ids <= val_unknown_ids
    assert splits["val"] == splits["val_known"] + splits["val_unknown"]

    component_paths = [
        row["path"]
        for name in mod.COMPONENT_SPLITS
        for row in splits[name]
    ]
    assert len(component_paths) == len(rows)
    assert set(component_paths) == {row["path"] for row in rows}
