import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_rgb_png(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def _make_rows(image_dir, prefix, identities):
    rows = []
    for idx, identity in enumerate(identities):
        image_path = image_dir / f"{prefix}_{idx}.png"
        image = np.full((10, 10, 3), idx * 20 + 20, dtype=np.uint8)
        _write_rgb_png(image_path, image)
        rows.append(
            {
                "dataset": identity.split("_", 1)[0],
                "identity": identity,
                "path": image_path.name,
            }
        )
    return rows


def test_debug_data_pipeline_script_smoke(tmp_path):
    image_dir = tmp_path / "images"
    train_csv = tmp_path / "train.csv"
    val_known_csv = tmp_path / "val_known.csv"
    val_unknown_csv = tmp_path / "val_unknown.csv"
    val_csv = tmp_path / "val.csv"
    config_path = tmp_path / "config.yaml"

    train_rows = _make_rows(
        image_dir,
        "train",
        ["lynx_001", "lynx_001", "turtle_001", "turtle_001"],
    )
    val_rows = _make_rows(
        image_dir,
        "val",
        ["lynx_001", "turtle_001", "salamander_001"],
    )
    val_rows[-1]["val_type"] = "unknown"
    val_rows[0]["val_type"] = "known"
    val_rows[1]["val_type"] = "known"
    pd.DataFrame(train_rows).to_csv(train_csv, index=False)
    pd.DataFrame(val_rows).to_csv(val_csv, index=False)
    pd.DataFrame([row for row in val_rows if row["val_type"] == "known"]).to_csv(val_known_csv, index=False)
    pd.DataFrame([row for row in val_rows if row["val_type"] == "unknown"]).to_csv(val_unknown_csv, index=False)

    config = {
        "seed": 42,
        "data": {
            "train_path": str(train_csv),
            "val_known_path": str(val_known_csv),
            "val_unknown_path": str(val_unknown_csv),
            "val_path": str(val_csv),
            "root_dir": str(image_dir),
            "image_size": [8, 8],
            "wavelet": False,
            "transforms": {
                "mean": [0.5, 0.5, 0.5],
                "std": [0.25, 0.25, 0.25],
            },
        },
        "dataloader": {
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "drop_last": False,
            "sampler": {
                "name": "pk",
                "num_identities": 2,
                "num_instances": 2,
                "shuffle": False,
            },
        },
        "training": {"classification_ignore_index": -100},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "debug_data_pipeline.py"),
            "--config",
            str(config_path),
            "--max-examples",
            "3",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "train csv:" in result.stdout
    assert "unique global identities: 2" in result.stdout
    assert "label_to_identity bijection: PASS" in result.stdout
    assert "[first DataLoader batch]" in result.stdout
    assert "every cls label in [0, 1]: True" in result.stdout
