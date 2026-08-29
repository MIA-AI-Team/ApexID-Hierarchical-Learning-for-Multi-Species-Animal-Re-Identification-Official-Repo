import cv2
import numpy as np
import pandas as pd
import yaml

from apexid.utils.calculate_mean_std import (
    calculate_from_config,
    write_normalization_provenance,
)


def _write_rgb_png(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def test_calculate_mean_std_from_config_and_write_config(tmp_path):
    image_dir = tmp_path / "images"
    image_a = np.full((12, 12, 3), [40, 80, 120], dtype=np.uint8)
    image_b = np.full((12, 12, 3), [160, 120, 80], dtype=np.uint8)
    _write_rgb_png(image_dir / "a.png", image_a)
    _write_rgb_png(image_dir / "b.png", image_b)

    csv_path = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "identity": ["a", "b"],
            "path": ["a.png", "b.png"],
        }
    ).to_csv(csv_path, index=False)

    config_path = tmp_path / "config.yaml"
    config = {
        "data": {
            "train_path": str(csv_path),
            "root_dir": str(image_dir),
            "image_size": [12, 12],
            "wavelet": False,
            "transforms": {
                "mean": [0.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0],
            },
        }
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    mean, std = calculate_from_config(
        config_path=config_path,
        precision=6,
        write_config=True,
    )

    assert len(mean) == 3
    assert len(std) == 3
    assert all(0.0 <= value <= 1.0 for value in mean)
    assert all(value > 0.0 for value in std)

    updated = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert updated["data"]["transforms"]["mean"] == mean
    assert updated["data"]["transforms"]["std"] == std


def test_write_normalization_provenance_is_train_only_and_hashed(tmp_path):
    train_csv = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "identity": ["a", "b"],
            "path": ["a.png", "b.png"],
        }
    ).to_csv(train_csv, index=False)
    manifest = tmp_path / "split_manifest.csv"
    manifest.write_text("source_index,split\n0,train\n1,train\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "train_path": str(train_csv),
                    "image_size": [440, 440],
                    "wavelet": False,
                    "transforms": {
                        "mean": [0.1, 0.2, 0.3],
                        "std": [0.4, 0.5, 0.6],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "normalization_stats.json"
    payload = write_normalization_provenance(
        output,
        config_path,
        train_csv,
        manifest,
        [0.11, 0.22, 0.33],
        [0.44, 0.55, 0.66],
        command="test command",
    )
    assert payload["row_count"] == 2
    assert payload["preprocessing"]["image_size"] == [440, 440]
    assert payload["command"] == "test command"
    assert payload["manifest_sha256"]
    assert yaml.safe_load(output.read_text(encoding="utf-8"))["mean"] == [
        0.11,
        0.22,
        0.33,
    ]
