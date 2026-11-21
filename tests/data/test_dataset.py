import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from apexid.data.datasets.dataset import ApexID_Dataset


def _write_rgb_png(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def test_apexid_dataset_loads_rgb_images_labels_and_identity_index(tmp_path):
    image_dir = tmp_path / "images"
    rgb_a = np.zeros((4, 5, 3), dtype=np.uint8)
    rgb_a[..., 0] = 10
    rgb_a[..., 1] = 20
    rgb_a[..., 2] = 30
    rgb_b = np.full((4, 5, 3), [40, 50, 60], dtype=np.uint8)
    rgb_c = np.full((4, 5, 3), [70, 80, 90], dtype=np.uint8)

    _write_rgb_png(image_dir / "a.png", rgb_a)
    _write_rgb_png(image_dir / "b.png", rgb_b)
    _write_rgb_png(image_dir / "c.png", rgb_c)

    data = pd.DataFrame(
        {
            "individual_id": ["zebra_2", "zebra_1", "zebra_2"],
            "image_path": ["images/a.png", "images/b.png", "images/c.png"],
        }
    )

    dataset = ApexID_Dataset(data, root_dir=tmp_path)

    image, cls_label, metric_label = dataset[0]

    assert len(dataset) == 3
    assert image.shape == (4, 5, 3)
    assert image.dtype == np.uint8
    np.testing.assert_array_equal(image, rgb_a)
    assert cls_label.shape == ()
    assert metric_label.shape == ()
    assert cls_label.dtype == torch.long
    assert metric_label.dtype == torch.long
    assert cls_label.item() == 1
    assert metric_label.item() == 1
    assert dataset.label_to_idx == {"zebra_1": 0, "zebra_2": 1}
    assert dataset.identity_to_indices == {"zebra_1": [1], "zebra_2": [0, 2]}


def test_apexid_dataset_keeps_global_animal_identities_distinct():
    data = pd.DataFrame(
        {
            "species": ["lynx", "turtle", "salamander"],
            "identity": ["lynx_001", "turtle_001", "salamander_001"],
            "path": ["lynx.jpg", "turtle.jpg", "salamander.jpg"],
        }
    )

    dataset = ApexID_Dataset(data)

    assert dataset.label_to_idx == {
        "lynx_001": 0,
        "salamander_001": 1,
        "turtle_001": 2,
    }
    assert len(set(dataset.label_to_idx.values())) == 3


def test_apexid_dataset_rejects_identity_collisions_across_species():
    data = pd.DataFrame(
        {
            "species": ["lynx", "turtle", "salamander"],
            "identity": ["001", "001", "001"],
            "path": ["lynx.jpg", "turtle.jpg", "salamander.jpg"],
        }
    )

    with pytest.raises(ValueError, match="not globally unique"):
        ApexID_Dataset(data)


def test_apexid_dataset_detects_source_collisions_across_splits():
    train = pd.DataFrame({"identity": ["001"], "dataset": ["Set A"], "species": ["whale"]})
    validation = pd.DataFrame({"identity": ["001"], "dataset": ["Set B"], "species": ["whale"]})

    with pytest.raises(ValueError, match="not globally unique.*"):
        ApexID_Dataset.build_label_mapping_union(train, validation)


def test_apexid_dataset_applies_transform_after_rgb_loading(tmp_path):
    image_path = tmp_path / "animal.png"
    rgb = np.full((6, 7, 3), [15, 25, 35], dtype=np.uint8)
    _write_rgb_png(image_path, rgb)

    class CaptureTransform:
        def __init__(self):
            self.seen_image = None

        def __call__(self, image):
            self.seen_image = image.copy()
            tensor = torch.from_numpy(np.transpose(image, (2, 0, 1)).copy())
            return {"image": tensor}

    transform = CaptureTransform()
    dataset = ApexID_Dataset(
        pd.DataFrame({"identity": ["animal_1"], "path": [str(image_path)]}),
        transform=transform,
    )

    image, cls_label, metric_label = dataset[0]

    np.testing.assert_array_equal(transform.seen_image, rgb)
    assert tuple(image.shape) == (3, 6, 7)
    assert image.dtype == torch.uint8
    assert cls_label.item() == 0
    assert metric_label.item() == 0


def test_apexid_dataset_supports_unknown_validation_identities(tmp_path):
    known_path = tmp_path / "known.png"
    unknown_path = tmp_path / "unknown.png"
    _write_rgb_png(known_path, np.full((3, 3, 3), 80, dtype=np.uint8))
    _write_rgb_png(unknown_path, np.full((3, 3, 3), 160, dtype=np.uint8))

    data = pd.DataFrame(
        {
            "identity": ["known_id", "unknown_id"],
            "path": [str(known_path), str(unknown_path)],
        }
    )
    metrics_label_map = ApexID_Dataset.build_label_mapping(data)
    dataset = ApexID_Dataset(
        data,
        label_to_idx={"known_id": 0},
        allow_unknown_identity=True,
        unknown_label=-100,
        metrics_label_map=metrics_label_map,
    )

    _, known_cls, known_metric = dataset[0]
    _, unknown_cls, unknown_metric = dataset[1]

    assert known_cls.item() == 0
    assert known_metric.item() == metrics_label_map["known_id"]
    assert unknown_cls.item() == -100
    assert unknown_metric.item() == metrics_label_map["unknown_id"]
