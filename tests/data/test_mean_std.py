import cv2
import numpy as np
import pandas as pd
import albumentations as A

from apexid.data.DWT import append_haar_wavelet_channels
from apexid.data.transforms import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    build_eval_transforms,
    calculate_dataset_mean_std,
)


def _write_rgb_png(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def _load_rgb(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert bgr is not None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _manual_stats(paths, image_size, wavelet=False):
    height, width = image_size
    pixels_per_image = []
    clahe = A.CLAHE(
        clip_limit=CLAHE_CLIP_LIMIT,
        tile_grid_size=CLAHE_TILE_GRID_SIZE,
        p=1.0,
    )

    for path in paths:
        image = _load_rgb(path)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        image = clahe(image=image)["image"]
        if wavelet:
            image = append_haar_wavelet_channels(image)
        else:
            image = image.astype(np.float32) / 255.0
        pixels_per_image.append(image.reshape(-1, image.shape[-1]).astype(np.float64))

    pixels = np.concatenate(pixels_per_image, axis=0)
    return pixels.mean(axis=0), pixels.std(axis=0)


def test_calculate_dataset_mean_std_matches_manual_rgb_calculation(tmp_path):
    image_a = np.zeros((16, 20, 3), dtype=np.uint8)
    image_a[..., 0] = np.arange(20, dtype=np.uint8)[None, :]
    image_a[..., 1] = 80
    image_a[..., 2] = 160
    image_b = np.zeros((16, 20, 3), dtype=np.uint8)
    image_b[..., 0] = 200
    image_b[..., 1] = np.arange(16, dtype=np.uint8)[:, None]
    image_b[..., 2] = 40

    paths = [tmp_path / "a.png", tmp_path / "b.png"]
    _write_rgb_png(paths[0], image_a)
    _write_rgb_png(paths[1], image_b)
    data = pd.DataFrame(
        {
            "identity": ["animal_1", "animal_2"],
            "path": [path.name for path in paths],
        }
    )
    config = {"data": {"image_size": [16, 20], "wavelet": False}}

    mean, std = calculate_dataset_mean_std(data, root_dir=tmp_path, config=config)
    expected_mean, expected_std = _manual_stats(paths, image_size=(16, 20), wavelet=False)

    assert len(mean) == 3
    assert len(std) == 3
    np.testing.assert_allclose(mean, expected_mean, rtol=0, atol=1e-12)
    np.testing.assert_allclose(std, expected_std, rtol=0, atol=1e-12)


def test_calculate_dataset_mean_std_matches_manual_wavelet_calculation(tmp_path):
    rng = np.random.default_rng(42)
    paths = [tmp_path / "a.png", tmp_path / "b.png", tmp_path / "c.png"]
    for path in paths:
        image = rng.integers(0, 256, size=(18, 22, 3), dtype=np.uint8)
        _write_rgb_png(path, image)

    data = pd.DataFrame(
        {
            "identity": ["animal_1", "animal_2", "animal_3"],
            "path": [path.name for path in paths],
        }
    )
    config = {"data": {"image_size": [18, 22], "wavelet": True}}

    mean, std = calculate_dataset_mean_std(data, root_dir=tmp_path, config=config)
    expected_mean, expected_std = _manual_stats(paths, image_size=(18, 22), wavelet=True)

    assert len(mean) == 6
    assert len(std) == 6
    np.testing.assert_allclose(mean, expected_mean, rtol=0, atol=1e-12)
    np.testing.assert_allclose(std, expected_std, rtol=0, atol=1e-12)


def test_eval_transform_applies_same_clahe_used_for_stats():
    image = np.zeros((16, 20, 3), dtype=np.uint8)
    image[..., 0] = np.arange(20, dtype=np.uint8)[None, :] * 8
    image[..., 1] = np.arange(16, dtype=np.uint8)[:, None] * 10
    image[..., 2] = 80
    config = {
        "data": {
            "image_size": [16, 20],
            "wavelet": False,
            "transforms": {
                "mean": [0.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0],
            },
        }
    }

    transform = build_eval_transforms(config=config)
    transformed = transform(image=image)["image"].permute(1, 2, 0).numpy()
    clahe = A.CLAHE(
        clip_limit=CLAHE_CLIP_LIMIT,
        tile_grid_size=CLAHE_TILE_GRID_SIZE,
        p=1.0,
    )
    expected = clahe(image=image)["image"].astype(np.float32) / 255.0

    np.testing.assert_allclose(transformed, expected, rtol=0, atol=1e-6)
