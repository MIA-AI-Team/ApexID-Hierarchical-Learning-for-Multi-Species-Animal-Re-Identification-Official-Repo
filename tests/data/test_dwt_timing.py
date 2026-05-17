import statistics
import time

import cv2
import numpy as np

from apexid.data.transforms import _append_haar_wavelet_channels, _image_to_float01


def _write_rgb_png(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def _load_rgb(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert bgr is not None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _elapsed_seconds(fn, repeats):
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        durations.append(time.perf_counter() - start)
    return durations


def test_dwt_on_the_fly_vs_offline_loading_timing(tmp_path):
    rng = np.random.default_rng(42)
    num_images = 8
    image_size = (440, 440)
    repeats = 5
    image_paths = []
    dwt_paths = []

    for idx in range(num_images):
        image = rng.integers(0, 256, size=(*image_size, 3), dtype=np.uint8)
        image_path = tmp_path / f"image_{idx:02d}.png"
        dwt_path = tmp_path / f"image_{idx:02d}_dwt.npy"
        _write_rgb_png(image_path, image)

        loaded = _load_rgb(image_path)
        dwt_channels = _append_haar_wavelet_channels(loaded)[..., 3:]
        np.save(dwt_path, dwt_channels.astype(np.float32))

        image_paths.append(image_path)
        dwt_paths.append(dwt_path)

    def on_the_fly():
        outputs = []
        for image_path in image_paths:
            image = _load_rgb(image_path)
            outputs.append(_append_haar_wavelet_channels(image))
        return outputs

    def offline_load():
        outputs = []
        for image_path, dwt_path in zip(image_paths, dwt_paths):
            image = _image_to_float01(_load_rgb(image_path))
            dwt_channels = np.load(dwt_path)
            outputs.append(np.concatenate([image, dwt_channels], axis=2))
        return outputs

    on_the_fly_outputs = on_the_fly()
    offline_outputs = offline_load()

    assert len(on_the_fly_outputs) == num_images
    assert len(offline_outputs) == num_images
    assert on_the_fly_outputs[0].shape == (*image_size, 6)
    assert offline_outputs[0].shape == (*image_size, 6)
    np.testing.assert_allclose(on_the_fly_outputs[0], offline_outputs[0], rtol=0, atol=0)

    on_the_fly_durations = _elapsed_seconds(on_the_fly, repeats=repeats)
    offline_durations = _elapsed_seconds(offline_load, repeats=repeats)
    on_the_fly_median = statistics.median(on_the_fly_durations)
    offline_median = statistics.median(offline_durations)
    faster = "offline_load" if offline_median < on_the_fly_median else "on_the_fly"
    speedup = max(on_the_fly_median, offline_median) / min(on_the_fly_median, offline_median)

    print(
        "DWT timing "
        f"num_images={num_images}, image_size={image_size}, repeats={repeats}: "
        f"on_the_fly_runs={[round(x, 6) for x in on_the_fly_durations]} seconds, "
        f"offline_load_runs={[round(x, 6) for x in offline_durations]} seconds, "
        f"on_the_fly_median={on_the_fly_median:.6f} seconds, "
        f"offline_load_median={offline_median:.6f} seconds, "
        f"faster={faster}, speedup={speedup:.3f}x"
    )

    assert all(duration > 0 for duration in on_the_fly_durations)
    assert all(duration > 0 for duration in offline_durations)
