from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from albumentations.pytorch import ToTensorV2

from apexid.utils.config import DEFAULT_CONFIG_PATH, load_config

from .DWT import append_haar_wavelet_channels as _append_haar_wavelet_channels


def _standardize_columns(data_csv):
    """Normalize common CSV column names to ApexID's canonical schema.

    Args:
        data_csv (pandas.DataFrame or str): Dataset metadata as a DataFrame with
            shape ``(num_samples, num_columns)`` or a path to a CSV file with the
            same tabular shape. Recognized image path columns are renamed to
            ``path`` and recognized identity columns are renamed to ``identity``.

    Returns:
        pandas.DataFrame: A copied DataFrame with the same row count as the input
        and normalized column names where possible.
    """
    if not isinstance(data_csv, pd.DataFrame):
        data_csv = pd.read_csv(data_csv)

    df = data_csv.copy()
    rename_map = {}
    for col in df.columns:
        col_norm = str(col).strip().lower()
        if col_norm in {"path", "image_path", "img_path", "filepath", "file_path"}:
            rename_map[col] = "path"
        elif col_norm in {"identity", "id", "individual_id", "label"}:
            rename_map[col] = "identity"
    return df.rename(columns=rename_map)


def _as_hw(image_size):
    """Convert a configured image size into ``(height, width)``.

    Args:
        image_size (int or Sequence[int]): Either one integer for square images or
            a two-item sequence ``[height, width]`` from the config.

    Returns:
        tuple[int, int]: The validated image height and width used by OpenCV and
        Albumentations resize operations.

    Raises:
        ValueError: If ``image_size`` is missing or is not an integer or two-item
        sequence.
    """
    if image_size is None:
        raise ValueError("data.image_size must be set in configs/default.yaml.")
    if isinstance(image_size, int):
        return image_size, image_size
    if len(image_size) != 2:
        raise ValueError("image_size must be an int or a two-item [height, width] sequence.")
    return int(image_size[0]), int(image_size[1])


def _resolve_image_path(img_path, root_dir=None):
    """Resolve a dataset image path against an optional root directory.

    Args:
        img_path (str or pathlib.Path): Image path from the CSV. It may be an
            absolute path, a path relative to ``root_dir``, or only a filename.
        root_dir (str or pathlib.Path, optional): Base directory used to resolve
            relative paths.

    Returns:
        str: A path string. If an existing file is found, the returned value points
        to that file; otherwise the original path string is returned so the image
        loader can raise a clear file-read error.
    """
    img_path = str(img_path)
    if Path(img_path).exists():
        return img_path
    if root_dir is not None and not Path(img_path).is_absolute():
        candidate = Path(root_dir) / img_path
        if candidate.exists():
            return str(candidate)
    if root_dir is not None:
        candidate = Path(root_dir) / Path(img_path).name
        if candidate.exists():
            return str(candidate)
    return img_path


def _load_image_rgb(img_path):
    """Load an image from disk with OpenCV and convert BGR to RGB.

    Args:
        img_path (str or pathlib.Path): Path to an image readable by OpenCV.

    Returns:
        numpy.ndarray: RGB image with shape ``(height, width, 3)`` and dtype
        usually ``uint8``. Values are in ``[0, 255]`` before later transform
        normalization.

    Raises:
        FileNotFoundError: If OpenCV cannot read the image.
    """
    image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image with OpenCV: {img_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _image_to_float01(image, **kwargs):
    """Convert an image array to float32 in ``[0, 1]``.

    Args:
        image (numpy.ndarray): Image with shape ``(height, width, channels)``.
            Typical input is RGB ``uint8`` with three channels, but any channel
            count is preserved.
        **kwargs: Extra Albumentations ``A.Lambda`` keyword arguments. They are
            accepted for compatibility and ignored.

    Returns:
        numpy.ndarray: Float32 image with the same shape as ``image``. Values are
        divided by 255 only when the input appears to use ``[0, 255]`` scale.
    """
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    return image


def _get_transform_settings(config=None, config_path=DEFAULT_CONFIG_PATH):
    """Read and validate transform settings from the loaded config.

    Args:
        config (dict, optional): Already-loaded config dictionary. Expected keys
            are ``data.image_size``, ``data.wavelet``, ``data.transforms.mean``,
            and ``data.transforms.std``.
        config_path (str or pathlib.Path): YAML config path used when ``config``
            is ``None``.

    Returns:
        dict: Settings dictionary containing ``image_size``, ``wavelet``,
        ``mean``, and ``std``. Mean and std are returned as float tuples with
        shape ``(3,)`` for RGB or ``(6,)`` when wavelet channels are enabled.

    Raises:
        ValueError: If mean/std are missing or do not match the expected channel
        count.
    """
    cfg = load_config(config=config, config_path=config_path)
    data_cfg = cfg.get("data", {})
    transforms_cfg = data_cfg.get("transforms", {})

    image_size = data_cfg.get("image_size")
    wavelet = bool(data_cfg.get("wavelet", False))
    mean = transforms_cfg.get("mean")
    std = transforms_cfg.get("std")

    expected_channels = 6 if wavelet else 3
    if mean is None or std is None:
        raise ValueError(
            "data.transforms.mean and data.transforms.std must be set. "
            "Use calculate_dataset_mean_std(...) on your training CSV, then put the values in configs/default.yaml."
        )
    if len(mean) != expected_channels or len(std) != expected_channels:
        raise ValueError(
            f"Expected mean/std with {expected_channels} values for wavelet={wavelet}, "
            f"got mean={len(mean)} and std={len(std)}."
        )

    return {
        "image_size": image_size,
        "wavelet": wavelet,
        "mean": tuple(float(x) for x in mean),
        "std": tuple(float(x) for x in std),
    }


def _normalization(mean, std):
    """Create the Albumentations normalization stage.

    Args:
        mean (Sequence[float]): Per-channel mean with shape ``(channels,)``.
        std (Sequence[float]): Per-channel standard deviation with shape
            ``(channels,)``.

    Returns:
        albumentations.Normalize: Transform that expects float images in
        ``[0, 1]`` with shape ``(height, width, channels)`` and normalizes each
        channel using the provided statistics.
    """
    return A.Normalize(mean=mean, std=std, max_pixel_value=1.0)


def build_train_transforms(config=None, config_path=DEFAULT_CONFIG_PATH):
    """Build the training image transform pipeline from config.

    Args:
        config (dict, optional): Already-loaded project config. When provided, it
            must contain ``data.image_size``, ``data.wavelet``, and
            ``data.transforms.mean/std``.
        config_path (str or pathlib.Path): YAML config path used when ``config``
            is ``None``.

    Returns:
        albumentations.Compose: Training transform that expects an RGB image under
        the ``image`` key with shape ``(height, width, 3)`` and returns a PyTorch
        tensor with shape ``(3, configured_height, configured_width)`` when
        ``wavelet=False`` or ``(6, configured_height, configured_width)`` when
        ``wavelet=True``.

    Functionality:
        Resizes the image, applies light geometric augmentation, optionally adds
        Haar wavelet detail channels, normalizes by dataset statistics, and
        converts the output to a tensor.
    """
    settings = _get_transform_settings(config=config, config_path=config_path)
    height, width = _as_hw(settings["image_size"])
    to_float = (
        A.Lambda(image=_append_haar_wavelet_channels, p=1.0)
        if settings["wavelet"]
        else A.Lambda(image=_image_to_float01, p=1.0)
    )

    return A.Compose(
        [
            A.Resize(height=height, width=width),
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=(-0.03, 0.03),
                rotate=(-8, 8),
                border_mode=0,
                p=0.35,
            ),
            A.ToGray(
                num_output_channels=3,
                p=0.15,
            ),
            A.GaussianBlur(
                blur_limit=(2, 3),
                p=0.2,
            ),
            A.GaussNoise(
                std_range=(0.02, 0.05),
                p=0.35,
            ),
            A.PixelDropout(
                dropout_prob=0.005,
                per_channel=False,
                drop_value=0,
                p=0.10,
            ),
            to_float,
            _normalization(settings["mean"], settings["std"]),
            ToTensorV2(),
        ]
    )


def build_eval_transforms(config=None, config_path=DEFAULT_CONFIG_PATH):
    """Build the validation/test image transform pipeline from config.

    Args:
        config (dict, optional): Already-loaded project config. When provided, it
            must contain ``data.image_size``, ``data.wavelet``, and
            ``data.transforms.mean/std``.
        config_path (str or pathlib.Path): YAML config path used when ``config``
            is ``None``.

    Returns:
        albumentations.Compose: Evaluation transform that expects an RGB image
        under the ``image`` key with shape ``(height, width, 3)`` and returns a
        PyTorch tensor with shape ``(3, configured_height, configured_width)``
        when ``wavelet=False`` or ``(6, configured_height, configured_width)``
        when ``wavelet=True``.

    Functionality:
        Applies deterministic preprocessing only: resize, optional
        wavelet channel construction, dataset-stat normalization, and tensor
        conversion.
    """
    settings = _get_transform_settings(config=config, config_path=config_path)
    height, width = _as_hw(settings["image_size"])
    to_float = (
        A.Lambda(image=_append_haar_wavelet_channels, p=1.0)
        if settings["wavelet"]
        else A.Lambda(image=_image_to_float01, p=1.0)
    )

    return A.Compose(
        [
            A.Resize(height=height, width=width),
            to_float,
            _normalization(settings["mean"], settings["std"]),
            ToTensorV2(),
        ]
    )


def calculate_dataset_mean_std(
    data_csv,
    root_dir=None,
    config=None,
    config_path=DEFAULT_CONFIG_PATH,
    path_column="path",
    max_images=None,
):
    """Calculate dataset mean and standard deviation for transform normalization.

    Args:
        data_csv (pandas.DataFrame or str): Dataset metadata with shape
            ``(num_samples, num_columns)`` or a CSV path. It must contain an image
            path column after column standardization.
        root_dir (str or pathlib.Path, optional): Base directory used to resolve
            relative image paths from the CSV.
        config (dict, optional): Already-loaded config. Only ``data.image_size``
            and ``data.wavelet`` are used for this calculation.
        config_path (str or pathlib.Path): YAML config path used when ``config``
            is ``None``.
        path_column (str): Name of the standardized image path column.
        max_images (int, optional): If set, compute statistics from only the
            first ``max_images`` rows.

    Returns:
        tuple[list[float], list[float]]: ``(mean, std)`` lists with length ``3``
        for RGB or length ``6`` when wavelet channels are enabled. Values are
        computed after resizing and optional wavelet channel construction.

    Functionality:
        Streams images with OpenCV, converts them to RGB, applies the same resize
        and channel construction used by transforms, then computes stable
        per-channel statistics over all pixels.

    Raises:
        KeyError: If the configured path column is missing.
        ValueError: If no images are available for statistics.
    """
    settings = _get_transform_settings_for_stats(config=config, config_path=config_path)
    height, width = _as_hw(settings["image_size"])
    df = _standardize_columns(data_csv).reset_index(drop=True)
    if path_column not in df.columns:
        raise KeyError(f"Missing required column: '{path_column}'")
    if max_images is not None:
        df = df.iloc[: int(max_images)]

    channel_sum = None
    channel_sq_sum = None
    pixel_count = 0

    for img_path in df[path_column].tolist():
        img_path = _resolve_image_path(img_path, root_dir=root_dir)
        image = _load_image_rgb(img_path)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        image = (
            _append_haar_wavelet_channels(image)
            if settings["wavelet"]
            else _image_to_float01(image)
        )

        pixels = image.reshape(-1, image.shape[-1]).astype(np.float64)
        if channel_sum is None:
            channel_sum = np.zeros(pixels.shape[1], dtype=np.float64)
            channel_sq_sum = np.zeros(pixels.shape[1], dtype=np.float64)
        channel_sum += pixels.sum(axis=0)
        channel_sq_sum += np.square(pixels).sum(axis=0)
        pixel_count += pixels.shape[0]

    if pixel_count == 0:
        raise ValueError("No images were found for mean/std calculation.")

    mean = channel_sum / pixel_count
    variance = np.maximum(channel_sq_sum / pixel_count - np.square(mean), 1e-12)
    std = np.sqrt(variance)
    return mean.astype(float).tolist(), std.astype(float).tolist()


def _get_transform_settings_for_stats(config=None, config_path=DEFAULT_CONFIG_PATH):
    """Read only the config fields needed for mean/std calculation.

    Args:
        config (dict, optional): Already-loaded config dictionary.
        config_path (str or pathlib.Path): YAML config path used when ``config``
            is ``None``.

    Returns:
        dict: Settings containing ``image_size`` and ``wavelet``. ``image_size``
        is later converted to ``(height, width)``, and ``wavelet`` controls
        whether statistics are calculated over 3 RGB channels or 6 RGB+detail
        channels.
    """
    cfg = load_config(config=config, config_path=config_path)
    data_cfg = cfg.get("data", {})
    return {
        "image_size": data_cfg.get("image_size"),
        "wavelet": bool(data_cfg.get("wavelet", False)),
    }


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "build_eval_transforms",
    "build_train_transforms",
    "calculate_dataset_mean_std",
]
