"""RGB input transforms for the ApexID encoder recipes."""

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

from apexid.utils.config import DEFAULT_CONFIG_PATH, load_config


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
        raise ValueError("data.image_size must be set in the model config.")
    if isinstance(image_size, int):
        return image_size, image_size
    if len(image_size) != 2:
        raise ValueError("image_size must be an int or a two-item [height, width] sequence.")
    return int(image_size[0]), int(image_size[1])


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


def _preprocessing_stages(config, config_path):
    data = load_config(config=config, config_path=config_path).get("data", {})
    if data.get("wavelet") or data.get("grayscale"):
        raise ValueError("ApexID model recipes require RGB input.")
    height, width = _as_hw(data.get("image_size"))
    transforms = data.get("transforms", {})
    mean, std = transforms.get("mean"), transforms.get("std")
    if mean is None or std is None or len(mean) != 3 or len(std) != 3:
        raise ValueError("Set three values each for data.transforms.mean and data.transforms.std.")
    resize = A.Resize(height=height, width=width)
    tensorize = [
        A.Lambda(image=_image_to_float01, p=1.0),
        A.Normalize(
            mean=tuple(float(value) for value in mean),
            std=tuple(float(value) for value in std),
            max_pixel_value=1.0,
        ),
        ToTensorV2(),
    ]
    return resize, tensorize


def build_train_transforms(config=None, config_path=DEFAULT_CONFIG_PATH):
    """Resize the RGB foreground, augment it, and return a normalized tensor."""
    resize, tensorize = _preprocessing_stages(config, config_path)
    return A.Compose(
        [
            resize,
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=(-0.03, 0.03),
                rotate=(-8, 8),
                border_mode=0,
                p=0.35,
            ),
            A.ToGray(num_output_channels=3, p=0.05),
            A.GaussianBlur(
                blur_limit=(2, 3),
                p=0.1,
            ),
            A.GaussNoise(
                std_range=(0.002, 0.005),
                p=0.05,
            ),
            A.PixelDropout(
                dropout_prob=0.005,
                per_channel=False,
                drop_value=0,
                p=0.03,
            ),
            *tensorize,
        ]
    )


def build_eval_transforms(config=None, config_path=DEFAULT_CONFIG_PATH):
    """Apply deterministic resizing and checkpoint-specific RGB normalization."""
    resize, tensorize = _preprocessing_stages(config, config_path)
    return A.Compose([resize, *tensorize])
