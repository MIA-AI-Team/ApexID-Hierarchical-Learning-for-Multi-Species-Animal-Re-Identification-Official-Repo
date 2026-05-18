"""Discrete wavelet transform utilities for ApexID image preprocessing."""

import cv2
import numpy as np


def append_haar_wavelet_channels(image, **kwargs):
    """Append grayscale Haar detail bands to an RGB image.

    Args:
        image (numpy.ndarray): RGB image with shape ``(height, width, 3)``. Input
            may be ``uint8`` in ``[0, 255]`` or float in ``[0, 1]``.
        **kwargs: Extra Albumentations ``A.Lambda`` keyword arguments. They are
            accepted for compatibility and ignored.

    Returns:
        numpy.ndarray: Float32 image with shape ``(height, width, 6)``. Channels
        are ``RGB + LH + HL + HH``. RGB channels are scaled to ``[0, 1]`` and
        wavelet detail channels are shifted/clipped into ``[0, 1]``.

    Functionality:
        Computes a one-level grayscale Haar DWT manually, resizes the three
        detail bands back to the image size, and concatenates them with RGB so a
        model can consume texture-frequency information alongside color.
    """
    rgb = image.astype(np.float32)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0

    gray = (
        cv2.cvtColor((rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        / 255.0
    )
    height, width = gray.shape
    if height % 2:
        gray = gray[:-1, :]
        rgb = rgb[:-1, :, :]
    if width % 2:
        gray = gray[:, :-1]
        rgb = rgb[:, :-1, :]

    a = gray[0::2, 0::2]
    b = gray[0::2, 1::2]
    c = gray[1::2, 0::2]
    d = gray[1::2, 1::2]

    lh = (a - b + c - d) * 0.25
    hl = (a + b - c - d) * 0.25
    hh = (a - b - c + d) * 0.25

    target_size = (rgb.shape[1], rgb.shape[0])
    details = []
    for band in (lh, hl, hh):
        band = np.clip(band + 0.5, 0.0, 1.0)
        band = cv2.resize(band, target_size, interpolation=cv2.INTER_LINEAR)
        details.append(band[..., None].astype(np.float32))

    return np.concatenate([rgb.astype(np.float32), *details], axis=2)


_append_haar_wavelet_channels = append_haar_wavelet_channels


__all__ = ["append_haar_wavelet_channels", "_append_haar_wavelet_channels"]
