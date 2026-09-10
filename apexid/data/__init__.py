__all__ = [
    "DEFAULT_CONFIG_PATH",
    "append_haar_wavelet_channels",
    "build_eval_transforms",
    "build_train_transforms",
    "calculate_dataset_mean_std",
]


def __getattr__(name):
    if name == "append_haar_wavelet_channels":
        from .DWT import append_haar_wavelet_channels

        return append_haar_wavelet_channels
    if name in {
        "DEFAULT_CONFIG_PATH",
        "build_eval_transforms",
        "build_train_transforms",
        "calculate_dataset_mean_std",
    }:
        from . import transforms as _transforms

        return getattr(_transforms, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
