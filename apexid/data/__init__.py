"""Datasets, image preprocessing, and observation-aware partitioning."""

__all__ = ["DEFAULT_CONFIG_PATH", "build_eval_transforms", "build_train_transforms"]


def __getattr__(name):
    if name in __all__:
        from . import transforms

        return getattr(transforms, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
