from .osnet import (
    osnet_ibn_x1_0,
    osnet_x0_25,
    osnet_x0_5,
    osnet_x0_75,
    osnet_x1_0,
)
from .osnet_pt import (
    osnet_ibn_x1_0_pt,
    osnet_x0_25_pt,
    osnet_x0_5_pt,
    osnet_x0_75_pt,
    osnet_x1_0_pt,
)


BACKBONE_REGISTRY = {}


def register_backbone(name):
    """Register a backbone builder by experiment-facing name."""

    def decorator(builder):
        key = name.lower()
        if key in BACKBONE_REGISTRY:
            raise ValueError(f"Backbone '{name}' is already registered.")
        BACKBONE_REGISTRY[key] = builder
        return builder

    return decorator


def build_backbone(name, **kwargs):
    """Build a registered backbone."""
    key = str(name).lower()
    if key not in BACKBONE_REGISTRY:
        available = ", ".join(available_backbones())
        raise ValueError(f"Unknown backbone '{name}'. Available backbones: {available}")
    return BACKBONE_REGISTRY[key](**kwargs)


def available_backbones():
    """Return registered backbone names in sorted order."""
    return sorted(BACKBONE_REGISTRY.keys())


register_backbone("osnet_x1_0")(osnet_x1_0)
register_backbone("osnet_x0_75")(osnet_x0_75)
register_backbone("osnet_x0_5")(osnet_x0_5)
register_backbone("osnet_x0_25")(osnet_x0_25)
register_backbone("osnet_ibn_x1_0")(osnet_ibn_x1_0)
register_backbone("osnet_x1_0_pt")(osnet_x1_0_pt)
register_backbone("osnet_x0_75_pt")(osnet_x0_75_pt)
register_backbone("osnet_x0_5_pt")(osnet_x0_5_pt)
register_backbone("osnet_x0_25_pt")(osnet_x0_25_pt)
register_backbone("osnet_ibn_x1_0_pt")(osnet_ibn_x1_0_pt)
