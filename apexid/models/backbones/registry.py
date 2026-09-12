from .convnext import (
    convnext_base,
    convnext_base_pt,
    convnext_large,
    convnext_large_pt,
    convnext_small,
    convnext_small_pt,
    convnext_tiny,
    convnext_tiny_pt,
)
from .convnextv2 import (
    convnextv2_base,
    convnextv2_base_pt,
    convnextv2_large,
    convnextv2_large_pt,
    convnextv2_tiny,
    convnextv2_tiny_pt,
)
from .dinov3_convnext import (
    dinov3_convnext_base,
    dinov3_convnext_base_pt,
    dinov3_convnext_large,
    dinov3_convnext_large_pt,
    dinov3_convnext_small,
    dinov3_convnext_small_pt,
    dinov3_convnext_tiny,
    dinov3_convnext_tiny_pt,
)
from .dinov3_vit import (
    dinov3_vitb16,
    dinov3_vitb16_pt,
    dinov3_vitl16,
    dinov3_vitl16_pt,
    dinov3_vits16,
    dinov3_vits16_pt,
    dinov3_vits16plus,
    dinov3_vits16plus_pt,
)
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
register_backbone("convnext_tiny")(convnext_tiny)
register_backbone("convnext_small")(convnext_small)
register_backbone("convnext_base")(convnext_base)
register_backbone("convnext_large")(convnext_large)
register_backbone("convnext_tiny_pt")(convnext_tiny_pt)
register_backbone("convnext_small_pt")(convnext_small_pt)
register_backbone("convnext_base_pt")(convnext_base_pt)
register_backbone("convnext_large_pt")(convnext_large_pt)
register_backbone("convnextv2_tiny")(convnextv2_tiny)
register_backbone("convnextv2_base")(convnextv2_base)
register_backbone("convnextv2_large")(convnextv2_large)
register_backbone("convnextv2_tiny_pt")(convnextv2_tiny_pt)
register_backbone("convnextv2_base_pt")(convnextv2_base_pt)
register_backbone("convnextv2_large_pt")(convnextv2_large_pt)
register_backbone("dinov3_convnext_tiny")(dinov3_convnext_tiny)
register_backbone("dinov3_convnext_small")(dinov3_convnext_small)
register_backbone("dinov3_convnext_base")(dinov3_convnext_base)
register_backbone("dinov3_convnext_large")(dinov3_convnext_large)
register_backbone("dinov3_convnext_tiny_pt")(dinov3_convnext_tiny_pt)
register_backbone("dinov3_convnext_small_pt")(dinov3_convnext_small_pt)
register_backbone("dinov3_convnext_base_pt")(dinov3_convnext_base_pt)
register_backbone("dinov3_convnext_large_pt")(dinov3_convnext_large_pt)
register_backbone("dinov3_vits16")(dinov3_vits16)
register_backbone("dinov3_vits16plus")(dinov3_vits16plus)
register_backbone("dinov3_vitb16")(dinov3_vitb16)
register_backbone("dinov3_vitl16")(dinov3_vitl16)
register_backbone("dinov3_vits16_pt")(dinov3_vits16_pt)
register_backbone("dinov3_vits16plus_pt")(dinov3_vits16plus_pt)
register_backbone("dinov3_vitb16_pt")(dinov3_vitb16_pt)
register_backbone("dinov3_vitl16_pt")(dinov3_vitl16_pt)
