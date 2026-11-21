from .convnext import convnext_small, convnext_small_pt
from .megadescriptor import megadescriptor_l_384, megadescriptor_l_384_pt
from .miewid import miewid_msv3, miewid_msv3_pt


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


register_backbone("convnext_small")(convnext_small)
register_backbone("convnext_small_pt")(convnext_small_pt)
register_backbone("miewid_msv3")(miewid_msv3)
register_backbone("miewid_msv3_pt")(miewid_msv3_pt)
register_backbone("megadescriptor_l_384")(megadescriptor_l_384)
register_backbone("megadescriptor_l_384_pt")(megadescriptor_l_384_pt)
