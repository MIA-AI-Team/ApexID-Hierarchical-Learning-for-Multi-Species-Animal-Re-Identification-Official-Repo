from torch import nn

from .apexid_model import ApexID
from .heads import (
    ArcFaceHead,
    ArcFaceSubCenterDynamicHead,
    DynArcFaceHead,
    SphereFace2FocalHead,
    SphereFace2Head,
)
from ..losses import DCMIPLoss
from ..utils.seed import DEFAULT_SEED, seed_everything


HEAD_REGISTRY = {
    "arcface": ArcFaceHead,
    "arcface_head": ArcFaceHead,
    "arcface_subcenter_dynamic": ArcFaceSubCenterDynamicHead,
    "arcface_subcenter": ArcFaceSubCenterDynamicHead,
    "subcenter_arcface": ArcFaceSubCenterDynamicHead,
    "dynarcface": DynArcFaceHead,
    "dynamic_arcface": DynArcFaceHead,
    "dyn_arcface": DynArcFaceHead,
    "sphereface2": SphereFace2Head,
    "sphereface2_head": SphereFace2Head,
    "sphereface2_focal": SphereFace2FocalHead,
    "sphereface2_focal_head": SphereFace2FocalHead,
    "sphereface2focal": SphereFace2FocalHead,
}

INTERNAL_LOSS_HEADS = {
    "sphereface2",
    "sphereface2_head",
    "sphereface2_focal",
    "sphereface2_focal_head",
    "sphereface2focal",
}


def _as_dict(cfg):
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    if hasattr(cfg, "items"):
        return dict(cfg.items())
    raise TypeError(f"Expected a dict-like config, got {type(cfg).__name__}.")


def _section(cfg, name):
    cfg = _as_dict(cfg)
    value = cfg.get(name, {})
    return _as_dict(value)


def _normalise_name(name):
    if name is None:
        return None
    return str(name).lower()


def _is_null_name(name):
    return _normalise_name(name) in {None, "", "none", "null"}


def _is_auto(value):
    return value is None or (
        isinstance(value, str) and value.strip().lower() in {"", "auto", "none", "null"}
    )


def _resolve_input_channels(cfg, model_cfg, backbone_kwargs):
    value = model_cfg.get("input_channels")
    backbone_value = backbone_kwargs.pop("input_channels", None)

    if _is_auto(value):
        value = backbone_value

    if _is_auto(value):
        data_cfg = _section(cfg, "data") if "data" in _as_dict(cfg) else {}
        value = 6 if bool(data_cfg.get("wavelet", False)) else 3

    input_channels = int(value)
    if input_channels <= 0:
        raise ValueError("model.input_channels must be a positive integer.")
    return input_channels


def build_model(cfg_or_kwargs=None):
    """Build the canonical ApexID embedding model from config or kwargs."""
    cfg = _as_dict(cfg_or_kwargs)
    seed_everything(cfg.get("seed", DEFAULT_SEED) or DEFAULT_SEED)
    model_cfg = _section(cfg, "model") if "model" in cfg else cfg

    model_name = _normalise_name(model_cfg.get("name", "apexid"))
    if model_name != "apexid":
        raise ValueError("Unknown model '{}'. Available models: apexid".format(model_cfg.get("name")))

    embedding_dim = int(model_cfg.get("embedding_dim", 512))
    backbone_cfg = _as_dict(model_cfg.get("backbone", {}))
    backbone_name = backbone_cfg.get("name", "osnet_x1_0")
    backbone_kwargs = dict(_as_dict(backbone_cfg.get("kwargs", {})))
    input_channels = _resolve_input_channels(cfg, model_cfg, backbone_kwargs)
    return ApexID(
        backbone_name=backbone_name,
        embedding_dim=embedding_dim,
        input_channels=input_channels,
        backbone_kwargs=backbone_kwargs,
    )


def build_head(name, embedding_dim, num_classes, **kwargs):
    """Build a trainable metric/classification head."""
    key = _normalise_name(name)
    if key not in HEAD_REGISTRY:
        available = ", ".join(sorted(HEAD_REGISTRY.keys()))
        raise ValueError(f"Unknown head '{name}'. Available heads: {available}")
    return HEAD_REGISTRY[key](
        in_features=int(embedding_dim),
        out_features=int(num_classes),
        **kwargs,
    )


def build_loss(name, **kwargs):
    """Build objective-only losses used outside trainable margin heads."""
    key = _normalise_name(name)
    if _is_null_name(key):
        return None
    if key in {"cross_entropy", "ce", "crossentropy"}:
        return nn.CrossEntropyLoss(**kwargs)
    if key == "dcmip":
        return DCMIPLoss(**kwargs)
    raise ValueError("Unknown loss '{}'. Available losses: cross_entropy, dcmip".format(name))


def build_components(cfg, num_classes):
    """Build ``(model, head, criterion)`` from an experiment config."""
    cfg = _as_dict(cfg)
    model = build_model(cfg)

    head_cfg = _section(cfg, "head")
    head_name = head_cfg.get("name", "sphereface2")
    head_key = _normalise_name(head_name)
    head_kwargs = dict(_as_dict(head_cfg.get("kwargs", {})))
    head_kwargs.pop("margin_min", None)
    head_kwargs.pop("margin_max", None)
    head = build_head(
        head_name,
        embedding_dim=model.embedding_dim,
        num_classes=num_classes,
        **head_kwargs,
    )

    loss_cfg = _section(cfg, "loss")
    loss_name = loss_cfg.get("name")
    if head_key in INTERNAL_LOSS_HEADS and not _is_null_name(loss_name):
        raise ValueError(
            "SphereFace2 heads compute their own loss. Set loss.name to null "
            "when using sphereface2 or sphereface2_focal."
        )
    criterion = build_loss(loss_name, **_as_dict(loss_cfg.get("kwargs", {})))
    return model, head, criterion
