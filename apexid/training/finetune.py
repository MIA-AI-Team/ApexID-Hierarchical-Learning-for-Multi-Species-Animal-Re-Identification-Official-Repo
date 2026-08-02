from __future__ import annotations

import copy
from typing import Any

from torch import nn

DEFAULT_FINETUNE: dict[str, Any] = {
    "enabled": "auto",
    "strategy": "full",
    "head_only_epochs": 3,
    "unfreeze_top_fraction": 0.0,
    "optimizer": {
        "backbone_lr": 1.0e-5,
        "head_lr": 1.0e-4,
        "head_lr_multiplier": None,
        "backbone_weight_decay": None,
        "head_weight_decay": None,
    },
    "scheduler": {
        "name": "cosine_annealing_lr",
        "kwargs": {
            "T_max": None,
            "eta_min": 1.0e-6,
        },
        "warmup": {
            "enabled": True,
            "epochs": 5,
            "start_lr": 1.0e-6,
        },
    },
    "lr_range_test": {
        "enabled": False,
        "mode": "suggest_lr",
        "apply_to": "head",
        "backbone_lr_divisor": 10.0,
    },
}


def _as_dict(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "items"):
        return dict(value.items())
    raise TypeError(f"Expected a dict-like value, got {type(value).__name__}.")


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def is_pretrained_backbone(cfg) -> bool:
    """Return True when the config selects an ImageNet-pretrained backbone."""
    model_cfg = _as_dict(_as_dict(cfg).get("model", {}))
    backbone_cfg = _as_dict(model_cfg.get("backbone", {}))
    name = str(backbone_cfg.get("name", "")).lower()
    kwargs = _as_dict(backbone_cfg.get("kwargs", {}))
    if name.endswith("_pt"):
        return True
    return bool(kwargs.get("pretrained"))


def is_finetune_active(cfg) -> bool:
    """Return True when the finetune profile should be applied."""
    finetune_cfg = _as_dict(_as_dict(cfg).get("finetune", {}))
    enabled = finetune_cfg.get("enabled", "auto")
    if isinstance(enabled, str) and enabled.strip().lower() == "false":
        return False
    if enabled is False:
        return False
    if enabled is True:
        return True
    return is_pretrained_backbone(cfg)


def resolve_training_profile(cfg) -> dict:
    """Return a config copy with finetune defaults merged when active."""
    resolved = copy.deepcopy(_as_dict(cfg))
    if not is_finetune_active(resolved):
        return resolved

    finetune_cfg = _deep_merge(DEFAULT_FINETUNE, _as_dict(resolved.get("finetune", {})))
    resolved["finetune"] = finetune_cfg

    training_epochs = int(_as_dict(resolved.get("training", {})).get("epochs", 100))
    scheduler_cfg = finetune_cfg["scheduler"]
    scheduler_kwargs = dict(_as_dict(scheduler_cfg.get("kwargs", {})))
    if scheduler_kwargs.get("T_max") in {None, ""}:
        scheduler_kwargs["T_max"] = training_epochs
    scheduler_cfg = dict(scheduler_cfg)
    scheduler_cfg["kwargs"] = scheduler_kwargs
    finetune_cfg["scheduler"] = scheduler_cfg

    resolved["scheduler"] = {
        "name": scheduler_cfg.get("name", "cosine_annealing_lr"),
        "kwargs": scheduler_kwargs,
        "warmup": dict(_as_dict(scheduler_cfg.get("warmup", {}))),
    }

    finetune_opt = _as_dict(finetune_cfg.get("optimizer", {}))
    optimizer_cfg = _as_dict(resolved.get("optimizer", {}))
    optimizer_kwargs = dict(_as_dict(optimizer_cfg.get("kwargs", {})))
    backbone_lr = float(finetune_opt.get("backbone_lr", 1.0e-5))
    head_lr_multiplier = finetune_opt.get("head_lr_multiplier")
    if head_lr_multiplier not in {None, ""}:
        head_lr = backbone_lr * float(head_lr_multiplier)
    else:
        head_lr = float(finetune_opt.get("head_lr", 1.0e-4))
    shared_weight_decay = optimizer_kwargs.get("weight_decay")
    backbone_weight_decay = finetune_opt.get("backbone_weight_decay")
    head_weight_decay = finetune_opt.get("head_weight_decay")
    if backbone_weight_decay in {None, ""}:
        backbone_weight_decay = shared_weight_decay
    if head_weight_decay in {None, ""}:
        head_weight_decay = shared_weight_decay
    optimizer_kwargs.setdefault("lr", backbone_lr)
    optimizer_cfg["kwargs"] = optimizer_kwargs
    resolved["optimizer"] = optimizer_cfg
    resolved["_finetune_backbone_lr"] = backbone_lr
    resolved["_finetune_head_lr"] = head_lr
    if backbone_weight_decay not in {None, ""}:
        resolved["_finetune_backbone_weight_decay"] = float(backbone_weight_decay)
    if head_weight_decay not in {None, ""}:
        resolved["_finetune_head_weight_decay"] = float(head_weight_decay)

    if not bool(_as_dict(finetune_cfg.get("lr_range_test", {})).get("enabled", False)):
        lr_test_cfg = _as_dict(resolved.get("lr_range_test", {}))
        lr_test_cfg["enabled"] = False
        resolved["lr_range_test"] = lr_test_cfg

    return resolved


def finetune_lr_range_test_cfg(cfg) -> dict:
    """Return LR range test settings for the active training profile."""
    if is_finetune_active(cfg):
        finetune_cfg = _as_dict(_as_dict(cfg).get("finetune", {}))
        lr_cfg = _as_dict(finetune_cfg.get("lr_range_test", {}))
        if bool(lr_cfg.get("enabled", False)):
            merged = _deep_merge(_as_dict(cfg.get("lr_range_test", {})), lr_cfg)
            merged["enabled"] = True
            return merged
        return {"enabled": False}
    return _as_dict(_as_dict(cfg).get("lr_range_test", {}))


def should_run_lr_range_test(cfg, resume_path=None) -> bool:
    if resume_path not in {None, ""}:
        return False
    return bool(finetune_lr_range_test_cfg(cfg).get("enabled", False))


def collect_backbone_params(model) -> list[nn.Parameter]:
    """Collect trainable backbone parameters from an ApexID model."""
    return list(model.backbone.parameters())


def collect_head_params(head) -> list[nn.Parameter]:
    """Collect head parameters."""
    return list(head.parameters())


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def _backbone_module(model) -> nn.Module:
    return getattr(model, "backbone", model)


def _freeze_backbone_for_head_first(model, finetune_cfg) -> None:
    _ = finetune_cfg
    _set_requires_grad(_backbone_module(model), False)


def _unfreeze_backbone(model) -> None:
    _set_requires_grad(_backbone_module(model), True)


def apply_finetune_phase(model, head, cfg, epoch: int) -> dict[str, Any]:
    """Apply freeze/unfreeze rules for the current epoch.

    Returns metadata describing the active phase (for logging).
    """
    _ = head
    if not is_finetune_active(cfg):
        _unfreeze_backbone(model)
        return {"phase": "scratch", "strategy": None, "backbone_trainable": True}

    finetune_cfg = _as_dict(_as_dict(cfg).get("finetune", {}))
    strategy = str(finetune_cfg.get("strategy", "full")).lower()
    head_only_epochs = int(finetune_cfg.get("head_only_epochs", 3))

    if strategy == "head_first" and epoch < head_only_epochs:
        _freeze_backbone_for_head_first(model, finetune_cfg)
        return {
            "phase": "head_first",
            "strategy": strategy,
            "backbone_trainable": False,
            "head_only_epochs": head_only_epochs,
        }

    _unfreeze_backbone(model)
    return {
        "phase": "full" if strategy == "full" else "unfrozen",
        "strategy": strategy,
        "backbone_trainable": True,
        "head_only_epochs": head_only_epochs,
    }


def apply_suggest_lr_from_range_test(cfg, optimizer, result) -> None:
    """Apply LR range test suggestions to finetune param groups."""
    finetune_cfg = _as_dict(_as_dict(cfg).get("finetune", {}))
    lr_cfg = _as_dict(finetune_cfg.get("lr_range_test", {}))
    divisor = float(lr_cfg.get("backbone_lr_divisor", 10.0))
    suggested_max = float(result.suggested_max_lr)
    head_lr = suggested_max
    backbone_lr = suggested_max / divisor

    if len(optimizer.param_groups) == 1:
        optimizer.param_groups[0]["lr"] = suggested_max
        return

    for group in optimizer.param_groups:
        name = str(group.get("param_group_name", "")).lower()
        if name == "head":
            group["lr"] = head_lr
        elif name == "backbone":
            group["lr"] = backbone_lr
