import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import torch
from torch import nn


class EarlyStopping:
    """Monitor a validation metric and save the best model weights.

    The class is designed for the ApexID training layout where the full trainable
    model is usually a ``ModuleDict`` with ``"backbone"`` and ``"head"`` keys.
    It can also work with a normal module exposing ``.backbone`` or with separate
    ``backbone`` and ``head`` arguments.
    """

    def __init__(
        self,
        monitor="val_loss",
        mode="auto",
        patience=7,
        min_delta=0.0,
        output_dir="checkpoints",
        full_model_name="best_full_model.pt",
        backbone_name="best_backbone.pt",
        metadata_name="best_metadata.json",
        save_full_model=True,
        save_backbone=True,
        verbose=False,
        delta=None,
        path=None,
    ):
        """Create an early stopping monitor.

        Args:
            monitor (str): Metric name to monitor when ``__call__`` receives a
                dictionary. Examples: ``"val_loss"``, ``"val_top1"``, ``"mAP"``.
            mode (str): ``"min"``, ``"max"``, or ``"auto"``. ``"min"`` treats
                lower values as better, ``"max"`` treats higher values as better,
                and ``"auto"`` chooses ``"min"`` for loss/error-like names and
                ``"max"`` otherwise.
            patience (int): Number of consecutive non-improving calls tolerated
                before ``early_stop`` becomes ``True``.
            min_delta (float): Minimum metric improvement needed to reset the
                patience counter.
            output_dir (str or pathlib.Path): Directory where best weights and
                metadata are saved.
            full_model_name (str): Filename for the best full model state dict.
                For ApexID training this contains backbone and head weights.
            backbone_name (str): Filename for the best backbone-only state dict.
            metadata_name (str): Filename for JSON metadata describing the best
                metric, epoch, and saved paths.
            save_full_model (bool): Whether to save full model weights on each
                improvement.
            save_backbone (bool): Whether to save backbone-only weights on each
                improvement.
            verbose (bool): Whether to print improvement and patience messages.
            delta (float, optional): Backward-compatible alias for ``min_delta``.
            path (str or pathlib.Path, optional): Backward-compatible path. If it
                points to a file, its parent is used as ``output_dir`` and its
                name becomes ``full_model_name``. If it points to a directory, it
                becomes ``output_dir``.

        Returns:
            None
        """
        if delta is not None:
            min_delta = delta
        if path is not None:
            path = Path(path)
            if path.suffix:
                output_dir = path.parent if str(path.parent) != "." else output_dir
                full_model_name = path.name
            else:
                output_dir = path

        self.monitor = str(monitor)
        self.mode = self._resolve_mode(mode, self.monitor)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.output_dir = Path(output_dir)
        self.full_model_path = self.output_dir / full_model_name
        self.backbone_path = self.output_dir / backbone_name
        self.metadata_path = self.output_dir / metadata_name
        self.save_full_model = bool(save_full_model)
        self.save_backbone = bool(save_backbone)
        self.verbose = bool(verbose)

        if self.patience < 0:
            raise ValueError("patience must be >= 0.")
        if self.mode not in {"min", "max"}:
            raise ValueError("mode must be 'min', 'max', or 'auto'.")
        if not self.save_full_model and not self.save_backbone:
            raise ValueError("At least one of save_full_model or save_backbone must be True.")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.counter = 0
        self.best_metric = None
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.val_loss_min = math.inf

    @staticmethod
    def _resolve_mode(mode, monitor):
        """Resolve ``auto`` mode into an explicit optimization direction."""
        mode = str(mode).lower()
        if mode != "auto":
            return mode

        monitor = str(monitor).lower()
        lower_is_better_tokens = ("loss", "error", "eer", "fpr", "fnr")
        return "min" if any(token in monitor for token in lower_is_better_tokens) else "max"

    @staticmethod
    def _unwrap_module(module):
        """Return the underlying module when wrapped by DataParallel/DDP."""
        while hasattr(module, "module"):
            module = module.module
        return module

    @staticmethod
    def _state_dict_to_cpu(state_dict):
        """Move a state dict to CPU before saving it."""
        return OrderedDict(
            (key, value.detach().cpu() if torch.is_tensor(value) else value)
            for key, value in state_dict.items()
        )

    def _score(self, metric):
        """Convert a metric into a score where larger is always better."""
        return -metric if self.mode == "min" else metric

    def _extract_metric(self, value):
        """Extract the monitored scalar metric from a number or metrics dict."""
        if isinstance(value, Mapping):
            if self.monitor not in value:
                available = ", ".join(sorted(str(key) for key in value.keys()))
                raise KeyError(
                    f"Metric '{self.monitor}' was not found. Available metrics: {available}"
                )
            value = value[self.monitor]

        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError("Monitored metric tensor must contain exactly one value.")
            value = value.detach().cpu().item()

        metric = float(value)
        if not math.isfinite(metric):
            raise ValueError(f"Monitored metric '{self.monitor}' must be finite, got {metric}.")
        return metric

    def _is_improvement(self, metric):
        """Return whether ``metric`` improves on the current best metric."""
        if self.best_metric is None:
            return True
        if self.mode == "min":
            return metric < (self.best_metric - self.min_delta)
        return metric > (self.best_metric + self.min_delta)

    def _find_backbone(self, model=None, backbone=None):
        """Find a backbone module from explicit input or a combined model."""
        if backbone is not None:
            return self._unwrap_module(backbone)
        if model is None:
            return None

        model = self._unwrap_module(model)
        if isinstance(model, (nn.ModuleDict, Mapping)) and "backbone" in model:
            return self._unwrap_module(model["backbone"])
        if hasattr(model, "backbone"):
            return self._unwrap_module(model.backbone)
        return None

    def _full_state_dict(self, model=None, backbone=None, head=None):
        """Build the full model state dict from a combined model or modules."""
        if model is not None:
            model = self._unwrap_module(model)
            return self._state_dict_to_cpu(model.state_dict())

        if backbone is None or head is None:
            raise ValueError(
                "Saving full model weights requires either model=... or both "
                "backbone=... and head=...."
            )

        state = OrderedDict()
        backbone = self._unwrap_module(backbone)
        head = self._unwrap_module(head)
        for key, value in backbone.state_dict().items():
            state[f"backbone.{key}"] = value
        for key, value in head.state_dict().items():
            state[f"head.{key}"] = value
        return self._state_dict_to_cpu(state)

    def _backbone_state_dict(self, model=None, backbone=None):
        """Build the backbone-only state dict."""
        backbone = self._find_backbone(model=model, backbone=backbone)
        if backbone is None:
            raise ValueError(
                "Saving backbone weights requires backbone=..., a model with "
                "model.backbone, or a ModuleDict/mapping with key 'backbone'."
            )
        return self._state_dict_to_cpu(backbone.state_dict())

    def _save_metadata(self, metric, epoch, extra_state=None):
        """Write JSON metadata for the currently best saved weights."""
        metadata = {
            "monitor": self.monitor,
            "mode": self.mode,
            "best_metric": metric,
            "best_epoch": epoch,
            "full_model_path": str(self.full_model_path) if self.save_full_model else None,
            "backbone_path": str(self.backbone_path) if self.save_backbone else None,
        }
        if extra_state:
            for key, value in extra_state.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    metadata[key] = value
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

    def _save_best(self, metric, model=None, backbone=None, head=None, epoch=None, extra_state=None):
        """Save full model and/or backbone weights for a new best metric."""
        if self.save_full_model:
            torch.save(
                self._full_state_dict(model=model, backbone=backbone, head=head),
                self.full_model_path,
            )

        if self.save_backbone:
            torch.save(
                self._backbone_state_dict(model=model, backbone=backbone),
                self.backbone_path,
            )

        self._save_metadata(metric=metric, epoch=epoch, extra_state=extra_state)

    def __call__(self, value, model=None, backbone=None, head=None, epoch=None, extra_state=None):
        """Update the monitor with a new metric value.

        Args:
            value (float or Mapping[str, float] or torch.Tensor): Metric value or
                metric dictionary containing ``self.monitor``.
            model (torch.nn.Module, optional): Full trainable model. In ApexID
                training this can be ``ModuleDict({"backbone": backbone,
                "head": head})``.
            backbone (torch.nn.Module, optional): Backbone module. Required for
                backbone saving when it cannot be inferred from ``model``.
            head (torch.nn.Module, optional): Head module. Required for full
                model saving only when ``model`` is not provided.
            epoch (int, optional): Epoch index written to metadata.
            extra_state (dict, optional): Extra JSON-serializable scalar metadata
                written to the metadata file.

        Returns:
            bool: ``True`` if the metric improved and best weights were saved,
            otherwise ``False``.
        """
        metric = self._extract_metric(value)

        if self._is_improvement(metric):
            self.best_metric = metric
            self.best_score = self._score(metric)
            self.best_epoch = epoch
            if self.mode == "min":
                self.val_loss_min = metric
            self.counter = 0
            self.early_stop = False
            self._save_best(
                metric=metric,
                model=model,
                backbone=backbone,
                head=head,
                epoch=epoch,
                extra_state=extra_state,
            )
            if self.verbose:
                print(
                    f"EarlyStopping: {self.monitor} improved to {metric:.6f}; "
                    f"saved best weights to {self.output_dir}."
                )
            return True

        self.counter += 1
        if self.verbose:
            print(
                f"EarlyStopping: no improvement in {self.monitor} "
                f"({self.counter}/{self.patience})."
            )
        if self.counter >= self.patience:
            self.early_stop = True
        return False


__all__ = ["EarlyStopping"]
