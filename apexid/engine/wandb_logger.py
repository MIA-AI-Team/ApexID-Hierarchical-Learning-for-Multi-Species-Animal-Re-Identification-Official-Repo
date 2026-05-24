import math
import threading
import warnings
from pathlib import Path

import numpy as np

from apexid.engine.loggers import NullLogger


def _as_dict(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "items"):
        return dict(value.items())
    raise TypeError(f"Expected a dict-like value, got {type(value).__name__}.")


def _is_enabled(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _normalize_log_artifacts(value):
    mode = str(value or "none").lower()
    if mode in {"none", "false", "off"}:
        return "none"
    if mode == "metadata":
        return "metadata"
    if mode == "best":
        return "best"
    raise ValueError("tracking.log_artifacts must be one of: none, metadata, best.")


def _sanitize_scalar(value):
    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None and torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    elif isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return float(value)
        return float(value)
    return None


def sanitize_metrics(metrics):
    """Return only scalar metrics suitable for ``wandb.log``."""
    clean = {}
    for key, value in dict(metrics or {}).items():
        scalar = _sanitize_scalar(value)
        if scalar is not None:
            clean[str(key)] = scalar
    return clean


def _artifact_allowed(path, log_artifacts):
    if log_artifacts == "best":
        return True
    if log_artifacts != "metadata":
        return False
    suffix = Path(path).suffix.lower()
    return suffix in {".json", ".png", ".jpg", ".jpeg", ".webp"}


class WandbLogger:
    """Weights & Biases implementation of the ApexID experiment logger."""

    def __init__(
        self,
        project,
        entity=None,
        run_name=None,
        group=None,
        tags=None,
        mode="online",
        config=None,
        log_artifacts="best",
        block_on_artifact_upload=False,
        artifact_finish_timeout=300,
    ):
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "Weights & Biases tracking is enabled, but the 'wandb' package is not installed. "
                "Install it with `pip install wandb` or disable tracking.enabled."
            ) from exc

        self.wandb = wandb
        self.log_artifacts = _normalize_log_artifacts(log_artifacts)
        self.block_on_artifact_upload = bool(block_on_artifact_upload)
        self.artifact_finish_timeout = None if artifact_finish_timeout in {None, ""} else int(artifact_finish_timeout)
        self._pending_artifact_threads = []

        self.run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            group=group,
            tags=list(tags or []),
            mode=mode,
            config=config,
        )

    def log_metrics(self, metrics, step=None):
        payload = sanitize_metrics(metrics)
        if not payload:
            return None
        self.wandb.log(payload, step=step)
        return payload

    def _upload_artifact(self, artifact, aliases):
        return self.run.log_artifact(artifact, aliases=list(aliases or []))

    def log_artifact(self, path, name=None, artifact_type="model", aliases=None, metadata=None):
        if self.log_artifacts == "none":
            return None

        path = Path(path)
        if not path.exists():
            return None
        if not _artifact_allowed(path, self.log_artifacts):
            return None

        artifact_name = name or path.stem
        artifact = self.wandb.Artifact(
            artifact_name,
            type=artifact_type,
            metadata=sanitize_metrics(metadata or {}),
        )
        artifact.add_file(str(path))

        if self.block_on_artifact_upload:
            return self._upload_artifact(artifact, aliases)

        thread = threading.Thread(
            target=self._upload_artifact_safe,
            args=(artifact, aliases, path),
            name=f"wandb-artifact-{artifact_name}",
            daemon=True,
        )
        thread.start()
        self._pending_artifact_threads.append(thread)
        return artifact

    def _upload_artifact_safe(self, artifact, aliases, path):
        try:
            self._upload_artifact(artifact, aliases)
        except Exception as exc:
            warnings.warn(f"W&B artifact upload failed for {path}: {exc}", stacklevel=2)

    def _join_pending_artifact_uploads(self):
        pending = [thread for thread in self._pending_artifact_threads if thread.is_alive()]
        if not pending:
            return

        if self.artifact_finish_timeout is None or self.artifact_finish_timeout <= 0:
            return

        per_thread_timeout = max(1, int(self.artifact_finish_timeout) // max(len(pending), 1))
        for thread in pending:
            thread.join(timeout=per_thread_timeout)
            if thread.is_alive():
                warnings.warn(
                    "W&B artifact upload is still running; run.finish() may block until uploads "
                    "complete. Use tracking.log_artifacts=none or metadata on slow networks, or "
                    "set tracking.block_on_artifact_upload=true only when you need synchronous uploads.",
                    stacklevel=2,
                )
                break

    def finish(self):
        if self.run is None:
            return

        self._join_pending_artifact_uploads()
        self.run.finish()
        self.run = None
        self._pending_artifact_threads.clear()


def build_experiment_logger(cfg, metadata=None):
    """Build the configured experiment logger.

    Tracking is disabled unless ``tracking.enabled`` is truthy.
    """
    cfg = _as_dict(cfg)
    tracking_cfg = _as_dict(cfg.get("tracking", {}))
    if not _is_enabled(tracking_cfg.get("enabled", False)):
        return NullLogger()

    provider = str(tracking_cfg.get("provider", "wandb")).lower()
    if provider != "wandb":
        raise ValueError("Unknown tracking.provider '{}'. Available providers: wandb.".format(provider))

    experiment_cfg = _as_dict(cfg.get("experiment", {}))
    run_config = dict(cfg)
    if metadata:
        run_config["derived"] = dict(metadata)

    return WandbLogger(
        project=tracking_cfg.get("project") or "ApexID",
        entity=tracking_cfg.get("entity"),
        run_name=tracking_cfg.get("run_name") or experiment_cfg.get("name"),
        group=tracking_cfg.get("group"),
        tags=tracking_cfg.get("tags") or [],
        mode=tracking_cfg.get("mode", "online"),
        config=run_config,
        log_artifacts=tracking_cfg.get("log_artifacts", "best"),
        block_on_artifact_upload=_is_enabled(tracking_cfg.get("block_on_artifact_upload", False)),
        artifact_finish_timeout=tracking_cfg.get("artifact_finish_timeout", 300),
    )


__all__ = ["WandbLogger", "build_experiment_logger", "sanitize_metrics"]
