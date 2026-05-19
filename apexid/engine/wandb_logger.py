from pathlib import Path

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
    ):
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "Weights & Biases tracking is enabled, but the 'wandb' package is not installed. "
                "Install it with `pip install wandb` or disable tracking.enabled."
            ) from exc

        self.wandb = wandb
        self.log_artifacts = str(log_artifacts or "none").lower()
        if self.log_artifacts not in {"best", "none", "false", "off"}:
            raise ValueError("tracking.log_artifacts must be one of: best, none.")

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
        self.wandb.log(dict(metrics), step=step)

    def log_artifact(self, path, name=None, artifact_type="model", aliases=None, metadata=None):
        if self.log_artifacts in {"none", "false", "off"}:
            return None

        path = Path(path)
        if not path.exists():
            return None

        artifact_name = name or path.stem
        artifact = self.wandb.Artifact(
            artifact_name,
            type=artifact_type,
            metadata=dict(metadata or {}),
        )
        artifact.add_file(str(path))
        return self.run.log_artifact(artifact, aliases=list(aliases or []))

    def finish(self):
        if self.run is not None:
            self.run.finish()
            self.run = None


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
    )


__all__ = ["WandbLogger", "build_experiment_logger"]
