from typing import Protocol


class ExperimentLogger(Protocol):
    """Small logging interface used by training loops."""

    def log_metrics(self, metrics, step=None):
        """Log scalar metrics for one training step or epoch."""

    def log_artifact(self, path, name=None, artifact_type="model", aliases=None, metadata=None):
        """Log a file artifact such as model weights or metadata."""

    def finish(self):
        """Close the logger run."""


class NullLogger:
    """No-op experiment logger used when tracking is disabled."""

    def log_metrics(self, metrics, step=None):
        return None

    def log_artifact(self, path, name=None, artifact_type="model", aliases=None, metadata=None):
        return None

    def finish(self):
        return None


__all__ = ["ExperimentLogger", "NullLogger"]
