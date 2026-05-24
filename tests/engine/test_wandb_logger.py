import sys
from types import SimpleNamespace

import pytest

from apexid.engine.loggers import NullLogger
from apexid.engine.wandb_logger import WandbLogger, build_experiment_logger, sanitize_metrics


class FakeArtifact:
    def __init__(self, name, type, metadata=None):
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files = []

    def add_file(self, path):
        self.files.append(path)


class FakeRun:
    def __init__(self):
        self.logged_artifacts = []
        self.finished = False

    def log_artifact(self, artifact, aliases=None):
        self.logged_artifacts.append((artifact, aliases))
        return artifact

    def finish(self):
        self.finished = True


def test_sanitize_metrics_keeps_scalars_and_drops_arrays():
    import numpy as np
    import torch

    metrics = sanitize_metrics(
        {
            "train_loss": 1.25,
            "known/top1": np.float32(0.5),
            "flag": True,
            "tensor": torch.tensor(0.1),
            "bad": np.array([1.0, 2.0]),
        }
    )

    assert metrics["train_loss"] == 1.25
    assert metrics["known/top1"] == 0.5
    assert metrics["flag"] == 1
    assert metrics["tensor"] == pytest.approx(0.1)
    assert "bad" not in metrics


def test_build_experiment_logger_returns_null_when_disabled():
    logger = build_experiment_logger({"tracking": {"enabled": False}})

    assert isinstance(logger, NullLogger)


def test_wandb_logger_uses_fake_wandb_without_network(monkeypatch, tmp_path):
    run = FakeRun()
    calls = {"init": None, "logs": []}

    fake_wandb = SimpleNamespace(
        init=lambda **kwargs: calls.__setitem__("init", kwargs) or run,
        log=lambda metrics, step=None: calls["logs"].append((metrics, step)),
        Artifact=FakeArtifact,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    logger = WandbLogger(
        project="ApexID",
        entity="team",
        run_name="run-1",
        group="group-1",
        tags=["test"],
        mode="offline",
        config={"seed": 42},
        log_artifacts="best",
        block_on_artifact_upload=True,
    )
    artifact_path = tmp_path / "best_metadata.json"
    artifact_path.write_text("{}", encoding="utf-8")

    logger.log_metrics({"train_loss": 1.0}, step=1)
    logger.log_artifact(
        artifact_path,
        name="best_metadata",
        artifact_type="metadata",
        aliases=["best"],
        metadata={"epoch": 1},
    )
    logger.finish()

    assert calls["init"]["project"] == "ApexID"
    assert calls["init"]["entity"] == "team"
    assert calls["init"]["name"] == "run-1"
    assert calls["init"]["group"] == "group-1"
    assert calls["init"]["tags"] == ["test"]
    assert calls["init"]["mode"] == "offline"
    assert calls["init"]["config"] == {"seed": 42}
    assert calls["logs"] == [({"train_loss": 1.0}, 1)]
    assert len(run.logged_artifacts) == 1
    artifact, aliases = run.logged_artifacts[0]
    assert artifact.name == "best_metadata"
    assert artifact.type == "metadata"
    assert artifact.metadata == {"epoch": 1}
    assert artifact.files == [str(artifact_path)]
    assert aliases == ["best"]
    assert run.finished is True


def test_wandb_logger_metadata_mode_skips_model_weights(monkeypatch, tmp_path):
    run = FakeRun()
    fake_wandb = SimpleNamespace(
        init=lambda **kwargs: run,
        log=lambda metrics, step=None: None,
        Artifact=FakeArtifact,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    logger = WandbLogger(
        project="ApexID",
        mode="offline",
        log_artifacts="metadata",
        block_on_artifact_upload=True,
    )
    model_path = tmp_path / "best_full_model.pt"
    metadata_path = tmp_path / "best_metadata.json"
    model_path.write_bytes(b"weights")
    metadata_path.write_text("{}", encoding="utf-8")

    logger.log_artifact(model_path, name="best_full_model", artifact_type="model")
    logger.log_artifact(metadata_path, name="best_metadata", artifact_type="metadata")
    logger.finish()

    assert len(run.logged_artifacts) == 1
    artifact, _aliases = run.logged_artifacts[0]
    assert artifact.name == "best_metadata"


def test_wandb_logger_uploads_in_background_when_not_blocking(monkeypatch, tmp_path):
    run = FakeRun()
    fake_wandb = SimpleNamespace(
        init=lambda **kwargs: run,
        log=lambda metrics, step=None: None,
        Artifact=FakeArtifact,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    logger = WandbLogger(
        project="ApexID",
        mode="offline",
        log_artifacts="metadata",
        block_on_artifact_upload=False,
        artifact_finish_timeout=5,
    )
    metadata_path = tmp_path / "best_metadata.json"
    metadata_path.write_text("{}", encoding="utf-8")

    logger.log_artifact(metadata_path, name="best_metadata", artifact_type="metadata")
    logger.finish()

    assert len(run.logged_artifacts) == 1
