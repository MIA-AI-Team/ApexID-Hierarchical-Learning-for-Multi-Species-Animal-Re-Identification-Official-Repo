import math

import torch
from torch import nn
from tqdm.auto import tqdm

from apexid.engine.checkpoint import save_training_checkpoint
from apexid.engine.early_stopping import EarlyStopping
from apexid.engine.loggers import NullLogger
from apexid.engine.validator import (
    Validator,
    _compute_loss_and_logits,
    _format_metric,
    _unpack_batch,
)
from apexid.utils.seed import DEFAULT_SEED, seed_everything


seed_everything(DEFAULT_SEED)


class Trainer:
    """Training loop for ApexID backbone+head experiments."""

    def __init__(
        self,
        backbone,
        head,
        loss=None,
        optimizer=None,
        scheduler=None,
        validator=None,
        epochs=None,
        device=None,
        classification_ignore_index=-100,
        metric_ignore_index=None,
        progress=True,
        verbose=True,
        gradient_clip_norm=None,
        validation_every_n_epochs=1,
        checkpoint_dir="checkpoints",
        checkpoint_every_n_epochs=1,
        checkpoint_filename="last_checkpoint.pt",
        checkpoint_delete_previous=True,
        early_stopping_enabled=True,
        early_stopping_monitor="val_mAP",
        early_stopping_mode="max",
        early_stopping_patience=7,
        early_stopping_delta=0.0,
        early_stopping_min_delta=None,
        early_stopping_dir=None,
        early_stopping_full_model_name="best_full_model.pt",
        early_stopping_backbone_name="best_backbone.pt",
        early_stopping_metadata_name="best_metadata.json",
        early_stopping_path=None,
        metrics_base_seed=DEFAULT_SEED,
        experiment_logger=None,
        embedding_plotter=None,
    ):
        """Initialize the training helper.

        Args:
            backbone (torch.nn.Module): Feature extractor that returns embeddings.
            head (torch.nn.Module): Trainable classification or metric-learning
                head.
            loss (callable, optional): External objective for heads that return
                logits.
            optimizer (torch.optim.Optimizer, optional): Optimizer. If omitted,
                Adam with ``lr=1e-3`` is created.
            scheduler (torch.optim.lr_scheduler.LRScheduler, optional): Scheduler.
                ``CyclicLR`` is stepped after each optimizer update; other
                schedulers are stepped after each epoch.
            validator (Validator, optional): Validator instance. If omitted, one
                is created from the same backbone/head/loss.
            epochs (int, optional): Default epoch count used when ``train`` is
                called with ``epochs=None``.
            device (str or torch.device, optional): Training device.
            classification_ignore_index (int or None): Label disallowed in the
                training loader and ignored by validation loss/accuracy.
            metric_ignore_index (int or None): Metric label ignored by re-ID
                validation metrics.
            progress (bool): Whether to show tqdm progress bars.
            verbose (bool): Whether to print one-line epoch summaries.
            gradient_clip_norm (float, optional): Max gradient norm. ``None``
                disables clipping.
            validation_every_n_epochs (int): Run validation every N epochs.
            checkpoint_dir (str): Directory for latest resumable checkpoint.
            checkpoint_every_n_epochs (int): Save checkpoint every N epochs.
            checkpoint_filename (str): Latest checkpoint filename or epoch format
                string.
            checkpoint_delete_previous (bool): Whether to keep only the latest
                checkpoint file.
            early_stopping_enabled (bool): Whether to monitor validation metrics.
            early_stopping_monitor (str): Metric key to monitor. Defaults to
                ``val_mAP``.
            early_stopping_mode (str): ``max`` for mAP/top-k metrics, ``min`` for
                losses.
            early_stopping_patience (int): Non-improving validations tolerated.
            early_stopping_delta (float): Backward-compatible min-delta value.
            early_stopping_min_delta (float, optional): Explicit min-delta value.
            early_stopping_dir (str, optional): Directory for best full/backbone
                weights. Defaults to ``checkpoint_dir``.
            early_stopping_full_model_name (str): Best full model filename.
            early_stopping_backbone_name (str): Best backbone-only filename.
            early_stopping_metadata_name (str): Best metadata JSON filename.
            early_stopping_path (str, optional): Backward-compatible full-model
                path accepted by ``EarlyStopping``.
            metrics_base_seed (int): Stored for experiment reproducibility.
            experiment_logger (ExperimentLogger, optional): Logger used for
                epoch metrics and selected artifacts.
            embedding_plotter (optional): Object with ``plot_epoch`` and
                ``save_best`` methods used to visualize validation embeddings.

        Returns:
            None
        """
        self.backbone = backbone
        self.head = head
        self.loss = loss
        self.epochs = None if epochs is None else int(epochs)
        self.classification_ignore_index = classification_ignore_index
        self.metric_ignore_index = metric_ignore_index
        self.progress = bool(progress)
        self.verbose = bool(verbose)
        self.gradient_clip_norm = gradient_clip_norm
        self.validation_every_n_epochs = int(validation_every_n_epochs)
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every_n_epochs = int(checkpoint_every_n_epochs)
        self.checkpoint_filename = checkpoint_filename
        self.checkpoint_delete_previous = bool(checkpoint_delete_previous)
        self.metrics_base_seed = int(metrics_base_seed)
        self.experiment_logger = experiment_logger or NullLogger()
        self.embedding_plotter = embedding_plotter

        if self.validation_every_n_epochs <= 0:
            raise ValueError("validation_every_n_epochs must be > 0.")
        if self.checkpoint_every_n_epochs <= 0:
            raise ValueError("checkpoint_every_n_epochs must be > 0.")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = nn.ModuleDict({"backbone": self.backbone, "head": self.head}).to(self.device)

        if optimizer is None:
            adam_kwargs = {"lr": 1e-3}
            if self.device.type == "cuda":
                adam_kwargs["fused"] = True
            self.optimizer = torch.optim.Adam(self.model.parameters(), **adam_kwargs)
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler
        self.validator = validator or Validator(
            self.backbone,
            self.head,
            loss=self.loss,
            device=self.device,
            classification_ignore_index=self.classification_ignore_index,
            metric_ignore_index=self.metric_ignore_index,
            progress=self.progress,
        )

        min_delta = early_stopping_delta if early_stopping_min_delta is None else early_stopping_min_delta
        self.early_stopping = None
        if early_stopping_enabled:
            self.early_stopping = EarlyStopping(
                monitor=early_stopping_monitor,
                mode=early_stopping_mode,
                patience=early_stopping_patience,
                min_delta=min_delta,
                output_dir=early_stopping_dir or checkpoint_dir,
                full_model_name=early_stopping_full_model_name,
                backbone_name=early_stopping_backbone_name,
                metadata_name=early_stopping_metadata_name,
                verbose=self.verbose,
                path=early_stopping_path,
            )

    @staticmethod
    def empty_history():
        """Return an empty training history dictionary."""
        return {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_top1": [],
            "val_top5": [],
            "val_mAP": [],
            "val_micro_f1": [],
        }

    def _known_mask(self, labels):
        """Return training samples that are valid known identities."""
        if self.classification_ignore_index is None:
            return torch.ones_like(labels, dtype=torch.bool)
        return labels != self.classification_ignore_index

    @staticmethod
    def _set_train_loader_epoch(train_loader, epoch):
        """Synchronize epoch-aware samplers before consuming a loader."""
        samplers = (
            getattr(train_loader, "batch_sampler", None),
            getattr(train_loader, "sampler", None),
        )
        seen = set()
        for sampler in samplers:
            if sampler is None or id(sampler) in seen:
                continue
            seen.add(id(sampler))
            set_epoch = getattr(sampler, "set_epoch", None)
            if set_epoch is not None:
                set_epoch(int(epoch))

    def _print_epoch_summary(self, epoch, epochs, train_metrics, val_metrics):
        """Print a stable, one-line epoch summary when verbose output is enabled."""
        if not self.verbose:
            return

        parts = [
            f"Epoch {int(epoch) + 1}/{int(epochs)}",
            f"train_loss={_format_metric(train_metrics['loss'])}",
            f"train_acc={_format_metric(train_metrics['acc'])}",
        ]
        if val_metrics is not None:
            parts.extend(
                [
                    f"val_loss={_format_metric(val_metrics['loss'])}",
                    f"val_acc={_format_metric(val_metrics['acc'])}",
                    f"top1={_format_metric(val_metrics['top1'])}",
                    f"top5={_format_metric(val_metrics['top5'])}",
                    f"mAP={_format_metric(val_metrics['mAP'])}",
                    f"micro_f1={_format_metric(val_metrics['micro_f1'])}",
                ]
            )
        parts.append(f"lr={self.optimizer.param_groups[0]['lr']:.2e}")

        message = " | ".join(parts)
        if self.progress:
            tqdm.write(message)
        else:
            print(message)

    def train_one_epoch(self, train_loader, epoch, epochs):
        """Run one training epoch.

        Args:
            train_loader (torch.utils.data.DataLoader): Loader yielding training
                batches.
            epoch (int): Zero-based epoch index.
            epochs (int): Total number of epochs.

        Returns:
            dict: Training metrics with keys ``loss`` and ``acc``.
        """
        self.backbone.train()
        self.head.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        iterator = tqdm(
            train_loader,
            desc=f"Train [{epoch + 1}/{epochs}]",
            leave=False,
            dynamic_ncols=True,
            disable=not self.progress,
        )

        for batch in iterator:
            images, labels, _ = _unpack_batch(batch)
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).long()

            if not self._known_mask(labels).all():
                raise ValueError(
                    "Unknown/ignored labels appeared in the training loader. "
                    "Training must contain only known IDs."
                )

            self.optimizer.zero_grad(set_to_none=True)
            embeddings = self.backbone(images)
            loss, logits = _compute_loss_and_logits(self.head, self.loss, embeddings, labels)
            loss.backward()

            if self.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.gradient_clip_norm), foreach=True)

            self.optimizer.step()
            if isinstance(self.scheduler, torch.optim.lr_scheduler.CyclicLR):
                self.scheduler.step()

            batch_size = int(images.size(0))
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            preds = torch.argmax(logits, dim=1)
            total_correct += int((preds == labels).sum().item())

            avg_loss = total_loss / max(total_samples, 1)
            avg_acc = total_correct / max(total_samples, 1)
            lr = self.optimizer.param_groups[0]["lr"]
            iterator.set_postfix(
                {
                    "loss": _format_metric(avg_loss),
                    "acc": _format_metric(avg_acc),
                    "lr": f"{lr:.2e}",
                }
            )

        return {
            "loss": total_loss / max(total_samples, 1),
            "acc": total_correct / max(total_samples, 1),
        }

    def _step_scheduler(self, epoch_metrics):
        """Step epoch-level schedulers after one epoch."""
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.CyclicLR):
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            metric = epoch_metrics.get("val_loss", epoch_metrics["train_loss"])
            if metric is None or math.isnan(float(metric)):
                metric = epoch_metrics["train_loss"]
            self.scheduler.step(metric)
        else:
            self.scheduler.step()

    def _append_history(self, history, train_metrics, val_metrics):
        """Append one epoch of train/validation metrics to history."""
        history["train_loss"].append(float(train_metrics["loss"]))
        history["train_acc"].append(float(train_metrics["acc"]))
        if val_metrics is None:
            history["val_loss"].append(float("nan"))
            history["val_acc"].append(float("nan"))
            history["val_top1"].append(float("nan"))
            history["val_top5"].append(float("nan"))
            history["val_mAP"].append(float("nan"))
            history["val_micro_f1"].append(float("nan"))
            return

        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_acc"].append(float(val_metrics["acc"]))
        history["val_top1"].append(float(val_metrics["top1"]))
        history["val_top5"].append(float(val_metrics["top5"]))
        history["val_mAP"].append(float(val_metrics["mAP"]))
        history["val_micro_f1"].append(float(val_metrics["micro_f1"]))

    @staticmethod
    def _prefixed_epoch_metrics(train_metrics, val_metrics):
        """Build a flat metric dictionary for checkpointing and early stopping."""
        metrics = {
            "train_loss": float(train_metrics["loss"]),
            "train_acc": float(train_metrics["acc"]),
        }
        if val_metrics is not None:
            metrics.update(
                {
                    "val_loss": float(val_metrics["loss"]),
                    "val_acc": float(val_metrics["acc"]),
                    "val_top1": float(val_metrics["top1"]),
                    "val_top5": float(val_metrics["top5"]),
                    "val_mAP": float(val_metrics["mAP"]),
                    "val_micro_f1": float(val_metrics["micro_f1"]),
                }
            )
        return metrics

    def _epoch_log_metrics(self, epoch, epochs, epoch_metrics, val_metrics, early_stopping_improved=False):
        """Build the metric payload sent to experiment loggers."""
        metrics = dict(epoch_metrics)
        metrics.update(
            {
                "epoch": int(epoch) + 1,
                "epoch_index": int(epoch),
                "epochs": int(epochs),
                "lr": float(self.optimizer.param_groups[0]["lr"]),
            }
        )

        if val_metrics is not None:
            metrics["val_num_known"] = int(val_metrics.get("num_known", 0))
            metrics["val_num_samples"] = int(val_metrics.get("num_samples", 0))

        if self.early_stopping is not None:
            metrics["early_stopping_counter"] = int(self.early_stopping.counter)
            metrics["early_stopping_early_stop"] = int(self.early_stopping.early_stop)
            metrics["early_stopping_improved"] = int(bool(early_stopping_improved))
            if self.early_stopping.best_metric is not None:
                metrics["early_stopping_best_metric"] = float(self.early_stopping.best_metric)

        return metrics

    def _log_best_artifacts(self, epoch, metrics, embedding_artifacts=None):
        """Log best-model files produced by EarlyStopping."""
        if self.early_stopping is None:
            return

        aliases = ["best", f"epoch-{int(epoch) + 1}"]
        artifact_specs = [
            (self.early_stopping.full_model_path, "best_full_model", "model"),
            (self.early_stopping.backbone_path, "best_backbone", "model"),
            (self.early_stopping.metadata_path, "best_metadata", "metadata"),
        ]
        if embedding_artifacts is not None:
            artifact_specs.extend(
                [
                    (embedding_artifacts["png"], "best_embedding_space", "image"),
                    (embedding_artifacts["json"], "best_embedding_space_metadata", "metadata"),
                ]
            )
        for path, name, artifact_type in artifact_specs:
            self.experiment_logger.log_artifact(
                path,
                name=name,
                artifact_type=artifact_type,
                aliases=aliases,
                metadata=metrics,
            )

    def train(self, train_loader, val_loader, epochs=None, start_epoch=0, resume_history=None):
        """Train for multiple epochs.

        Args:
            train_loader (torch.utils.data.DataLoader): Training loader.
            val_loader (torch.utils.data.DataLoader): Validation loader.
            epochs (int, optional): Total epoch count. If omitted, uses
                ``self.epochs``.
            start_epoch (int): Zero-based starting epoch, usually
                ``checkpoint["next_epoch"]`` when resuming.
            resume_history (dict, optional): Existing history loaded from a
                checkpoint.

        Returns:
            dict: Training history.
        """
        epochs = self.epochs if epochs is None else int(epochs)
        if epochs is None:
            raise ValueError("epochs must be provided either to Trainer(...) or train(...).")

        history = resume_history if resume_history is not None else self.empty_history()

        try:
            for epoch in range(int(start_epoch), int(epochs)):
                self._set_train_loader_epoch(train_loader, epoch)
                train_metrics = self.train_one_epoch(train_loader, epoch, epochs)
                run_validation = ((epoch + 1) % self.validation_every_n_epochs == 0) or (epoch == epochs - 1)
                return_embeddings = bool(self.embedding_plotter) and run_validation
                if run_validation and return_embeddings:
                    val_metrics = self.validator.validate(
                        val_loader,
                        epoch=epoch,
                        epochs=epochs,
                        return_embeddings=True,
                    )
                elif run_validation:
                    val_metrics = self.validator.validate(val_loader, epoch=epoch, epochs=epochs)
                else:
                    val_metrics = None
                embedding_plot_artifacts = None
                if val_metrics is not None and self.embedding_plotter is not None:
                    embeddings = val_metrics.pop("embeddings", None)
                    metric_labels = val_metrics.pop("metric_labels", None)
                    if embeddings is None or metric_labels is None:
                        raise ValueError(
                            "Embedding plotting requires validator.validate(..., return_embeddings=True) "
                            "to return 'embeddings' and 'metric_labels'."
                        )
                    embedding_plot_artifacts = self.embedding_plotter.plot_epoch(
                        embeddings,
                        metric_labels,
                        epoch=epoch,
                    )
                self._append_history(history, train_metrics, val_metrics)

                epoch_metrics = self._prefixed_epoch_metrics(train_metrics, val_metrics)
                self._step_scheduler(epoch_metrics)

                self._print_epoch_summary(epoch, epochs, train_metrics, val_metrics)

                early_stopping_improved = False
                best_embedding_artifacts = None
                if self.early_stopping is not None and val_metrics is not None:
                    early_stopping_improved = self.early_stopping(epoch_metrics, model=self.model, epoch=epoch)
                    if early_stopping_improved and embedding_plot_artifacts is not None:
                        best_embedding_artifacts = self.embedding_plotter.save_best(embedding_plot_artifacts)

                log_metrics = self._epoch_log_metrics(
                    epoch,
                    epochs,
                    epoch_metrics,
                    val_metrics,
                    early_stopping_improved=early_stopping_improved,
                )
                self.experiment_logger.log_metrics(log_metrics, step=int(epoch) + 1)
                if early_stopping_improved:
                    self._log_best_artifacts(
                        epoch,
                        log_metrics,
                        embedding_artifacts=best_embedding_artifacts,
                    )

                save_training_checkpoint(
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    output_dir=self.checkpoint_dir,
                    every_n_epochs=self.checkpoint_every_n_epochs,
                    history=history,
                    metrics=epoch_metrics,
                    filename=self.checkpoint_filename,
                    delete_previous=self.checkpoint_delete_previous,
                    extra_state={
                        "classification_ignore_index": self.classification_ignore_index,
                        "metric_ignore_index": self.metric_ignore_index,
                    },
                )

                if self.early_stopping is not None and self.early_stopping.early_stop:
                    if self.verbose:
                        print("\nEarly stopping triggered.")
                    break
        finally:
            self.experiment_logger.finish()

        return history


__all__ = ["Trainer"]
