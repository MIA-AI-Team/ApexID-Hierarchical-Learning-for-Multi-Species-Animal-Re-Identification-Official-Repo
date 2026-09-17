import math
from collections.abc import Mapping

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
from apexid.training.finetune import apply_finetune_phase


seed_everything(DEFAULT_SEED)


OPEN_SET_SUMMARY_ORDER = (
    "known/mAP",
    "known/top1",
    "known/top5",
    "known/top10",
    "known/BAKS",
    "known/cmc_rank20",
    "known/cmc_rank50",
    "unknown/ARI",
    "unknown/AMI",
    "unknown/v_measure",
    "unknown/BAUS",
    "rejection/AUROC",
    "rejection/FAR",
    "rejection/FPR95",
    "rejection/threshold_at_95_tpr",
    "rejection/known_tpr_at_threshold",
    "rejection/unknown_recall_at_threshold",
    "mixed/ARI",
    "mixed/F1",
    "mixed/NormAcc",
    "mixed/BAKS",
    "mixed/BAUS",
    "mixed/OSCR_AUC",
    "mixed/DIR_at_1FAR",
    "mixed/DIR_at_0_1FAR",
    "mixed/known_unknown_threshold",
    "validation/mixed_embeddings_reused",
    "validation/mixed_scores_reused",
    "validation/metrics_sampled",
    "validation/gallery_samples_used",
    "validation/gallery_samples_total",
    "validation/known_samples_used",
    "validation/known_samples_total",
    "validation/unknown_samples_used",
    "validation/unknown_samples_total",
    "validation/mixed_samples_used",
    "validation/mixed_samples_total",
    "validation/diagnostics_ran",
    "diagnostics/intra_similarity",
    "diagnostics/inter_similarity",
    "diagnostics/intra_inter_gap",
    "diagnostics/embedding_norm_mean",
    "diagnostics/embedding_norm_std",
    "diagnostics/embedding_norm_min",
    "diagnostics/embedding_norm_max",
    "diagnostics/alignment",
    "diagnostics/uniformity",
    "diagnostics/knn_top1",
    "diagnostics/knn_top5",
)

OPEN_SET_SUMMARY_GROUP_ORDER = (
    "known",
    "unknown",
    "rejection",
    "mixed",
    "validation",
    "diagnostics",
)

INTERNAL_VALIDATION_KEYS = {
    "embeddings",
    "metric_labels",
    "loss",
    "acc",
    "top1",
    "top5",
    "mAP",
    "micro_f1",
    "num_known",
    "num_samples",
}


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
        validation_metric_chunk_size=1024,
        checkpoint_dir="checkpoints",
        checkpoint_every_n_epochs=1,
        checkpoint_filename="last_checkpoint.pt",
        checkpoint_delete_previous=True,
        early_stopping_enabled=True,
        early_stopping_monitor="mixed/NormAcc",
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
        evaluation_config=None,
        finetune_config=None,
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
            validation_metric_chunk_size (int): Query chunk size used by
                validation retrieval metrics.
            checkpoint_dir (str): Directory for latest resumable checkpoint.
            checkpoint_every_n_epochs (int): Save checkpoint every N epochs.
            checkpoint_filename (str): Latest checkpoint filename or epoch format
                string.
            checkpoint_delete_previous (bool): Whether to keep only the latest
                checkpoint file.
            early_stopping_enabled (bool): Whether to monitor validation metrics.
            early_stopping_monitor (str): Metric key to monitor. Defaults to
                ``mixed/NormAcc``.
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
            evaluation_config (dict, optional): Open-set evaluation settings
                passed to validators that support three-split validation.

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
        self.validation_metric_chunk_size = int(validation_metric_chunk_size)
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every_n_epochs = int(checkpoint_every_n_epochs)
        self.checkpoint_filename = checkpoint_filename
        self.checkpoint_delete_previous = bool(checkpoint_delete_previous)
        self.metrics_base_seed = int(metrics_base_seed)
        self.experiment_logger = experiment_logger or NullLogger()
        self.embedding_plotter = embedding_plotter
        self.evaluation_config = dict(evaluation_config or {})
        self.finetune_config = dict(finetune_config or {}) if finetune_config else None
        self._finetune_phase = None

        if self.validation_every_n_epochs <= 0:
            raise ValueError("validation_every_n_epochs must be > 0.")
        if self.validation_metric_chunk_size <= 0:
            raise ValueError("validation_metric_chunk_size must be > 0.")
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
            metric_chunk_size=self.validation_metric_chunk_size,
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
            "val_loss": [],
            "val_top1": [],
            "val_top5": [],
            "val_mAP": [],
            "val_micro_f1": [],
        }

    @staticmethod
    def _is_scalar_metric(value):
        """Return whether a metric value is a scalar suitable for logs/prints."""
        if torch.is_tensor(value):
            return value.numel() == 1
        return isinstance(value, (int, float, bool))

    @classmethod
    def _ordered_open_set_metric_keys(cls, metrics):
        """Return slash-named open-set metrics in a stable display order."""
        ordered = [key for key in OPEN_SET_SUMMARY_ORDER if key in metrics]
        extras = sorted(
            key
            for key, value in metrics.items()
            if key not in ordered
            and key not in INTERNAL_VALIDATION_KEYS
            and "/" in str(key)
            and cls._is_scalar_metric(value)
        )
        return ordered + extras

    @classmethod
    def _grouped_open_set_metric_keys(cls, metrics):
        """Return ordered slash-named metrics grouped by their namespace."""
        grouped = {name: [] for name in OPEN_SET_SUMMARY_GROUP_ORDER}
        for key in cls._ordered_open_set_metric_keys(metrics):
            namespace = str(key).split("/", 1)[0]
            grouped.setdefault(namespace, []).append(key)
        return [(name, keys) for name, keys in grouped.items() if keys]

    @staticmethod
    def _summary_metric_name(key, group_name):
        """Return a compact display name for a grouped metric key."""
        prefix = f"{group_name}/"
        key = str(key)
        if key.startswith(prefix):
            return key[len(prefix):]
        return key

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
        """Print a stable grouped epoch summary when verbose output is enabled."""
        if not self.verbose:
            return

        header_parts = [
            f"Epoch {int(epoch) + 1}/{int(epochs)}",
            f"train_loss={_format_metric(train_metrics['loss'])}",
        ]
        if val_metrics is not None:
            header_parts.append(f"val_loss={_format_metric(val_metrics['loss'])}")
        header_parts.append(f"lr={self.optimizer.param_groups[0]['lr']:.2e}")

        lines = [" | ".join(header_parts)]
        if val_metrics is not None:
            for group_name, keys in self._grouped_open_set_metric_keys(val_metrics):
                metric_parts = [
                    f"{self._summary_metric_name(key, group_name)}={_format_metric(val_metrics[key])}"
                    for key in keys
                ]
                lines.append(f"  {group_name}: " + " | ".join(metric_parts))

        message = "\n".join(lines)
        if self.progress:
            tqdm.write(message)
        else:
            print(message)

    def train_one_epoch(
        self,
        train_loader,
        epoch,
        epochs,
        *,
        scheduler_enabled=True,
        backbone_train=True,
        phase_label="Train",
    ):
        """Run one training epoch.

        Args:
            train_loader (torch.utils.data.DataLoader): Loader yielding training
                batches.
            epoch (int): Zero-based epoch index.
            epochs (int): Total number of epochs.
            scheduler_enabled (bool): Whether to step per-batch schedulers.
            backbone_train (bool): Whether the backbone should be in train mode.
                Use ``False`` for frozen head-only warmup.
            phase_label (str): Progress-bar prefix for the phase.

        Returns:
            dict: Training metrics with key ``loss``.
        """
        self.backbone.train(bool(backbone_train))
        self.head.train()

        total_loss = 0.0
        total_samples = 0
        head_metric_totals = {}

        iterator = tqdm(
            train_loader,
            desc=f"{phase_label} [{epoch + 1}/{epochs}]",
            leave=False,
            dynamic_ncols=True,
            disable=not self.progress,
        )

        for batch in iterator:
            images, labels, _ = _unpack_batch(batch)
            species_targets = batch[3] if len(batch) >= 4 else None
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).long()
            if species_targets is not None:
                species_targets = species_targets.to(self.device, non_blocking=True).long()

            if not self._known_mask(labels).all():
                raise ValueError(
                    "Unknown/ignored labels appeared in the training loader. "
                    "Training must contain only known IDs."
                )

            self.optimizer.zero_grad(set_to_none=True)
            embeddings = self.backbone(images)
            loss, logits = _compute_loss_and_logits(
                self.head,
                self.loss,
                embeddings,
                labels,
                species_targets=species_targets,
            )
            loss.backward()

            if self.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.gradient_clip_norm), foreach=True)

            self.optimizer.step()
            if scheduler_enabled and isinstance(self.scheduler, torch.optim.lr_scheduler.CyclicLR):
                self.scheduler.step()

            batch_size = int(images.size(0))
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            for key, value in dict(getattr(self.head, "last_metrics", {}) or {}).items():
                if torch.is_tensor(value):
                    if value.numel() != 1:
                        continue
                    value = value.detach().cpu().item()
                if isinstance(value, (int, float, bool)) and math.isfinite(float(value)):
                    head_metric_totals[key] = head_metric_totals.get(key, 0.0) + float(value) * batch_size

            avg_loss = total_loss / max(total_samples, 1)
            lr = self.optimizer.param_groups[0]["lr"]
            iterator.set_postfix(
                {
                    "loss": _format_metric(avg_loss),
                    "lr": f"{lr:.2e}",
                }
            )

        result = {"loss": total_loss / max(total_samples, 1)}
        result.update({
            key: value / max(total_samples, 1)
            for key, value in head_metric_totals.items()
        })
        return result

    def _step_scheduler(self, epoch_metrics):
        """Step epoch-level schedulers after one epoch."""
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.CyclicLR):
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.SequentialLR):
            self.scheduler.step()
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            metric = epoch_metrics.get("val_loss", epoch_metrics["train_loss"])
            if metric is None or math.isnan(float(metric)):
                metric = epoch_metrics["train_loss"]
            self.scheduler.step(metric)
        else:
            self.scheduler.step()

    def _optimizer_lrs(self):
        """Return current optimizer learning rates."""
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def _optimizer_base_lrs(self):
        """Return optimizer base LRs before scheduler initialization effects."""
        scheduler_base_lrs = list(getattr(self.scheduler, "base_lrs", []) or [])
        lrs = []
        for index, group in enumerate(self.optimizer.param_groups):
            lr = group.get("initial_lr")
            if lr is None and index < len(scheduler_base_lrs):
                lr = scheduler_base_lrs[index]
            if lr is None:
                lr = group["lr"]
            lrs.append(float(lr))
        return lrs

    def _set_optimizer_lrs(self, lrs):
        """Set optimizer learning rates without changing scheduler state."""
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = float(lr)

    def _head_first_warmup_epochs(self):
        """Return extra head-only warmup epochs for active head-first finetuning."""
        if self.finetune_config is None:
            return 0
        strategy = str(self.finetune_config.get("strategy", "full")).lower()
        if strategy != "head_first":
            return 0
        return max(0, int(self.finetune_config.get("head_only_epochs", 3)))

    def _apply_and_log_finetune_phase(self, phase_epoch):
        """Apply finetune freeze state and log phase changes once."""
        if self.finetune_config is None:
            return None
        # Trainer only stores the finetune block; force enabled=True so
        # is_finetune_active() does not require model.backbone for "auto".
        finetune_cfg = {"finetune": {**self.finetune_config, "enabled": True}}
        phase = apply_finetune_phase(self.backbone, self.head, finetune_cfg, phase_epoch)
        if phase != self._finetune_phase:
            self._finetune_phase = phase
            if self.verbose:
                tqdm.write(
                    f"Finetune phase: {phase.get('phase')} "
                    f"(strategy={phase.get('strategy')}, "
                    f"backbone_trainable={phase.get('backbone_trainable')})"
                )
        return phase

    def _append_history(self, history, train_metrics, val_metrics):
        """Append one epoch of train/validation metrics to history."""
        previous_len = len(history.get("train_loss", []))
        history["train_loss"].append(float(train_metrics["loss"]))

        values = {}
        if val_metrics is None:
            values.update(
                {
                    "val_loss": float("nan"),
                    "val_top1": float("nan"),
                    "val_top5": float("nan"),
                    "val_mAP": float("nan"),
                    "val_micro_f1": float("nan"),
                }
            )
        else:
            values.update(
                {
                    "val_loss": float(val_metrics["loss"]),
                    "val_top1": float(val_metrics["top1"]),
                    "val_top5": float(val_metrics["top5"]),
                    "val_mAP": float(val_metrics["mAP"]),
                    "val_micro_f1": float(val_metrics["micro_f1"]),
                }
            )
            for key, value in val_metrics.items():
                if key in INTERNAL_VALIDATION_KEYS:
                    continue
                if torch.is_tensor(value):
                    if value.numel() != 1:
                        continue
                    value = value.detach().cpu().item()
                if isinstance(value, (int, float, bool)):
                    values[str(key)] = float(value)

        for key in values:
            if key not in history:
                history[key] = [float("nan")] * previous_len
        for key in list(history.keys()):
            if key == "train_loss":
                continue
            history[key].append(float(values.get(key, float("nan"))))

    @staticmethod
    def _prefixed_epoch_metrics(train_metrics, val_metrics):
        """Build a flat metric dictionary for checkpointing and early stopping."""
        metrics = {
            "train_loss": float(train_metrics["loss"]),
        }
        for key, value in train_metrics.items():
            if key == "loss":
                continue
            if torch.is_tensor(value):
                if value.numel() != 1:
                    continue
                value = value.detach().cpu().item()
            if isinstance(value, (int, float, bool)):
                metrics[str(key)] = float(value)
                metrics[f"train/{key}"] = float(value)
        if val_metrics is not None:
            metrics.update(
                {
                    "val_loss": float(val_metrics["loss"]),
                    "val_top1": float(val_metrics["top1"]),
                    "val_top5": float(val_metrics["top5"]),
                    "val_mAP": float(val_metrics["mAP"]),
                    "val_micro_f1": float(val_metrics["micro_f1"]),
                }
            )
            for key, value in val_metrics.items():
                if key in INTERNAL_VALIDATION_KEYS:
                    continue
                if torch.is_tensor(value):
                    if value.numel() != 1:
                        continue
                    value = value.detach().cpu().item()
                if isinstance(value, (int, float, bool)):
                    metrics[str(key)] = float(value)
        return metrics

    @staticmethod
    def _is_open_set_loader_bundle(val_loader):
        """Return whether ``val_loader`` is the three-split open-set bundle."""
        if not isinstance(val_loader, Mapping):
            return False
        required = {"gallery", "known", "unknown", "mixed"}
        return required.issubset(set(val_loader.keys()))

    def _run_validation(self, val_loader, epoch, epochs, return_embeddings=False):
        """Dispatch to the compatible single-loader or open-set validator path."""
        if self._is_open_set_loader_bundle(val_loader):
            validate_open_set = getattr(self.validator, "validate_open_set", None)
            if validate_open_set is None:
                raise ValueError("Open-set validation loaders require a validator with validate_open_set(...).")
            return validate_open_set(
                gallery_loader=val_loader["gallery"],
                known_loader=val_loader["known"],
                unknown_loader=val_loader["unknown"],
                mixed_loader=val_loader["mixed"],
                evaluation_config=self.evaluation_config,
                epoch=epoch,
                epochs=epochs,
                return_embeddings=return_embeddings,
            )
        if return_embeddings:
            return self.validator.validate(
                val_loader,
                epoch=epoch,
                epochs=epochs,
                return_embeddings=True,
            )
        return self.validator.validate(val_loader, epoch=epoch, epochs=epochs)

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
        warmup_epochs = self._head_first_warmup_epochs()

        try:
            if self.early_stopping is not None and self.early_stopping.early_stop:
                if self.verbose:
                    print("Early stopping was already triggered in the resumed checkpoint.")
                return history

            if int(start_epoch) == 0 and warmup_epochs > 0:
                scheduler_start_lrs = self._optimizer_lrs()
                self._set_optimizer_lrs(self._optimizer_base_lrs())
                try:
                    for warmup_epoch in range(warmup_epochs):
                        self._apply_and_log_finetune_phase(warmup_epoch)
                        self._set_train_loader_epoch(train_loader, warmup_epoch)
                        self.train_one_epoch(
                            train_loader,
                            warmup_epoch,
                            warmup_epochs,
                            scheduler_enabled=False,
                            backbone_train=False,
                            phase_label="Warmup",
                        )
                finally:
                    self._set_optimizer_lrs(scheduler_start_lrs)

            for epoch in range(int(start_epoch), int(epochs)):
                phase_epoch = warmup_epochs + epoch
                self._apply_and_log_finetune_phase(phase_epoch)
                self._set_train_loader_epoch(train_loader, phase_epoch)
                train_metrics = self.train_one_epoch(train_loader, epoch, epochs)
                run_validation = ((epoch + 1) % self.validation_every_n_epochs == 0) or (epoch == epochs - 1)
                return_embeddings = bool(self.embedding_plotter) and run_validation
                if run_validation:
                    val_metrics = self._run_validation(
                        val_loader,
                        epoch=epoch,
                        epochs=epochs,
                        return_embeddings=return_embeddings,
                    )
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
                self.experiment_logger.log_metrics(log_metrics)
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
                    early_stopping=self.early_stopping,
                    data_loaders={"train": train_loader, "validation": val_loader},
                )

                if self.early_stopping is not None and self.early_stopping.early_stop:
                    if self.verbose:
                        print("\nEarly stopping triggered.")
                    break
        finally:
            self.experiment_logger.finish()

        return history


__all__ = ["Trainer"]
