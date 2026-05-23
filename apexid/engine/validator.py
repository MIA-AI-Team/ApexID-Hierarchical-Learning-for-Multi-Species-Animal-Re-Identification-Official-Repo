import math

import numpy as np
import torch
from tqdm.auto import tqdm

from apexid.evaluation import compute_reid_metrics, evaluate_all_validation_splits


def _format_metric(value):
    """Format a metric for progress-bar display."""
    if value is None:
        return "nan"
    try:
        value = float(value)
    except Exception:
        return "nan"
    if math.isnan(value) or math.isinf(value):
        return "nan"
    return f"{value:.4f}"


def _unpack_batch(batch):
    """Unpack a batch into images, classification labels, and metric labels.

    Args:
        batch (tuple or list): Batch shaped as ``(images, cls_labels)`` or
            ``(images, cls_labels, metric_labels)``.

    Returns:
        tuple: ``(images, cls_labels, metric_labels)``. When metric labels are not
        provided, classification labels are reused as metric labels.
    """
    if not isinstance(batch, (list, tuple)):
        raise ValueError(
            "Each batch must be a tuple/list like "
            "(images, cls_labels) or (images, cls_labels, metric_labels)."
        )
    if len(batch) == 2:
        images, cls_labels = batch
        metric_labels = cls_labels
    elif len(batch) >= 3:
        images, cls_labels, metric_labels = batch[:3]
    else:
        raise ValueError(
            "Batch must contain at least 2 elements: "
            "(images, cls_labels) or (images, cls_labels, metric_labels)."
        )
    return images, cls_labels, metric_labels


def _prediction_logits(head, embeddings):
    """Return non-margin prediction logits from a head exposing ``predict_logits``."""
    if hasattr(head, "predict_logits"):
        return head.predict_logits(embeddings)
    raise ValueError("Head returned a scalar loss but does not expose predict_logits(...).")


def _head_num_classes(head):
    """Infer the number of classifier identities from a head when available."""
    for attr in ("out_features", "num_class", "num_classes"):
        if hasattr(head, attr):
            return int(getattr(head, attr))
    return None


def _validate_labels(labels, num_classes, context="classification"):
    """Validate label dtype and range before invoking margin heads or losses."""
    if labels.dtype != torch.long:
        raise ValueError(f"{context} labels must have dtype torch.long, got {labels.dtype}.")
    if labels.ndim != 1:
        raise ValueError(f"{context} labels must be one-dimensional, got shape {tuple(labels.shape)}.")
    if labels.numel() == 0:
        raise ValueError(f"{context} labels must not be empty.")
    if num_classes is None:
        return

    min_label = int(labels.min().item())
    max_label = int(labels.max().item())
    if min_label < 0 or max_label >= int(num_classes):
        raise ValueError(
            f"{context} labels must be in range [0, {int(num_classes) - 1}], "
            f"got min={min_label}, max={max_label}."
        )


def _validate_logits(logits, labels, num_classes, context="classification"):
    """Validate that logits match the labels and classifier class count."""
    if not torch.is_tensor(logits):
        raise ValueError(f"{context} logits must be a torch.Tensor.")
    if logits.ndim != 2:
        raise ValueError(f"{context} logits must have shape [batch_size, num_classes].")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            f"{context} logits batch size must match labels. "
            f"Got logits={tuple(logits.shape)} and labels={tuple(labels.shape)}."
        )
    inferred_classes = int(logits.shape[1])
    if num_classes is None:
        num_classes = inferred_classes
    if inferred_classes != int(num_classes):
        raise ValueError(
            f"{context} logits class dimension must equal num_classes={int(num_classes)}, "
            f"got shape {tuple(logits.shape)}."
        )
    _validate_labels(labels, int(num_classes), context=context)


def _compute_loss_and_logits(head, criterion, embeddings, labels):
    """Compute loss and prediction logits for ApexID heads.

    Args:
        head (torch.nn.Module): Trainable classification or metric-learning head.
        criterion (callable or None): External loss used when ``head`` returns
            logits instead of a scalar loss.
        embeddings (torch.Tensor): Backbone embeddings with shape
            ``(batch_size, embedding_dim)``.
        labels (torch.Tensor): Classification labels with shape ``(batch_size,)``.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Scalar loss and class logits.

    Functionality:
        Supports ArcFace-style heads that return logits and SphereFace2-style
        heads that return scalar loss and expose ``predict_logits``.
    """
    num_classes = _head_num_classes(head)
    _validate_labels(labels, num_classes, context="classification")

    head_output = head(embeddings, labels)

    if torch.is_tensor(head_output) and head_output.ndim == 0:
        logits = _prediction_logits(head, embeddings)
        _validate_logits(logits, labels, num_classes, context="classification")
        return head_output, logits

    if criterion is None:
        raise ValueError("Head returned class logits but no external loss was provided.")

    _validate_logits(head_output, labels, num_classes, context="classification")
    return criterion(head_output, labels), head_output


def _loader_metadata(loader, num_samples):
    """Return optional path/source metadata when the loader exposes dataset rows."""
    dataset = getattr(loader, "dataset", None)
    data_csv = getattr(dataset, "data_csv", None)
    if data_csv is None or len(data_csv) != int(num_samples):
        return {}

    metadata = {}
    for attr, key in (
        (getattr(dataset, "path_column", "path"), "paths"),
        ("species", "species"),
        ("dataset", "dataset"),
    ):
        if attr in data_csv.columns:
            metadata[key] = data_csv[attr].astype(str).to_numpy(copy=True)
    return metadata


def extract_embeddings(backbone, dataloader, device=None, progress=True, desc="Embeddings"):
    """Extract CPU embeddings, classifier labels, metric labels, and metadata.

    Args:
        backbone (torch.nn.Module): Embedding model.
        dataloader (torch.utils.data.DataLoader): Loader yielding batches shaped
            like ``(images, cls_labels, metric_labels)``.
        device (str or torch.device, optional): Device used for model inference.
        progress (bool): Whether to show a tqdm progress bar.
        desc (str): Progress-bar description.

    Returns:
        dict: ``embeddings``, ``cls_labels``, ``metric_labels``, and optional
        ``paths``, ``species``, or ``dataset`` arrays when the loader exposes
        dataset metadata in the same order.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    backbone.eval()

    all_embeddings = []
    all_cls_labels = []
    all_metric_labels = []
    total_samples = 0

    iterator = tqdm(
        dataloader,
        desc=desc,
        leave=False,
        dynamic_ncols=True,
        disable=not progress,
    )
    with torch.no_grad():
        for batch in iterator:
            images, cls_labels, metric_labels = _unpack_batch(batch)
            images = images.to(device, non_blocking=True)
            embeddings = backbone(images)
            all_embeddings.append(embeddings.detach().cpu())
            all_cls_labels.append(cls_labels.detach().cpu())
            all_metric_labels.append(metric_labels.detach().cpu())
            total_samples += int(images.size(0))

    result = {
        "embeddings": torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty(0),
        "cls_labels": (
            torch.cat(all_cls_labels, dim=0)
            if all_cls_labels
            else torch.empty(0, dtype=torch.long)
        ),
        "metric_labels": (
            torch.cat(all_metric_labels, dim=0)
            if all_metric_labels
            else torch.empty(0, dtype=torch.long)
        ),
        "num_samples": int(total_samples),
    }
    result.update(_loader_metadata(dataloader, total_samples))
    return result


class Validator:
    """Validation loop for ApexID re-identification experiments."""

    def __init__(
        self,
        backbone,
        head,
        loss=None,
        device=None,
        classification_ignore_index=-100,
        metric_ignore_index=None,
        metric_chunk_size=1024,
        progress=True,
    ):
        """Initialize the validation helper.

        Args:
            backbone (torch.nn.Module): Embedding model.
            head (torch.nn.Module): Classification or metric-learning head.
            loss (callable, optional): External loss for heads that return logits.
            device (str or torch.device, optional): Device used for validation.
            classification_ignore_index (int or None): Classification label to
                ignore for validation loss and accuracy.
            metric_ignore_index (int or None): Metric label to ignore for re-ID
                metrics. Defaults to ``None`` so metric labels remain distinct.
            metric_chunk_size (int): Number of query embeddings processed at once
                when computing pairwise retrieval metrics.
            progress (bool): Whether to show a tqdm validation progress bar.

        Returns:
            None
        """
        self.backbone = backbone
        self.head = head
        self.loss = loss
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.classification_ignore_index = classification_ignore_index
        self.metric_ignore_index = metric_ignore_index
        self.metric_chunk_size = int(metric_chunk_size)
        self.progress = bool(progress)

        if self.metric_chunk_size <= 0:
            raise ValueError("metric_chunk_size must be > 0.")

        self.backbone.to(self.device)
        self.head.to(self.device)

    def _known_mask(self, cls_labels):
        """Return the mask of samples that should contribute to loss/accuracy."""
        if self.classification_ignore_index is None:
            return torch.ones_like(cls_labels, dtype=torch.bool)
        return cls_labels != self.classification_ignore_index

    def _classification_metrics_from_embeddings(self, embeddings, cls_labels):
        """Compute validation loss/accuracy for known classifier labels only."""
        if not torch.is_tensor(embeddings):
            embeddings = torch.as_tensor(embeddings)
        if not torch.is_tensor(cls_labels):
            cls_labels = torch.as_tensor(cls_labels)

        cls_labels = cls_labels.long()
        known_mask = self._known_mask(cls_labels)
        if embeddings.numel() == 0 or not known_mask.any():
            return {
                "loss": float("nan"),
                "acc": float("nan"),
                "num_known": 0,
            }

        known_embeddings = embeddings[known_mask]
        known_labels = cls_labels[known_mask]
        total_loss = 0.0
        total_correct = 0
        total_known = int(known_labels.numel())
        chunk_size = max(1, int(self.metric_chunk_size))

        self.head.eval()
        with torch.no_grad():
            for start in range(0, total_known, chunk_size):
                stop = min(start + chunk_size, total_known)
                emb_chunk = known_embeddings[start:stop].to(self.device, non_blocking=True)
                label_chunk = known_labels[start:stop].to(self.device, non_blocking=True)
                loss, logits = _compute_loss_and_logits(
                    self.head,
                    self.loss,
                    emb_chunk,
                    label_chunk,
                )
                batch_size = int(label_chunk.numel())
                total_loss += float(loss.item()) * batch_size
                total_correct += int((torch.argmax(logits, dim=1) == label_chunk).sum().item())

        return {
            "loss": float(total_loss / total_known),
            "acc": float(total_correct / total_known),
            "num_known": int(total_known),
        }

    def validate_open_set(
        self,
        gallery_loader,
        known_loader,
        unknown_loader,
        mixed_loader,
        evaluation_config=None,
        epoch=None,
        epochs=None,
        return_embeddings=False,
    ):
        """Run three-split open-set validation using train embeddings as gallery.

        Args:
            gallery_loader: Evaluation-transform DataLoader over ``train.csv``.
            known_loader: DataLoader over ``val_known.csv``.
            unknown_loader: DataLoader over ``val_unknown.csv``.
            mixed_loader: DataLoader over mixed ``val.csv``.
            evaluation_config (dict, optional): Evaluation metric configuration.
            epoch (int, optional): Zero-based epoch index.
            epochs (int, optional): Total epoch count.
            return_embeddings (bool): Whether to include mixed embeddings and
                metric labels for downstream visualization.

        Returns:
            dict: Flat metric dictionary containing open-set W&B metric names and
            legacy ``val_*`` aliases used by existing history/checkpoint code.
        """
        self.backbone.eval()
        self.head.eval()

        if epoch is not None and epochs is not None:
            suffix = f" [{epoch + 1}/{epochs}]"
        else:
            suffix = ""

        gallery = extract_embeddings(
            self.backbone,
            gallery_loader,
            device=self.device,
            progress=self.progress,
            desc=f"Val gallery{suffix}",
        )
        known = extract_embeddings(
            self.backbone,
            known_loader,
            device=self.device,
            progress=self.progress,
            desc=f"Val known{suffix}",
        )
        unknown = extract_embeddings(
            self.backbone,
            unknown_loader,
            device=self.device,
            progress=self.progress,
            desc=f"Val unknown{suffix}",
        )
        mixed = extract_embeddings(
            self.backbone,
            mixed_loader,
            device=self.device,
            progress=self.progress,
            desc=f"Val mixed{suffix}",
        )

        known_classification = self._classification_metrics_from_embeddings(
            known["embeddings"],
            known["cls_labels"],
        )
        metric_config = dict(evaluation_config or {})
        metric_config.setdefault("diagnostics_every_n_epochs", 5)
        metric_config.setdefault("diagnostics", {"enabled": True, "max_pairs": 100000})

        open_set_metrics = evaluate_all_validation_splits(
            gallery["embeddings"],
            gallery["metric_labels"],
            known["embeddings"],
            known["metric_labels"],
            unknown["embeddings"],
            unknown["metric_labels"],
            mixed["embeddings"],
            mixed["metric_labels"],
            known_cls_labels=known["cls_labels"],
            unknown_cls_labels=unknown["cls_labels"],
            mixed_cls_labels=mixed["cls_labels"],
            config=metric_config,
            epoch_index=epoch,
            classification_ignore_index=self.classification_ignore_index,
            chunk_size=self.metric_chunk_size,
        )

        result = {
            "loss": float(known_classification["loss"]),
            "acc": float(known_classification["acc"]),
            "top1": float(open_set_metrics.get("known/top1", np.nan)),
            "top5": float(open_set_metrics.get("known/top5", np.nan)),
            "mAP": float(open_set_metrics.get("known/mAP", np.nan)),
            "micro_f1": float(open_set_metrics.get("mixed/F1", np.nan)),
            "num_known": int(known_classification["num_known"]),
            "num_samples": int(mixed["num_samples"]),
        }
        result.update(open_set_metrics)
        if return_embeddings:
            result["embeddings"] = mixed["embeddings"]
            result["metric_labels"] = mixed["metric_labels"]
        return result

    def validate(self, val_loader, epoch=None, epochs=None, return_embeddings=False):
        """Run validation and compute canonical re-ID metrics.

        Args:
            val_loader (torch.utils.data.DataLoader): Loader yielding validation
                batches shaped as ``(images, cls_labels, metric_labels)``.
            epoch (int, optional): Zero-based epoch index for display.
            epochs (int, optional): Total epoch count for display.
            return_embeddings (bool): Whether to include collected CPU embeddings
                and metric labels in the returned dictionary.

        Returns:
            dict: Validation results with keys ``loss``, ``acc``, ``top1``,
            ``top5``, ``mAP``, ``micro_f1``, ``num_known``, and ``num_samples``.
            If ``return_embeddings=True``, also includes ``embeddings`` and
            ``metric_labels``.
        """
        self.backbone.eval()
        self.head.eval()

        total_loss = 0.0
        total_correct = 0
        total_known = 0
        total_samples = 0
        all_embeddings = []
        all_metric_labels = []

        if epoch is not None and epochs is not None:
            desc = f"Val   [{epoch + 1}/{epochs}]"
        else:
            desc = "Val"

        iterator = tqdm(
            val_loader,
            desc=desc,
            leave=False,
            dynamic_ncols=True,
            disable=not self.progress,
        )

        with torch.no_grad():
            for batch in iterator:
                images, cls_labels, metric_labels = _unpack_batch(batch)
                images = images.to(self.device, non_blocking=True)
                cls_labels = cls_labels.to(self.device, non_blocking=True)
                metric_labels = metric_labels.to(self.device, non_blocking=True)

                embeddings = self.backbone(images)
                all_embeddings.append(embeddings.detach().cpu())
                all_metric_labels.append(metric_labels.detach().cpu())
                total_samples += int(images.size(0))

                known_mask = self._known_mask(cls_labels)
                if known_mask.any():
                    known_embeddings = embeddings[known_mask]
                    known_labels = cls_labels[known_mask].long()
                    loss, logits = _compute_loss_and_logits(
                        self.head,
                        self.loss,
                        known_embeddings,
                        known_labels,
                    )
                    batch_known = int(known_labels.size(0))
                    total_loss += float(loss.item()) * batch_known
                    total_known += batch_known
                    preds = torch.argmax(logits, dim=1)
                    total_correct += int((preds == known_labels).sum().item())

                avg_loss = total_loss / total_known if total_known > 0 else float("nan")
                avg_acc = total_correct / total_known if total_known > 0 else float("nan")
                iterator.set_postfix(
                    {
                        "loss": _format_metric(avg_loss),
                        "acc": _format_metric(avg_acc),
                    }
                )

        avg_loss = total_loss / total_known if total_known > 0 else float("nan")
        avg_acc = total_correct / total_known if total_known > 0 else float("nan")
        embeddings = torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty(0)
        metric_labels = (
            torch.cat(all_metric_labels, dim=0)
            if all_metric_labels
            else torch.empty(0, dtype=torch.long)
        )

        if embeddings.numel() == 0:
            reid_metrics = {"top1": np.nan, "top5": np.nan, "mAP": np.nan, "micro_f1": np.nan}
        else:
            reid_metrics = compute_reid_metrics(
                embeddings,
                metric_labels,
                ignore_index=self.metric_ignore_index,
                chunk_size=self.metric_chunk_size,
            )

        result = {
            "loss": float(avg_loss),
            "acc": float(avg_acc),
            "top1": float(reid_metrics["top1"]),
            "top5": float(reid_metrics["top5"]),
            "mAP": float(reid_metrics["mAP"]),
            "micro_f1": float(reid_metrics["micro_f1"]),
            "num_known": int(total_known),
            "num_samples": int(total_samples),
        }
        if return_embeddings:
            result["embeddings"] = embeddings
            result["metric_labels"] = metric_labels
        return result


__all__ = [
    "Validator",
    "extract_embeddings",
    "_compute_loss_and_logits",
    "_format_metric",
    "_head_num_classes",
    "_prediction_logits",
    "_unpack_batch",
    "_validate_labels",
    "_validate_logits",
]
