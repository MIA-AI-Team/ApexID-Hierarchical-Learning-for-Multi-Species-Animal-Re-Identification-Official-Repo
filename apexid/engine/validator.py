import math

import numpy as np
import torch
from tqdm.auto import tqdm

from apexid.evaluation import compute_reid_metrics


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
    head_output = head(embeddings, labels)

    if torch.is_tensor(head_output) and head_output.ndim == 0:
        return head_output, _prediction_logits(head, embeddings)

    if criterion is None:
        raise ValueError("Head returned class logits but no external loss was provided.")

    return criterion(head_output, labels), head_output


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
        self.progress = bool(progress)

        self.backbone.to(self.device)
        self.head.to(self.device)

    def _known_mask(self, cls_labels):
        """Return the mask of samples that should contribute to loss/accuracy."""
        if self.classification_ignore_index is None:
            return torch.ones_like(cls_labels, dtype=torch.bool)
        return cls_labels != self.classification_ignore_index

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
    "_compute_loss_and_logits",
    "_format_metric",
    "_prediction_logits",
    "_unpack_batch",
]
