import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score, roc_curve

try:
    from Callback.EarlyStopping import EarlyStopping
    from Callback.Checkpoint import save_checkpoint, save_backbone_weights
    from Callback.Plotting import TensorBoardLogger
except ImportError:
    from ..Callback.EarlyStopping import EarlyStopping
    from ..Callback.Checkpoint import save_checkpoint, save_backbone_weights
    from ..Callback.Plotting import TensorBoardLogger


torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class Trainer:
    def __init__(
        self,
        backbone,
        head,
        loss=None,
        optimizer=None,
        scheduler=None,
        early_stopping_patience=7,
        early_stopping_delta=0.0,
        early_stopping_path="best_state_dict.pt",
        checkpoint_dir="checkpoints",
        tensorboard_log_dir=None,
        classification_ignore_index=-100,
        device=None,
        metrics_base_seed=42,
    ):
        """Initialize the training helper and its optional utilities.

        Args:
            backbone (torch.nn.Module): Feature extractor that produces embeddings.
            head (torch.nn.Module): Classification or metric-learning head operating
                on the embeddings.
            loss (callable, optional): External loss function used when the head
                returns logits instead of a scalar loss.
            optimizer (torch.optim.Optimizer, optional): Optimizer used for model
                updates. If omitted, Adam is created automatically.
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning
                rate scheduler stepped during training.
            early_stopping_patience (int, optional): Number of non-improving
                validation epochs tolerated before stopping.
            early_stopping_delta (float, optional): Minimum validation-loss
                improvement required to reset early stopping.
            early_stopping_path (str, optional): Path used by the early stopping
                checkpoint saver.
            checkpoint_dir (str, optional): Directory used to store training
                checkpoints.
            tensorboard_log_dir (str, optional): Directory enabling TensorBoard
                logging when provided.
            classification_ignore_index (int, optional): Label value ignored during
                classification metrics and validation classification loss.
            device (str or torch.device, optional): Device where the model is run.
            metrics_base_seed (int, optional): Base seed stored for metric-related
                reproducibility settings.

        Returns:
            None
        """
        self.backbone = backbone
        self.head = head
        self.loss = loss
        self.classification_ignore_index = classification_ignore_index
        self.metrics_base_seed = int(metrics_base_seed)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = nn.ModuleDict({
            "backbone": self.backbone,
            "head": self.head
        }).to(self.device)

        if optimizer is None:
            adam_kwargs = {"lr": 1e-3}
            if "cuda" in str(self.device).lower():
                adam_kwargs["fused"] = True
            self.optimizer = torch.optim.Adam(self.model.parameters(), **adam_kwargs)
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler

        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            verbose=True,
            delta=early_stopping_delta,
            path=early_stopping_path
        )

        self.checkpoint_dir = checkpoint_dir
        self.tb_logger = TensorBoardLogger(tensorboard_log_dir) if tensorboard_log_dir else None

    def _head_name(self):
        """Return the class name of the active head module.

        Args:
            None

        Returns:
            str: Name of the head module class.
        """
        return type(self.head).__name__

    def _format_metric(self, x):
        """Format a metric value for progress-bar display.

        Args:
            x (Any): Metric value to format.

        Returns:
            str: Four-decimal string for finite numbers, otherwise ``"nan"``.
        """
        if x is None:
            return "nan"
        try:
            x = float(x)
        except Exception:
            return "nan"
        if np.isnan(x) or np.isinf(x):
            return "nan"
        return f"{x:.4f}"

    def _unpack_batch(self, batch):
        """
        Unpack a training or validation batch into images and labels.

        Args:
            batch (tuple or list): Batch shaped as ``(images, cls_labels)`` or
                ``(images, cls_labels, metric_labels)``.

        Returns:
            tuple: Three-item tuple containing images, classification labels, and
            metric labels.
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

    def _prediction_logits(self, embeddings):
        """Obtain prediction logits from a head exposing ``predict_logits``.

        Args:
            embeddings (torch.Tensor): Embedding tensor passed to the head.

        Returns:
            torch.Tensor: Predicted class logits for the given embeddings.
        """
        if hasattr(self.head, "predict_logits"):
            return self.head.predict_logits(embeddings)
        raise ValueError(
            "Head returned a scalar loss but does not expose predict_logits(...)."
        )

    def _compute_loss_and_logits(self, embeddings, labels):
        """
        Compute the training loss and logits for the configured head type.

        Args:
            embeddings (torch.Tensor): Embedding tensor produced by the backbone.
            labels (torch.Tensor): Class labels corresponding to the embeddings.

        Returns:
            tuple: Two-item tuple containing the scalar loss tensor and the logits
            used for predictions.
        """
        head_name = self._head_name()

        if head_name in {"ArcFaceLoss", "DynArcFaceLoss"}:
            logits = self.head(embeddings, labels)
            if self.loss is None:
                raise ValueError(
                    f"{head_name} returns logits, so you must provide an external loss "
                    "(usually nn.CrossEntropyLoss())."
                )
            loss = self.loss(logits, labels)
            return loss, logits

        elif head_name == "SphereFace2Loss":
            loss = self.head(embeddings, labels)
            logits = self._prediction_logits(embeddings)
            return loss, logits

        else:
            # generic fallback if we added new head
            head_output = self.head(embeddings, labels)

            if torch.is_tensor(head_output) and head_output.ndim == 0:
                loss = head_output
                logits = self._prediction_logits(embeddings)
                return loss, logits

            if self.loss is None:
                raise ValueError(
                    "Head returned class logits but no external loss was provided."
                )

            loss = self.loss(head_output, labels)
            return loss, head_output

    def _compute_retrieval_metrics(self, embeddings, metric_labels):
        """
        Compute leave-one-out retrieval metrics on the validation embeddings.

        Args:
            embeddings (torch.Tensor): Embedding tensor collected from validation.
            metric_labels (torch.Tensor): Identity labels used for retrieval
                evaluation.

        Returns:
            dict: Retrieval metric dictionary containing ``top1``, ``top5``, and
            ``mAP`` scores.
        """
        if embeddings is None or metric_labels is None:
            return {"top1": np.nan, "top5": np.nan, "mAP": np.nan}

        if len(embeddings) == 0:
            return {"top1": np.nan, "top5": np.nan, "mAP": np.nan}

        emb = F.normalize(embeddings.float(), dim=1).cpu().numpy()
        labels = metric_labels.cpu().numpy()

        if self.classification_ignore_index is not None:
            valid_mask = labels != self.classification_ignore_index
            emb = emb[valid_mask]
            labels = labels[valid_mask]

        n = len(labels)
        if n < 2:
            return {"top1": np.nan, "top5": np.nan, "mAP": np.nan}

        sim = emb @ emb.T
        np.fill_diagonal(sim, -np.inf)
        ranked = np.argsort(-sim, axis=1)

        top1_hits = []
        top5_hits = []
        ap_list = []

        for i in range(n):
            order = ranked[i]
            rel = (labels[order] == labels[i]).astype(np.int32)

            n_pos = int(rel.sum())
            if n_pos == 0:
                continue

            top1_hits.append(float(rel[0]))
            top5_hits.append(float(rel[: min(5, len(rel))].any()))

            pos_ranks = np.flatnonzero(rel)
            precisions = []
            for hit_num, rank_idx in enumerate(pos_ranks, start=1):
                precisions.append(hit_num / float(rank_idx + 1))
            ap = float(np.mean(precisions)) if len(precisions) > 0 else np.nan
            ap_list.append(ap)

        if len(ap_list) == 0:
            return {"top1": np.nan, "top5": np.nan, "mAP": np.nan}

        return {
            "top1": float(np.mean(top1_hits)),
            "top5": float(np.mean(top5_hits)),
            "mAP": float(np.mean(ap_list)),
        }

    def _compute_verification_metrics(self, embeddings, metric_labels):
        """
        Compute pairwise verification metrics on the validation embeddings.

        Args:
            embeddings (torch.Tensor): Embedding tensor collected from validation.
            metric_labels (torch.Tensor): Identity labels used for pairwise
                verification.

        Returns:
            dict: Verification metric dictionary containing ``roc_auc``, ``eer``,
            and ``tar_at_far_1e3``.
        """
        if embeddings is None or metric_labels is None:
            return {"roc_auc": np.nan, "eer": np.nan, "tar_at_far_1e3": np.nan}

        if len(embeddings) == 0:
            return {"roc_auc": np.nan, "eer": np.nan, "tar_at_far_1e3": np.nan}

        emb = F.normalize(embeddings.float(), dim=1).cpu().numpy()
        labels = metric_labels.cpu().numpy()

        if self.classification_ignore_index is not None:
            valid_mask = labels != self.classification_ignore_index
            emb = emb[valid_mask]
            labels = labels[valid_mask]

        n = len(labels)
        if n < 2:
            return {"roc_auc": np.nan, "eer": np.nan, "tar_at_far_1e3": np.nan}

        sim = emb @ emb.T
        iu = np.triu_indices(n, k=1)
        scores = sim[iu]
        targets = (labels[iu[0]] == labels[iu[1]]).astype(np.int32)

        n_pos = int(targets.sum())
        n_neg = int(len(targets) - n_pos)
        if n_pos == 0 or n_neg == 0:
            return {"roc_auc": np.nan, "eer": np.nan, "tar_at_far_1e3": np.nan}

        roc_auc = float(roc_auc_score(targets, scores))

        fpr, tpr, _ = roc_curve(targets, scores, drop_intermediate=False)
        fnr = 1.0 - tpr

        idx_eer = int(np.nanargmin(np.abs(fpr - fnr)))
        eer = float((fpr[idx_eer] + fnr[idx_eer]) / 2.0)

        far_target = 1e-3
        valid = np.where(fpr <= far_target)[0]
        if len(valid) == 0:
            tar_at_far = 0.0
        else:
            tar_at_far = float(np.max(tpr[valid]))

        return {
            "roc_auc": roc_auc,
            "eer": eer,
            "tar_at_far_1e3": tar_at_far,
        }

    def _compute_open_set_metrics(self, embeddings, metric_labels):
        """Combine retrieval and verification metrics into one dictionary.

        Args:
            embeddings (torch.Tensor): Embedding tensor collected from validation.
            metric_labels (torch.Tensor): Identity labels used for open-set
                evaluation.

        Returns:
            dict: Dictionary merging retrieval and verification metrics.
        """
        retrieval = self._compute_retrieval_metrics(embeddings, metric_labels)
        verification = self._compute_verification_metrics(embeddings, metric_labels)
        return {**retrieval, **verification}

    def train_one_epoch(self, train_loader, epoch, epochs):
        """Run one full training epoch.

        Args:
            train_loader (torch.utils.data.DataLoader): Loader providing training
                batches.
            epoch (int): Zero-based epoch index.
            epochs (int): Total number of training epochs.

        Returns:
            tuple: Two-item tuple containing average training loss and average
            training accuracy for the epoch.
        """
        self.backbone.train()
        self.head.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(
            train_loader,
            desc=f"Train [{epoch+1}/{epochs}]",
            leave=False,
            dynamic_ncols=True
        )

        for batch in pbar:
            images, labels, _ = self._unpack_batch(batch)

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if self.classification_ignore_index is not None:
                if (labels == self.classification_ignore_index).any():
                    raise ValueError(
                        "Unknown/ignored labels appeared in the training loader. "
                        "Training must contain only known IDs."
                    )

            self.optimizer.zero_grad()

            embeddings = self.backbone(images)
            loss, logits = self._compute_loss_and_logits(embeddings, labels)

            loss.backward()
            self.optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()

            avg_loss = total_loss / max(total_samples, 1)
            avg_acc = total_correct / max(total_samples, 1)
            current_lr = self.optimizer.param_groups[0]["lr"]

            pbar.set_postfix({
                "loss": self._format_metric(avg_loss),
                "acc": self._format_metric(avg_acc),
                "lr": f"{current_lr:.2e}"
            })

        avg_loss = total_loss / max(total_samples, 1)
        avg_acc = total_correct / max(total_samples, 1)

        return avg_loss, avg_acc

    def validate_one_epoch(self, val_loader, epoch, epochs, collect_embeddings=False):
        """Run one validation epoch and compute open-set evaluation metrics.

        Args:
            val_loader (torch.utils.data.DataLoader): Loader providing validation
                batches.
            epoch (int): Zero-based epoch index.
            epochs (int): Total number of training epochs.
            collect_embeddings (bool, optional): Whether to include validation
                embeddings and metric labels in the returned tuple.

        Returns:
            tuple: Validation loss, validation accuracy, metric dictionary, and
            optionally the collected embeddings and metric labels.
        """
        self.backbone.eval()
        self.head.eval()

        total_loss = 0.0
        total_correct = 0
        total_known = 0

        all_embeddings = []
        all_metric_labels = []

        pbar = tqdm(
            val_loader,
            desc=f"Val   [{epoch+1}/{epochs}]",
            leave=False,
            dynamic_ncols=True
        )

        with torch.no_grad():
            for batch in pbar:
                images, cls_labels, metric_labels = self._unpack_batch(batch)

                images = images.to(self.device, non_blocking=True)
                cls_labels = cls_labels.to(self.device, non_blocking=True)
                metric_labels = metric_labels.to(self.device, non_blocking=True)

                embeddings = self.backbone(images)

                # collect ALL val embeddings for retrieval / verification metrics
                all_embeddings.append(embeddings.detach().cpu())
                all_metric_labels.append(metric_labels.detach().cpu())

                if self.classification_ignore_index is None:
                    known_mask = torch.ones_like(cls_labels, dtype=torch.bool)
                else:
                    known_mask = cls_labels != self.classification_ignore_index

                # only known samples go through the classification head for val_loss / val_acc
                if known_mask.any():
                    known_embeddings = embeddings[known_mask]
                    known_labels = cls_labels[known_mask]

                    loss, logits = self._compute_loss_and_logits(known_embeddings, known_labels)

                    batch_known = known_labels.size(0)
                    total_loss += loss.item() * batch_known
                    total_known += batch_known

                    preds = torch.argmax(logits, dim=1)
                    total_correct += (preds == known_labels).sum().item()

                avg_loss = total_loss / max(total_known, 1)
                avg_acc = total_correct / max(total_known, 1)

                pbar.set_postfix({
                    "loss": self._format_metric(avg_loss),
                    "acc": self._format_metric(avg_acc),
                })

        avg_loss = total_loss / max(total_known, 1)
        avg_acc = total_correct / max(total_known, 1)

        all_embeddings = torch.cat(all_embeddings, dim=0) if len(all_embeddings) > 0 else torch.empty(0)
        all_metric_labels = torch.cat(all_metric_labels, dim=0) if len(all_metric_labels) > 0 else torch.empty(0, dtype=torch.long)

        metric_dict = self._compute_open_set_metrics(all_embeddings, all_metric_labels)

        if collect_embeddings:
            return avg_loss, avg_acc, metric_dict, all_embeddings, all_metric_labels
        return avg_loss, avg_acc, metric_dict

    def train(self, train_loader, val_loader, epochs, log_embeddings_every=5,
              start_epoch=0, resume_history=None):
        """Train the model for multiple epochs and record history.

        Args:
            train_loader (torch.utils.data.DataLoader): Loader providing training
                batches.
            val_loader (torch.utils.data.DataLoader): Loader providing validation
                batches.
            epochs (int): Total number of epochs to run.
            log_embeddings_every (int, optional): Frequency for logging validation
                embeddings to TensorBoard.
            start_epoch (int, optional): Epoch index to start from when resuming.
            resume_history (dict, optional): Existing history dictionary to extend.

        Returns:
            dict: History dictionary containing loss, accuracy, and open-set metric
            series accumulated during training.
        """
        history = resume_history if resume_history is not None else {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_top1": [],
            "val_top5": [],
            "val_mAP": [],
            "val_roc_auc": [],
            "val_eer": [],
            "val_tar_at_far_1e3": [],
        }

        best_val_loss = float("inf")
        if resume_history and len(resume_history.get("val_loss", [])) > 0:
            best_val_loss = min(resume_history["val_loss"])
            self.early_stopping.best_score = -best_val_loss
            self.early_stopping.val_loss_min = best_val_loss

        if self.tb_logger:
            sample_batch = next(iter(train_loader))
            sample_input, _, _ = self._unpack_batch(sample_batch)
            sample_input = sample_input[:1].to(self.device)
            self.tb_logger.log_graph(self.backbone, sample_input)

        epoch_bar = tqdm(range(start_epoch, epochs), desc="Epochs", dynamic_ncols=True)

        for epoch in epoch_bar:
            train_loss, train_acc = self.train_one_epoch(train_loader, epoch, epochs)

            should_log_emb = self.tb_logger and (epoch % log_embeddings_every == 0 or epoch == epochs - 1)
            val_result = self.validate_one_epoch(
                val_loader,
                epoch,
                epochs,
                collect_embeddings=should_log_emb
            )

            if should_log_emb:
                val_loss, val_acc, val_metrics, val_embeddings, val_metric_labels = val_result
                self.tb_logger.log_embeddings(val_embeddings, val_metric_labels, epoch)
            else:
                val_loss, val_acc, val_metrics = val_result

            history["train_loss"].append(float(train_loss))
            history["train_acc"].append(float(train_acc))
            history["val_loss"].append(float(val_loss))
            history["val_acc"].append(float(val_acc))
            history["val_top1"].append(float(val_metrics["top1"]) if not np.isnan(val_metrics["top1"]) else np.nan)
            history["val_top5"].append(float(val_metrics["top5"]) if not np.isnan(val_metrics["top5"]) else np.nan)
            history["val_mAP"].append(float(val_metrics["mAP"]) if not np.isnan(val_metrics["mAP"]) else np.nan)
            history["val_roc_auc"].append(float(val_metrics["roc_auc"]) if not np.isnan(val_metrics["roc_auc"]) else np.nan)
            history["val_eer"].append(float(val_metrics["eer"]) if not np.isnan(val_metrics["eer"]) else np.nan)
            history["val_tar_at_far_1e3"].append(
                float(val_metrics["tar_at_far_1e3"]) if not np.isnan(val_metrics["tar_at_far_1e3"]) else np.nan
            )

            epoch_bar.set_postfix({
                "tr_loss": self._format_metric(train_loss),
                "tr_acc": self._format_metric(train_acc),
                "va_loss": self._format_metric(val_loss),
                "va_acc": self._format_metric(val_acc),
                "top1": self._format_metric(val_metrics["top1"]),
                "mAP": self._format_metric(val_metrics["mAP"]),
                "auc": self._format_metric(val_metrics["roc_auc"]),
                "eer": self._format_metric(val_metrics["eer"]),
            })

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            is_best = self.early_stopping(val_loss, self.model)

            checkpoint_state = {
                "epoch": epoch,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                "history": history,
                "val_metrics": val_metrics,
            }

            save_checkpoint(
                checkpoint_state,
                is_best=is_best,
                save_dir=self.checkpoint_dir,
                best_name="best.pt",
            )

            if is_best:
                save_backbone_weights(self.backbone, self.checkpoint_dir)

            if self.tb_logger:
                self.tb_logger.log_epoch(
                    epoch,
                    train_loss,
                    val_loss,
                    train_acc,
                    val_acc,
                    lr=self.optimizer.param_groups[0]["lr"],
                )

                # optional extra scalar logging if your TensorBoardLogger exposes .writer
                if hasattr(self.tb_logger, "writer") and self.tb_logger.writer is not None:
                    for k, v in val_metrics.items():
                        if v is not None and not np.isnan(v):
                            self.tb_logger.writer.add_scalar(f"val/{k}", v, epoch)

                self.tb_logger.log_weight_histograms(self.model, epoch)

            best_val_loss = min(best_val_loss, val_loss)

            if self.early_stopping.early_stop:
                print("\nEarly stopping triggered.")
                break

        if self.tb_logger:
            final_idx = len(history["train_loss"]) - 1
            self.tb_logger.log_hparams(
                hparam_dict={
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "optimizer": type(self.optimizer).__name__,
                    "early_stopping_patience": self.early_stopping.patience,
                },
                metric_dict={
                    "hparam/best_val_loss": best_val_loss,
                    "hparam/final_train_loss": history["train_loss"][final_idx],
                    "hparam/final_val_acc": history["val_acc"][final_idx],
                    "hparam/final_val_top1": history["val_top1"][final_idx],
                    "hparam/final_val_mAP": history["val_mAP"][final_idx],
                    "hparam/final_val_roc_auc": history["val_roc_auc"][final_idx],
                    "hparam/final_val_eer": history["val_eer"][final_idx],
                    "hparam/final_val_tar_at_far_1e3": history["val_tar_at_far_1e3"][final_idx],
                },
            )
            self.tb_logger.close()

        return history
