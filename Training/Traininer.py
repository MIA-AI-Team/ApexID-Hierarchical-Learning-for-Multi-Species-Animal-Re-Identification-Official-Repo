import torch
import torch.nn as nn
from tqdm.auto import tqdm

try:
    from Callback.EarlyStopping import EarlyStopping
    from Callback.Checkpoint import save_checkpoint, save_backbone_weights
    from Callback.Plotting import TensorBoardLogger
except ImportError:  
    from ..Callback.EarlyStopping import EarlyStopping
    from ..Callback.Checkpoint import save_checkpoint, save_backbone_weights
    from ..Callback.Plotting import TensorBoardLogger

# Available loss functions (use as the `head` argument):
#   from Loss_functions import ArcFaceLoss, SphereFace2Loss, DynArcFaceLoss
#
# ArcFace / DynArcFace return logits -> pair with nn.CrossEntropyLoss():
#   head = ArcFaceLoss(embedding_dim, num_classes, s=30.0, m=0.5)
#   head = DynArcFaceLoss(embedding_dim, num_classes, s=30.0, k1=1.1, k2=0.8, k3=0.332)
#   loss = nn.CrossEntropyLoss()
#
# SphereFace2 returns a scalar loss directly (no CrossEntropyLoss needed):
#   head = SphereFace2Loss(embedding_dim, num_classes, r=40.0, m=0.4, t=3, lam=0.7)
#   loss = None  # forward already computes the loss

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

class Trainer:
    def __init__(
        self,
        backbone,
        head,
        loss,
        optimizer=None,
        scheduler=None,
        early_stopping_patience=7,
        early_stopping_delta=0.0,
        early_stopping_path="best_state_dict.pt",
        checkpoint_dir="checkpoints",
        tensorboard_log_dir=None,
    ):
        self.backbone = backbone
        self.head = head
        self.loss = loss

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = nn.ModuleDict({
            "backbone": self.backbone,
            "head": self.head
        }).to(self.device, non_blocking=True)

        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, fused=True)
        else:
            self.optimizer = optimizer

        if scheduler is None:
            self.scheduler = None
        else:
            self.scheduler = scheduler

        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            verbose=True,
            delta=early_stopping_delta,
            path=early_stopping_path
        )

        self.checkpoint_dir = checkpoint_dir
        self.tb_logger = TensorBoardLogger(tensorboard_log_dir) if tensorboard_log_dir else None

    def _prediction_logits(self, embeddings):
        if hasattr(self.head, "predict_logits"):
            return self.head.predict_logits(embeddings)
        raise ValueError(
            "Head returned a scalar loss but does not expose predict_logits(...) "
            "for accuracy computation."
        )

    def _compute_loss_and_logits(self, embeddings, labels):
        head_output = self.head(embeddings, labels)

        if torch.is_tensor(head_output) and head_output.ndim == 0:
            loss = head_output
            logits = self._prediction_logits(embeddings)
            return loss, logits

        if self.loss is None:
            raise ValueError(
                "Trainer received class logits from the head but no loss function was provided."
            )

        loss = self.loss(head_output, labels)
        return loss, head_output

    def train_one_epoch(self, train_loader, epoch, epochs):
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

        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

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

            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_samples
            current_lr = self.optimizer.param_groups[0]["lr"]

            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "acc": f"{avg_acc:.4f}",
                "lr": f"{current_lr:.2e}"
            })

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        return avg_loss, avg_acc

    def validate_one_epoch(self, val_loader, epoch, epochs, collect_embeddings=False):
        self.backbone.eval()
        self.head.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_embeddings = []
        all_labels = []

        pbar = tqdm(
            val_loader,
            desc=f"Val   [{epoch+1}/{epochs}]",
            leave=False,
            dynamic_ncols=True
        )

        with torch.no_grad():
            for images, labels in pbar:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                embeddings = self.backbone(images)
                loss, logits = self._compute_loss_and_logits(embeddings, labels)

                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                preds = torch.argmax(logits, dim=1)
                total_correct += (preds == labels).sum().item()

                if collect_embeddings:
                    all_embeddings.append(embeddings.cpu())
                    all_labels.append(labels.cpu())

                avg_loss = total_loss / total_samples
                avg_acc = total_correct / total_samples

                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "acc": f"{avg_acc:.4f}"
                })

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        if collect_embeddings:
            return avg_loss, avg_acc, torch.cat(all_embeddings), torch.cat(all_labels)
        return avg_loss, avg_acc

    def train(self, train_loader, val_loader, epochs, log_embeddings_every=5):
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        best_val_loss = float("inf")

        if self.tb_logger:
            sample_input = next(iter(train_loader))[0][:1].to(self.device)
            self.tb_logger.log_graph(self.backbone, sample_input)

        epoch_bar = tqdm(range(epochs), desc="Epochs", dynamic_ncols=True)

        for epoch in epoch_bar:
            train_loss, train_acc = self.train_one_epoch(train_loader, epoch, epochs)

            should_log_emb = self.tb_logger and (epoch % log_embeddings_every == 0 or epoch == epochs - 1)
            val_result = self.validate_one_epoch(val_loader, epoch, epochs, collect_embeddings=should_log_emb)

            if should_log_emb:
                val_loss, val_acc, val_embeddings, val_labels = val_result
                self.tb_logger.log_embeddings(val_embeddings, val_labels, epoch)
            else:
                val_loss, val_acc = val_result

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            epoch_bar.set_postfix({
                "train_loss": f"{train_loss:.4f}",
                "train_acc": f"{train_acc:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "val_acc": f"{val_acc:.4f}",
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
                "best_loss": min(best_val_loss, val_loss),
            }

            if self.scheduler is not None:
                checkpoint_state["lr_scheduler"] = self.scheduler.state_dict()

            save_checkpoint(
                state=checkpoint_state,
                is_best=is_best,
                save_dir=self.checkpoint_dir,
                last_name="last.pt",
                best_name="best.pt",
            )

            if is_best:
                save_backbone_weights(self.backbone, self.checkpoint_dir)

            if self.tb_logger:
                self.tb_logger.log_epoch(
                    epoch, train_loss, val_loss, train_acc, val_acc,
                    lr=self.optimizer.param_groups[0]["lr"],
                )
                self.tb_logger.log_weight_histograms(self.model, epoch)

            best_val_loss = min(best_val_loss, val_loss)

            if self.early_stopping.early_stop:
                print("\nEarly stopping triggered.")
                break

        if self.tb_logger:
            self.tb_logger.log_hparams(
                hparam_dict={
                    'lr': self.optimizer.param_groups[0]["lr"],
                    'optimizer': type(self.optimizer).__name__,
                    'early_stopping_patience': self.early_stopping.patience,
                },
                metric_dict={
                    'hparam/best_val_loss': best_val_loss,
                    'hparam/final_train_loss': history["train_loss"][-1],
                    'hparam/final_val_acc': history["val_acc"][-1],
                },
            )
            self.tb_logger.close()

        return history
