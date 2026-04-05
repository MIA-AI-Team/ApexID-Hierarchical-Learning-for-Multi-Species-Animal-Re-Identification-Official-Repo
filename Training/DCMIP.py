import copy
import torch
import torch.nn.functional as F

try:
    from Loss_functions.DCMIPLoss import DCMIPLoss
except ImportError:  # Supports nested package imports if the repo is packaged later.
    from ..Loss_functions.DCMIPLoss import DCMIPLoss


@torch.no_grad()
def update_momentum_encoder(model, momentum_model, alpha = 0.999):
    """
    Momentum encoder update:
        theta_m <- alpha * theta_m + (1 - alpha) * theta
    This follows Eq. (3) from the paper. :contentReference[oaicite:11]{index=11}
    """
    for p, pm in zip(model.parameters(), momentum_model.parameters()):
        pm.data.mul_(alpha).add_(p.data, alpha=(1.0 - alpha))



class DCMIPTrainer:
    """
    Minimal helper showing how DCMIP fits into training.

    This is NOT a full pipeline.
    You still need:
      1) feature extraction for the whole dataset
      2) clustering (DBSCAN as in the paper / our CW implementation)
      3) building a pseudo-labeled dataloader
    """

    def __init__(
        self,
        encoder,
        feat_dim,
        num_classes,
        optimizer,
        device,
        num_cluster_proxies = 2,
        num_instance_proxies = 16,
        tau = 0.05,
        lambda_dcmip = 0.5,
        start_instance_epoch = 20,
        proxy_momentum = 0.1,
        encoder_momentum = 0.999,
        num_hard_negatives = 256,
        cluster_proxy_update_modes = None,
    ):
        self.encoder = encoder.to(device, non_blocking=True)
        self.momentum_encoder = copy.deepcopy(encoder).to(device, non_blocking=True)
        for p in self.momentum_encoder.parameters():
            p.requires_grad = False

        self.loss_fn = DCMIPLoss(
            feat_dim=feat_dim,
            num_classes=num_classes,
            num_cluster_proxies=num_cluster_proxies,
            num_instance_proxies=num_instance_proxies,
            tau=tau,
            lambda_dcmip=lambda_dcmip,
            start_instance_epoch=start_instance_epoch,
            proxy_momentum=proxy_momentum,
            num_hard_negatives=num_hard_negatives,
            cluster_proxy_update_modes=cluster_proxy_update_modes,
            device=device,
        )

        self.optimizer = optimizer
        self.device = device
        self.encoder_momentum = encoder_momentum

    @torch.no_grad()
    def extract_features(self, dataloader):
        """
        Extract full-dataset features from both encoders.
        Expects each batch to return:
            images, indices, pseudo_labels
        or similar structure you can adapt.
        """
        self.encoder.eval()
        self.momentum_encoder.eval()

        feats = []
        mom_feats = []
        labels = []

        for batch in dataloader:
            images, pseudo_labels = batch[:2]
            images = images.to(self.device, non_blocking=True)
            pseudo_labels = pseudo_labels.to(self.device, non_blocking=True)

            f = self.encoder(images)
            fm = self.momentum_encoder(images)

            feats.append(F.normalize(f, dim=1).cpu())
            mom_feats.append(F.normalize(fm, dim=1).cpu())
            labels.append(pseudo_labels.cpu())

        feats = torch.cat(feats, dim=0)
        mom_feats = torch.cat(mom_feats, dim=0)
        labels = torch.cat(labels, dim=0)
        return feats, mom_feats, labels

    @torch.no_grad()
    def initialize_epoch_memories(self, full_epoch_loader):
        """
        Call this once after pseudo labels for the epoch are ready.
        """
        feats, mom_feats, labels = self.extract_features(full_epoch_loader)
        self.loss_fn.initialize_memories(feats, mom_feats, labels)

    def train_one_epoch(self, train_loader, epoch):
        """
        Expects train_loader batches like:
            images, pseudo_labels
        """
        self.encoder.train()
        self.momentum_encoder.eval()

        running = {"loss": 0.0, "ldcp": 0.0, "lmip": 0.0, "count": 0}

        for batch in train_loader:
            images, pseudo_labels = batch[:2]
            images = images.to(self.device, non_blocking=True)
            pseudo_labels = pseudo_labels.to(self.device, non_blocking=True).long()

            emb = self.encoder(images)

            with torch.no_grad():
                mom_emb = self.momentum_encoder(images)

            loss, stats = self.loss_fn(
                batch_embeddings=emb,
                batch_momentum_embeddings=mom_emb,
                batch_labels=pseudo_labels,
                epoch=epoch,
                update_memories=True,
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                update_momentum_encoder(
                    model=self.encoder,
                    momentum_model=self.momentum_encoder,
                    alpha=self.encoder_momentum,
                )

            bs = images.size(0)
            running["loss"] += stats["loss"] * bs
            running["ldcp"] += stats["ldcp"] * bs
            running["lmip"] += stats["lmip"] * bs
            running["count"] += bs

        for k in ["loss", "ldcp", "lmip"]:
            running[k] /= max(running["count"], 1)

        return running
