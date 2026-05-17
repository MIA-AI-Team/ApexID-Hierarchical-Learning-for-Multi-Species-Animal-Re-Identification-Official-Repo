import math

import torch
from torch import nn
from torch.nn import functional as F

from apexid.utils.seed import DEFAULT_SEED, seed_everything


seed_everything(DEFAULT_SEED)


class DCMIPLoss(nn.Module):
    """Discrepant Cluster Proxies + Multi-Instance Proxies objective."""

    def __init__(
        self,
        feat_dim,
        num_classes,
        num_cluster_proxies=2,
        num_instance_proxies=16,
        tau=0.05,
        lambda_dcmip=0.5,
        start_instance_epoch=20,
        proxy_momentum=0.1,
        num_hard_negatives=256,
        cluster_proxy_update_modes=None,
        device=None,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.num_cluster_proxies = num_cluster_proxies
        self.num_instance_proxies = num_instance_proxies
        self.tau = tau
        self.lambda_dcmip = lambda_dcmip
        self.start_instance_epoch = start_instance_epoch
        self.proxy_momentum = proxy_momentum
        self.num_hard_negatives = num_hard_negatives
        self.device = device if device is not None else torch.device("cpu")

        if cluster_proxy_update_modes is None:
            cluster_proxy_update_modes = ["mean", "hard"]
        if len(cluster_proxy_update_modes) != num_cluster_proxies:
            raise ValueError("cluster_proxy_update_modes length must equal num_cluster_proxies")
        for mode in cluster_proxy_update_modes:
            if mode not in {"mean", "rand", "hard"}:
                raise ValueError(f"Unsupported update mode: {mode}")
        self.cluster_proxy_update_modes = cluster_proxy_update_modes

        self.register_buffer(
            "dcp_memory",
            F.normalize(
                torch.randn(num_classes, num_cluster_proxies, feat_dim, device=self.device),
                p=2,
                dim=-1,
            ),
        )
        self.register_buffer(
            "mip_memory",
            F.normalize(
                torch.randn(num_classes, num_instance_proxies, feat_dim, device=self.device),
                p=2,
                dim=-1,
            ),
        )
        self._initialized = False

    @staticmethod
    def _normalize(x, dim=-1, eps=1e-12):
        return F.normalize(x, p=2, dim=dim, eps=eps)

    @staticmethod
    def _repeat_or_pad_rows(x, target_rows):
        n = x.size(0)
        if n == target_rows:
            return x
        if n > target_rows:
            return x[:target_rows]
        reps = math.ceil(target_rows / n)
        return x.repeat(reps, 1)[:target_rows]

    @torch.no_grad()
    def initialize_memories(self, all_embeddings, all_momentum_embeddings, all_labels):
        device = self.dcp_memory.device
        all_embeddings = self._normalize(all_embeddings.to(device, non_blocking=True))
        all_momentum_embeddings = self._normalize(all_momentum_embeddings.to(device, non_blocking=True))
        all_labels = all_labels.to(device, non_blocking=True).long()

        for cls in range(self.num_classes):
            cls_mask = all_labels == cls
            cls_emb = all_embeddings[cls_mask]
            cls_mom_emb = all_momentum_embeddings[cls_mask]
            if cls_emb.numel() == 0:
                continue

            centroid = self._normalize(cls_emb.mean(dim=0, keepdim=True)).squeeze(0)
            self.dcp_memory[cls] = centroid.unsqueeze(0).repeat(self.num_cluster_proxies, 1)

            if cls_mom_emb.size(0) >= self.num_instance_proxies:
                indices = torch.randperm(cls_mom_emb.size(0), device=device)[:self.num_instance_proxies]
                selected = cls_mom_emb[indices]
            else:
                selected = self._repeat_or_pad_rows(cls_mom_emb, self.num_instance_proxies)
            self.mip_memory[cls] = self._normalize(selected)

        self._initialized = True

    @torch.no_grad()
    def update_dcp_memory(self, batch_embeddings, batch_labels):
        device = self.dcp_memory.device
        batch_embeddings = self._normalize(batch_embeddings.to(device, non_blocking=True))
        batch_labels = batch_labels.to(device, non_blocking=True).long()

        for cls in torch.unique(batch_labels).tolist():
            cls_feats = batch_embeddings[batch_labels == cls]
            if cls_feats.size(0) == 0:
                continue

            q_mean = self._normalize(cls_feats.mean(dim=0, keepdim=True)).squeeze(0)
            for j, mode in enumerate(self.cluster_proxy_update_modes):
                proxy_old = self.dcp_memory[cls, j]
                if mode == "mean":
                    update_feat = q_mean
                elif mode == "rand":
                    rand_idx = torch.randint(0, cls_feats.size(0), (1,), device=device).item()
                    update_feat = cls_feats[rand_idx]
                elif mode == "hard":
                    sims = torch.matmul(cls_feats, proxy_old.unsqueeze(-1)).squeeze(-1)
                    update_feat = cls_feats[torch.argmin(sims).item()]
                else:
                    raise ValueError(f"Unknown update mode: {mode}")

                proxy_new = self.proxy_momentum * proxy_old + (1.0 - self.proxy_momentum) * update_feat
                self.dcp_memory[cls, j] = self._normalize(proxy_new.unsqueeze(0)).squeeze(0)

    @torch.no_grad()
    def update_mip_memory(self, batch_momentum_embeddings, batch_labels):
        device = self.mip_memory.device
        batch_momentum_embeddings = self._normalize(batch_momentum_embeddings.to(device, non_blocking=True))
        batch_labels = batch_labels.to(device, non_blocking=True).long()

        for cls in torch.unique(batch_labels).tolist():
            cls_feats = batch_momentum_embeddings[batch_labels == cls]
            if cls_feats.size(0) == 0:
                continue
            cls_feats = self._repeat_or_pad_rows(cls_feats, self.num_instance_proxies)
            self.mip_memory[cls] = self._normalize(cls_feats)

    def cluster_contrastive_loss(self, batch_embeddings, batch_labels):
        device = self.dcp_memory.device
        q = self._normalize(batch_embeddings.to(device, non_blocking=True))
        y = batch_labels.to(device, non_blocking=True).long()

        losses = []
        for j in range(self.num_cluster_proxies):
            proxies_j = self._normalize(self.dcp_memory[:, j, :])
            logits_j = torch.matmul(q, proxies_j.t()) / self.tau
            losses.append(F.cross_entropy(logits_j, y, reduction="mean"))
        return torch.stack(losses).mean()

    def _select_hard_positive_from_batch(self, batch_embeddings, batch_momentum_embeddings, batch_labels):
        device = batch_embeddings.device
        q = self._normalize(batch_embeddings)
        qm = self._normalize(batch_momentum_embeddings)
        y = batch_labels.long()

        hard_pos = torch.zeros_like(q)
        sim_mat = torch.matmul(q, qm.t())
        for i in range(q.size(0)):
            same_indices = torch.where(y == y[i])[0]
            same_sims = sim_mat[i, same_indices]
            hard_pos[i] = qm[same_indices[torch.argmin(same_sims)]]
        return self._normalize(hard_pos)

    def instance_contrastive_loss(self, batch_embeddings, batch_momentum_embeddings, batch_labels):
        device = self.mip_memory.device
        q = self._normalize(batch_embeddings.to(device, non_blocking=True))
        qm = self._normalize(batch_momentum_embeddings.to(device, non_blocking=True))
        y = batch_labels.to(device, non_blocking=True).long()

        batch_size, feat_dim = q.shape
        num_classes, num_instances, _ = self.mip_memory.shape

        hard_pos = self._select_hard_positive_from_batch(q, qm, y)
        pos_logits = torch.sum(q * hard_pos, dim=1, keepdim=True) / self.tau

        all_mip = self._normalize(self.mip_memory.view(num_classes * num_instances, feat_dim))
        neg_logits_all = torch.matmul(q, all_mip.t()) / self.tau

        class_ids = torch.arange(num_classes, device=device).unsqueeze(1).repeat(1, num_instances).view(-1)
        same_class_mask = class_ids.unsqueeze(0) == y.unsqueeze(1)
        neg_logits_all = neg_logits_all.masked_fill(same_class_mask, float("-inf"))

        num_available_neg = num_classes * num_instances - num_instances
        topk = min(self.num_hard_negatives, max(num_available_neg, 1))
        hard_neg_logits, _ = torch.topk(neg_logits_all, k=topk, dim=1)

        logits = torch.cat([pos_logits, hard_neg_logits], dim=1)
        targets = torch.zeros(batch_size, device=device, dtype=torch.long)
        return F.cross_entropy(logits, targets, reduction="mean")

    def forward(
        self,
        batch_embeddings,
        batch_momentum_embeddings,
        batch_labels,
        epoch,
        update_memories=True,
    ):
        if not self._initialized:
            raise RuntimeError(
                "Memories are not initialized. Call initialize_memories(...) at the start of each clustering epoch."
            )

        ldcp = self.cluster_contrastive_loss(batch_embeddings, batch_labels)
        if epoch <= self.start_instance_epoch:
            loss = ldcp
            lmip = torch.zeros((), device=ldcp.device, dtype=ldcp.dtype)
        else:
            lmip = self.instance_contrastive_loss(batch_embeddings, batch_momentum_embeddings, batch_labels)
            loss = self.lambda_dcmip * ldcp + (1.0 - self.lambda_dcmip) * lmip

        if update_memories:
            with torch.no_grad():
                self.update_dcp_memory(batch_embeddings, batch_labels)
                self.update_mip_memory(batch_momentum_embeddings, batch_labels)

        stats = {
            "loss": loss.detach().item(),
            "ldcp": ldcp.detach().item(),
            "lmip": lmip.detach().item(),
        }
        return loss, stats
