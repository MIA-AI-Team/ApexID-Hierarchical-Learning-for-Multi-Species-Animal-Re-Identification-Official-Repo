import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DCMIPLoss(nn.Module):
    """
    DCMIP: Discrepant Cluster Proxies + Multi-Instance Proxies
    based on:
        "Discrepant and Multi-Instance Proxies for Unsupervised Person Re-Identification" (ICCV 2023)

    This module implements the core losses and memory banks:
      - DCP memory: [num_classes, num_cluster_proxies, feat_dim]
      - MIP memory: [num_classes, num_instance_proxies, feat_dim]

    Main ideas from the paper:
      1) Multiple discrepant cluster proxies per class reduce intra-class variance.
      2) Multi-instance proxies enable global hard negative mining.
      3) Use cluster-level contrastive loss first, then add instance-level loss later.
    See Sections 3.2, 3.3, and Eq. (8), (10), (11). :contentReference[oaicite:3]{index=3}

    Args:
        feat_dim: Feature embedding dimension.
        num_classes: Number of pseudo-classes / clusters in current epoch.
        num_cluster_proxies: M in the paper. Usually 2.
        num_instance_proxies: K in the paper. Usually same as images-per-identity in batch.
        tau: Temperature. Paper uses 0.05.
        lambda_dcmip: lambda in Eq. (11). Weight for LDCP after instance loss starts.
        start_instance_epoch: Eins in Eq. (11). Paper uses 20 -> start LMIP from epoch 21.
        proxy_momentum: mu in Eq. (2). Paper uses 0.1.
        num_hard_negatives: N in Eq. (10). Paper uses 256.
        cluster_proxy_update_modes:
            list of update policies for the M cluster proxies.
            Each element must be one of {"mean", "rand", "hard"}.
            Example: ["mean", "hard"] or ["mean", "rand"].
        device: torch device
    """

    def __init__(
        self,
        feat_dim,
        num_classes,
        num_cluster_proxies = 2,
        num_instance_proxies = 16,
        tau = 0.05,
        lambda_dcmip = 0.5,
        start_instance_epoch = 20,
        proxy_momentum = 0.1,
        num_hard_negatives = 256,
        cluster_proxy_update_modes = None,
        device = None,
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

        assert len(cluster_proxy_update_modes) == num_cluster_proxies, (
            "cluster_proxy_update_modes length must equal num_cluster_proxies"
        )
        for mode in cluster_proxy_update_modes:
            assert mode in {"mean", "rand", "hard"}, f"Unsupported update mode: {mode}"

        self.cluster_proxy_update_modes = cluster_proxy_update_modes

        # DCP memory: cluster proxies
        self.register_buffer(
            "dcp_memory",
            F.normalize(torch.randn(num_classes, num_cluster_proxies, feat_dim, device=self.device), dim=-1)
        )

        # MIP memory: instance proxies
        self.register_buffer(
            "mip_memory",
            F.normalize(torch.randn(num_classes, num_instance_proxies, feat_dim, device=self.device), dim=-1)
        )

        # Marks whether initialize_memories() has been called for current clustering epoch
        self._initialized = False

    @staticmethod
    def _normalize(x, dim = -1, eps = 1e-12):
        return F.normalize(x, p=2, dim=dim, eps=eps)

    @staticmethod
    def _unique_labels(labels):
        return torch.unique(labels)

    @staticmethod
    def _repeat_or_pad_rows(x, target_rows):
        """
        Ensure x has exactly target_rows rows by:
          - truncating if too many
          - repeating cyclically if too few
        """
        n = x.size(0)
        if n == target_rows:
            return x
        if n > target_rows:
            return x[:target_rows]
        # repeat
        reps = math.ceil(target_rows / n)
        x_rep = x.repeat(reps, 1)
        return x_rep[:target_rows]


    @torch.no_grad()
    def initialize_memories(
        self,
        all_embeddings,
        all_momentum_embeddings,
        all_labels,
    ):
        """
        Initialize DCP and MIP for the current clustering epoch.

        Paper behavior:
          - DCP initialized with cluster centroids
          - MIP initialized with K randomly selected instance features from momentum encoder
        See Section 3.2 and 3.3. :contentReference[oaicite:4]{index=4}

        Args:
            all_embeddings: [N, D] normalized or unnormalized features from main encoder
            all_momentum_embeddings: [N, D] features from momentum encoder
            all_labels: [N] pseudo labels in [0, num_classes-1]
        """
        device = self.dcp_memory.device

        all_embeddings = self._normalize(all_embeddings.to(device, non_blocking=True))
        all_momentum_embeddings = self._normalize(all_momentum_embeddings.to(device, non_blocking=True))
        all_labels = all_labels.to(device, non_blocking=True).long()

        for cls in range(self.num_classes):
            cls_mask = all_labels == cls
            cls_emb = all_embeddings[cls_mask]
            cls_mom_emb = all_momentum_embeddings[cls_mask]

            if cls_emb.numel() == 0:
                # If class absent, keep previous memory or random normalized values
                continue

            # DCP init: all M cluster proxies start from centroid
            centroid = self._normalize(cls_emb.mean(dim=0, keepdim=True)).squeeze(0)
            self.dcp_memory[cls] = centroid.unsqueeze(0).repeat(self.num_cluster_proxies, 1)

            # MIP init: randomly select K momentum features
            if cls_mom_emb.size(0) >= self.num_instance_proxies:
                indices = torch.randperm(cls_mom_emb.size(0), device=device)[:self.num_instance_proxies]
                selected = cls_mom_emb[indices]
            else:
                selected = self._repeat_or_pad_rows(cls_mom_emb, self.num_instance_proxies)

            self.mip_memory[cls] = self._normalize(selected)

        self._initialized = True


    @torch.no_grad()
    def update_dcp_memory(
        self,
        batch_embeddings,
        batch_labels,
    ):
        """
        Update DCP memory using Eq. (2) with different update designs:
          - mean: Eq. (5)
          - rand: Eq. (6)
          - hard: Eq. (7)
        See Section 3.2. :contentReference[oaicite:5]{index=5}
        """
        device = self.dcp_memory.device
        batch_embeddings = self._normalize(batch_embeddings.to(device, non_blocking=True))
        batch_labels = batch_labels.to(device, non_blocking=True).long()

        unique_labels = self._unique_labels(batch_labels)

        for cls in unique_labels.tolist():
            cls_mask = batch_labels == cls
            cls_feats = batch_embeddings[cls_mask]  # [Kc, D]
            if cls_feats.size(0) == 0:
                continue

            q_mean = self._normalize(cls_feats.mean(dim=0, keepdim=True)).squeeze(0)

            for j, mode in enumerate(self.cluster_proxy_update_modes):
                proxy_old = self.dcp_memory[cls, j]  # [D]

                if mode == "mean":
                    update_feat = q_mean

                elif mode == "rand":
                    rand_idx = torch.randint(0, cls_feats.size(0), (1,), device=device).item()
                    update_feat = cls_feats[rand_idx]

                elif mode == "hard":
                    # select sample with lowest similarity to current proxy
                    sims = torch.matmul(cls_feats, proxy_old.unsqueeze(-1)).squeeze(-1)  # [Kc]
                    hard_idx = torch.argmin(sims).item()
                    update_feat = cls_feats[hard_idx]

                else:
                    raise ValueError(f"Unknown update mode: {mode}")

                proxy_new = self.proxy_momentum * proxy_old + (1.0 - self.proxy_momentum) * update_feat
                proxy_new = self._normalize(proxy_new.unsqueeze(0)).squeeze(0)
                self.dcp_memory[cls, j] = proxy_new


    @torch.no_grad()
    def update_mip_memory(
        self,
        batch_momentum_embeddings,
        batch_labels,
    ):
        """
        Update MIP memory by replacement with current batch momentum features.
        See Eq. (9), Section 3.3. :contentReference[oaicite:6]{index=6}
        """
        device = self.mip_memory.device
        batch_momentum_embeddings = self._normalize(batch_momentum_embeddings.to(device, non_blocking=True))
        batch_labels = batch_labels.to(device, non_blocking=True).long()

        unique_labels = self._unique_labels(batch_labels)

        for cls in unique_labels.tolist():
            cls_mask = batch_labels == cls
            cls_feats = batch_momentum_embeddings[cls_mask]

            if cls_feats.size(0) == 0:
                continue

            cls_feats = self._repeat_or_pad_rows(cls_feats, self.num_instance_proxies)
            self.mip_memory[cls] = self._normalize(cls_feats)


    def cluster_contrastive_loss(
        self,
        batch_embeddings,
        batch_labels,
    ):
        """
        LDCP from Eq. (8):
            LDCP = -(1/M) * sum_j log( exp(q·p_j^+ / tau) / sum_i exp(q·p_{i,j}/tau) )

        Where:
          - for each proxy index j
          - compare query to the j-th proxy of every class
        See Eq. (8). :contentReference[oaicite:7]{index=7}
        """
        device = self.dcp_memory.device
        q = self._normalize(batch_embeddings.to(device, non_blocking=True))        # [B, D]
        y = batch_labels.to(device, non_blocking=True).long()                      # [B]

        losses = []

        for j in range(self.num_cluster_proxies):
            proxies_j = self._normalize(self.dcp_memory[:, j, :])   # [C, D]
            logits_j = torch.matmul(q, proxies_j.t()) / self.tau    # [B, C]
            loss_j = F.cross_entropy(logits_j, y, reduction="mean")
            losses.append(loss_j)

        return torch.stack(losses).mean()

    def _select_hard_positive_from_batch(
        self,
        batch_embeddings,
        batch_momentum_embeddings,
        batch_labels,
    ):
        """
        For each query, choose the momentum-encoded positive from the same class in the batch
        with the lowest similarity to the query (hard positive), as described in Section 3.3. :contentReference[oaicite:8]{index=8}

        Returns:
            hard_pos: [B, D]
        """
        device = batch_embeddings.device
        q = self._normalize(batch_embeddings)
        qm = self._normalize(batch_momentum_embeddings)
        y = batch_labels.long()

        B, D = q.shape
        hard_pos = torch.zeros(B, D, device=device, dtype=q.dtype)

        # pairwise similarity between main-encoder query and momentum-encoder positives
        sim_mat = torch.matmul(q, qm.t())  # [B, B]

        for i in range(B):
            same_cls = (y == y[i])
            # Keep same-class entries; usually includes self / same image position if both streams align.
            same_indices = torch.where(same_cls)[0]
            same_sims = sim_mat[i, same_indices]
            hard_idx_local = torch.argmin(same_sims)
            hard_idx_global = same_indices[hard_idx_local]
            hard_pos[i] = qm[hard_idx_global]

        return self._normalize(hard_pos)

    def instance_contrastive_loss(
        self,
        batch_embeddings,
        batch_momentum_embeddings,
        batch_labels,
    ):
        """
        LMIP from Eq. (10):
            LMIP = -log( exp(q·m+ / tau) / (exp(q·m+ / tau) + sum_i exp(q·p_n^i / tau)) )

        Hard positive:
          - lowest-similarity momentum feature from same class in current batch

        Hard negatives:
          - top-N most similar instance proxies among all other classes in MIP memory

        See Eq. (10), Section 3.3. :contentReference[oaicite:9]{index=9}
        """
        device = self.mip_memory.device
        q = self._normalize(batch_embeddings.to(device, non_blocking=True))                    # [B, D]
        qm = self._normalize(batch_momentum_embeddings.to(device, non_blocking=True))          # [B, D]
        y = batch_labels.to(device, non_blocking=True).long()                                  # [B]

        B, D = q.shape
        C, K, _ = self.mip_memory.shape

        # Hard positives from same class in batch
        hard_pos = self._select_hard_positive_from_batch(q, qm, y)          # [B, D]
        pos_logits = torch.sum(q * hard_pos, dim=1, keepdim=True) / self.tau  # [B, 1]

        # All instance proxies from memory
        all_mip = self._normalize(self.mip_memory.view(C * K, D))           # [C*K, D]
        neg_logits_all = torch.matmul(q, all_mip.t()) / self.tau            # [B, C*K]

        # Mask out proxies belonging to the same class
        class_ids = torch.arange(C, device=device).unsqueeze(1).repeat(1, K).view(-1)  # [C*K]
        same_class_mask = class_ids.unsqueeze(0) == y.unsqueeze(1)          # [B, C*K]

        # Set same-class logits to -inf so they are never selected as negatives
        neg_logits_all = neg_logits_all.masked_fill(same_class_mask, float("-inf"))

        # Select top-N hardest negatives globally
        num_available_neg = C * K - K
        topk = min(self.num_hard_negatives, max(num_available_neg, 1))

        hard_neg_logits, _ = torch.topk(neg_logits_all, k=topk, dim=1)      # [B, topk]

        # Build InfoNCE-style loss:
        # numerator: positive
        # denominator: positive + hard negatives
        logits = torch.cat([pos_logits, hard_neg_logits], dim=1)             # [B, 1+topk]
        targets = torch.zeros(B, device=device, dtype=torch.long)            # positive is column 0
        loss = F.cross_entropy(logits, targets, reduction="mean")
        return loss

    def forward(
        self,
        batch_embeddings,
        batch_momentum_embeddings,
        batch_labels,
        epoch,
        update_memories = True,
    ):
        """
        Overall loss Eq. (11):
            if epoch <= Eins:
                L = LDCP
            else:
                L = lambda * LDCP + (1-lambda) * LMIP

        See Eq. (11). :contentReference[oaicite:10]{index=10}
        """
        assert self._initialized, (
            "Memories are not initialized. Call initialize_memories(...) at the start of each clustering epoch."
        )

        ldcp = self.cluster_contrastive_loss(batch_embeddings, batch_labels)

        if epoch <= self.start_instance_epoch:
            loss = ldcp
            lmip = torch.zeros((), device=ldcp.device, dtype=ldcp.dtype)
        else:
            lmip = self.instance_contrastive_loss(
                batch_embeddings=batch_embeddings,
                batch_momentum_embeddings=batch_momentum_embeddings,
                batch_labels=batch_labels,
            )
            loss = self.lambda_dcmip * ldcp + (1.0 - self.lambda_dcmip) * lmip

        if update_memories:
            with torch.no_grad():
                self.update_dcp_memory(batch_embeddings, batch_labels)
                self.update_mip_memory(batch_momentum_embeddings, batch_labels)

        stats = {
            "loss"(loss.detach().item()),
            "ldcp"(ldcp.detach().item()),
            "lmip"(lmip.detach().item()),
        }
        return loss, stats
