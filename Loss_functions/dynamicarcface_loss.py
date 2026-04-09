import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynArcFaceLoss(nn.Module):
    """
    Dyn-arcFace: Dynamic Additive Angular Margin Loss (Jiao et al., MTA 2021).

    Extends ArcFace by replacing the fixed angular margin m with a per-class
    dynamic margin phi(W_n, W_yi) that adapts based on the angular distance
    between a class centre and its nearest neighbour.

    phi(W_n, W_yi) = relu(phi_{yi-n} - k1) * k2 + k3 + phi_smooth

    phi_smooth = (0.03 / (abs(phi_{yi-n} - k1) / 0.05)^1.1 + 1) * k3

    where phi_{yi-n} is the angle between W_yi and its nearest-neighbour W_n.

    Args:
        in_features:  Embedding dimension.
        out_features: Number of classes / identities.
        s:            Feature re-scale. Default 30.0.
        k1, k2, k3:  Dyn-arcFace hyperparameters. Defaults from the paper's
                      best configuration: k1=1.1, k2=0.8, k3=0.332.
    """

    def __init__(self, in_features, out_features, s=30.0, k1=1.1, k2=0.8, k3=0.332):
        """Initialize the dynamic ArcFace classification head.

        Args:
            in_features (int): Embedding dimension of the input features.
            out_features (int): Number of target classes.
            s (float, optional): Logit scaling factor applied after margin updates.
            k1 (float, optional): Dynamic margin threshold hyperparameter.
            k2 (float, optional): Dynamic margin scaling hyperparameter.
            k3 (float, optional): Dynamic margin offset hyperparameter.

        Returns:
            None
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def _compute_dynamic_margins(self):
        """Compute the current per-class dynamic angular margins.

        Args:
            None

        Returns:
            torch.Tensor: One dynamic margin value for each output class.
        """
        W_norm = F.normalize(self.weight, dim=1)
        # cosine similarity between all class-centre pairs
        cos_sim = torch.mm(W_norm, W_norm.t())
        # mask self-similarity so it doesn't count as nearest neighbour
        cos_sim.fill_diagonal_(-2.0)
        # nearest neighbour = highest cosine = smallest angle
        max_cos, _ = cos_sim.max(dim=1)
        phi_nn = torch.acos(torch.clamp(max_cos, -1.0 + 1e-7, 1.0 - 1e-7))

        diff = phi_nn - self.k1
        phi_smooth = (0.03 / (torch.pow(torch.abs(diff) / 0.05, 1.1) + 1.0)) * self.k3
        margins = F.relu(diff) * self.k2 + self.k3 + phi_smooth
        return margins

    def forward(self, embeddings, labels):
        """Compute dynamic ArcFace logits for a batch of embeddings.

        Args:
            embeddings (torch.Tensor): Input embedding tensor.
            labels (torch.Tensor): Target class indices for each embedding.

        Returns:
            torch.Tensor: Margin-adjusted logits scaled by ``self.s``.
        """
        cosine = F.linear(embeddings, F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.clamp(cosine * cosine, 0, 1))

        # per-class dynamic margins  -> shape (out_features,)
        margins = self._compute_dynamic_margins()
        # select the margin for each sample's target class
        m = margins[labels]  # (batch,)

        cos_m = torch.cos(m)
        sin_m = torch.sin(m)

        # cos(theta + m_yi) for the target class logit of each sample
        target_cos = cosine[torch.arange(len(labels)), labels]
        target_sin = sine[torch.arange(len(labels)), labels]
        phi_target = target_cos * cos_m - target_sin * sin_m

        # monotonicity fallback (same idea as standard ArcFace)
        th = torch.cos(math.pi - m)
        mm = torch.sin(math.pi - m) * m
        phi_target = torch.where(target_cos > th, phi_target, target_cos - mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        logits = cosine.clone()
        logits[torch.arange(len(labels)), labels] = phi_target
        logits *= self.s

        return logits
