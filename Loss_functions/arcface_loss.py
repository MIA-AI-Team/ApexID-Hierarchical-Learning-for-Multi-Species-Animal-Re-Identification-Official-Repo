import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss (Deng et al., CVPR 2019 / TPAMI 2022).

    L = -log( exp(s * cos(theta_yi + m)) /
              (exp(s * cos(theta_yi + m)) + sum_{j!=yi} exp(s * cos(theta_j))) )

    The classifier weights W are L2-normalised internally, so this module
    acts as both the final linear layer and the loss.

    Args:
        in_features:  Embedding dimension (e.g. 512).
        out_features: Number of classes / identities.
        s:            Feature re-scale (radius of hypersphere). Default 30.0.
        m:            Additive angular margin in radians.   Default 0.5.
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # sin(pi - m) for numerical safety when theta + m > pi
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels):
        # cos(theta) via normalised dot product
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        # sin(theta) from cos(theta)
        sine = torch.sqrt(1.0 - torch.clamp(cosine * cosine, 0, 1))
        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m
        # when theta + m >= pi, fall back to cosine - mm for monotonicity
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.s

        return logits
