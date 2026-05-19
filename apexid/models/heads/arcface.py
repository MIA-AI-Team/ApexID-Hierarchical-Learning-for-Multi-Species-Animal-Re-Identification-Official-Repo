import math
import torch
from torch import nn
from torch.nn import functional as F


class ArcFaceHead(nn.Module):
    """
    ArcFace additive angular margin head.

    This module returns margin-adjusted logits for training with
    nn.CrossEntropyLoss.

    During training:
        target logit     = s * cos(theta_y + m)
        non-target logit = s * cos(theta_j)

    During inference:
        use predict_logits() or directly use normalized embeddings
        for cosine retrieval.
    """

    def __init__(self, in_features, out_features, s=64.0, m=0.5):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        # Shape: (num_classes, embedding_dim)
        # Each row is one class center.
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # Monotonicity correction threshold.
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def predict_logits(self, embeddings):
        """
        Inference logits without angular margin.

        Args:
            embeddings:
                Tensor of shape (B, D)

        Returns:
            Cosine logits of shape (B, C), scaled by s.
        """

        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        cosine = F.linear(embeddings, weight)

        return cosine * self.s

    def forward(self, embeddings, labels):
        """
        Training forward pass.

        Args:
            embeddings:
                Tensor of shape (B, D)

            labels:
                Tensor of shape (B,), containing class indices.

        Returns:
            Margin-adjusted logits of shape (B, C).
        """

        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        cosine = F.linear(embeddings, weight)

        # sin(theta) = sqrt(1 - cos^2(theta))
        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-7))

        # cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # Stable monotonicity correction.
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        logits = one_hot * phi + (1.0 - one_hot) * cosine

        return logits * self.s