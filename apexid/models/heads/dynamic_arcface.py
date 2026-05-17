import math

import torch
from torch import nn
from torch.nn import functional as F


class DynArcFaceHead(nn.Module):
    """Dynamic ArcFace margin head.

    Forward returns margin-adjusted logits. Use with an external classification
    criterion such as ``nn.CrossEntropyLoss``.
    """

    def __init__(self, in_features, out_features, s=30.0, k1=1.1, k2=0.8, k3=0.332):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def _compute_dynamic_margins(self):
        weight = F.normalize(self.weight, p=2, dim=1)
        cos_sim = torch.mm(weight, weight.t())
        cos_sim.fill_diagonal_(-2.0)
        max_cos, _ = cos_sim.max(dim=1)
        phi_nn = torch.acos(torch.clamp(max_cos, -1.0 + 1e-7, 1.0 - 1e-7))

        diff = phi_nn - self.k1
        phi_smooth = (0.03 / (torch.pow(torch.abs(diff) / 0.05, 1.1) + 1.0)) * self.k3
        return F.relu(diff) * self.k2 + self.k3 + phi_smooth

    def predict_logits(self, embeddings):
        cosine = F.linear(F.normalize(embeddings, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        return cosine * self.s

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        sine = torch.sqrt(1.0 - torch.clamp(cosine * cosine, 0, 1))

        margins = self._compute_dynamic_margins()
        labels = labels.long()
        m = margins[labels]

        target_cos = cosine[torch.arange(len(labels), device=labels.device), labels]
        target_sin = sine[torch.arange(len(labels), device=labels.device), labels]
        phi_target = target_cos * torch.cos(m) - target_sin * torch.sin(m)

        th = torch.cos(math.pi - m)
        mm = torch.sin(math.pi - m) * m
        phi_target = torch.where(target_cos > th, phi_target, target_cos - mm)

        logits = cosine.clone()
        logits[torch.arange(len(labels), device=labels.device), labels] = phi_target
        return logits * self.s

