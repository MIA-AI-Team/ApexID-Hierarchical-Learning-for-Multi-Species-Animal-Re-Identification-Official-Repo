import math

import torch
from torch import nn
from torch.nn import functional as F


class ArcFaceHead(nn.Module):
    """ArcFace margin head.

    Forward returns margin-adjusted logits. Use with an external classification
    criterion such as ``nn.CrossEntropyLoss``.
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def predict_logits(self, embeddings):
        cosine = F.linear(F.normalize(embeddings, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        return cosine * self.s

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        sine = torch.sqrt(1.0 - torch.clamp(cosine * cosine, 0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return logits * self.s

