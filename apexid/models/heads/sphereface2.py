import math

import torch
from torch import nn
from torch.nn import functional as F


class SphereFace2Head(nn.Module):
    """SphereFace2 trainable head.

    Forward returns the scalar SphereFace2 loss. ``predict_logits`` returns
    class scores for accuracy/evaluation.
    """

    def __init__(self, in_features, out_features, r=40.0, m=0.4, t=3, lam=0.7):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.m = m
        self.t = t
        self.lam = lam

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.bias = nn.Parameter(torch.zeros(1))
        self._init_bias()

    def _init_bias(self):
        a_y = self.r * (self._g_scalar(0.0) - self.m)
        a_i = self.r * (self._g_scalar(0.0) + self.m)
        z = self.lam / ((1.0 - self.lam) * max(self.out_features - 1, 1))
        inside = (1.0 - z) ** 2 + 4.0 * z * math.exp(a_y - a_i)
        b0 = math.log(2.0 * z) - a_i - math.log((1.0 - z) + math.sqrt(inside))
        self.bias.data.fill_(b0)

    def _g_scalar(self, cos_val):
        return 2.0 * ((cos_val + 1.0) / 2.0) ** self.t - 1.0

    def _g(self, cos_theta):
        return 2.0 * torch.pow((cos_theta + 1.0) / 2.0, self.t) - 1.0

    def predict_logits(self, embeddings):
        return F.linear(
            F.normalize(embeddings),
            F.normalize(self.weight),
        )

    def forward(self, embeddings, labels):
        cosine = F.linear(
            F.normalize(embeddings),
            F.normalize(self.weight),
        )
        g_cos = self._g(cosine)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        pos_logits = -self.r * (g_cos - self.m) - self.bias
        pos_loss = (self.lam / self.r) * F.softplus(pos_logits)
        pos_loss = (one_hot * pos_loss).sum(dim=1)

        neg_logits = self.r * (g_cos + self.m) + self.bias
        neg_loss = ((1.0 - self.lam) / self.r) * F.softplus(neg_logits)
        neg_loss = ((1.0 - one_hot) * neg_loss).sum(dim=1)

        return (pos_loss + neg_loss).mean()

