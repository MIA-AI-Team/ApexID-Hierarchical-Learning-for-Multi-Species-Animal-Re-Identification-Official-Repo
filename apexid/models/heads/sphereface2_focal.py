import math

import torch
from torch import nn
from torch.nn import functional as F


class SphereFace2FocalHead(nn.Module):
    """SphereFace2 head with focal weighting.

    Forward returns the scalar loss. ``predict_logits`` returns class scores for
    accuracy/evaluation.
    """

    def __init__(
        self,
        in_features,
        out_features,
        magn_type="C",
        alpha=0.7,
        r=40.0,
        m=0.4,
        t=3.0,
        lw=50.0,
        focal_gamma=2.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.magn_type = magn_type
        self.alpha = alpha
        self.r = r
        self.m = m
        self.t = t
        self.lw = lw
        self.focal_gamma = focal_gamma

        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_normal_(self.weight)

        z = alpha / ((1.0 - alpha) * max(out_features - 1, 1))
        if magn_type == "C":
            ay = r * (2.0 * 0.5**t - 1.0 - m)
            ai = r * (2.0 * 0.5**t - 1.0 + m)
        elif magn_type == "A":
            theta_y = min(math.pi, math.pi / 2.0 + m)
            ay = r * (2.0 * ((math.cos(theta_y) + 1.0) / 2.0) ** t - 1.0)
            ai = r * (2.0 * 0.5**t - 1.0)
        elif magn_type == "M":
            theta_y = min(math.pi, m * math.pi / 2.0)
            ay = r * (2.0 * ((math.cos(theta_y) + 1.0) / 2.0) ** t - 1.0)
            ai = r * (2.0 * 0.5**t - 1.0)
        else:
            raise ValueError(f"Unknown SphereFace2 focal magn_type: {magn_type}")

        temp = (1.0 - z) ** 2 + 4.0 * z * math.exp(ay - ai)
        b = math.log(2.0 * z) - ai - math.log(1.0 - z + math.sqrt(temp))
        self.bias = nn.Parameter(torch.empty(1))
        nn.init.constant_(self.bias, b)

    def _cosine(self, embeddings):
        weight = F.normalize(self.weight, p=2, dim=0)
        return F.normalize(embeddings, p=2, dim=1).mm(weight)

    def predict_logits(self, embeddings):
        return self.r * self._cosine(embeddings) + self.bias

    def forward(self, embeddings, labels):
        cos_theta = self._cosine(embeddings)
        labels = labels.long()

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        with torch.no_grad():
            if self.magn_type == "C":
                g_cos_theta = 2.0 * ((cos_theta + 1.0) / 2.0).pow(self.t) - 1.0
                g_cos_theta = g_cos_theta - self.m * (2.0 * one_hot - 1.0)
            elif self.magn_type == "A":
                theta_m = torch.acos(cos_theta.clamp(-1.0 + 1e-5, 1.0 - 1e-5))
                theta_m[torch.arange(labels.size(0), device=labels.device), labels] += self.m
                theta_m.clamp_(1e-5, math.pi)
                g_cos_theta = 2.0 * ((torch.cos(theta_m) + 1.0) / 2.0).pow(self.t) - 1.0
            elif self.magn_type == "M":
                theta_m = torch.acos(cos_theta.clamp(-1.0 + 1e-5, 1.0 - 1e-5))
                theta_m[torch.arange(labels.size(0), device=labels.device), labels] *= self.m
                theta_m.clamp_(1e-5, math.pi)
                g_cos_theta = 2.0 * ((torch.cos(theta_m) + 1.0) / 2.0).pow(self.t) - 1.0
            else:
                raise ValueError(f"Unknown SphereFace2 focal magn_type: {self.magn_type}")

            d_theta = g_cos_theta - cos_theta

        logits = self.r * (cos_theta + d_theta) + self.bias
        weight = self.alpha * one_hot + (1.0 - self.alpha) * (1.0 - one_hot)
        weight = self.lw * self.out_features / self.r * weight

        probs = torch.sigmoid(logits)
        pt = one_hot * probs + (1.0 - one_hot) * (1.0 - probs)
        focal_weight = (1.0 - pt).pow(self.focal_gamma)
        return F.binary_cross_entropy_with_logits(
            logits,
            one_hot,
            weight=(weight * focal_weight).detach(),
        )
