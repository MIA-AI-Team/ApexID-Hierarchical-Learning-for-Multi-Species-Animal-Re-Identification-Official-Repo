import math

import torch
from torch import nn
from torch.nn import functional as F


class ArcMarginProductSubcenter(nn.Module):
    """Sub-center cosine projection for ArcFace-style heads."""

    def __init__(self, in_features, out_features, k=3):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.k = int(k)
        if self.in_features <= 0:
            raise ValueError("in_features must be a positive integer.")
        if self.out_features <= 1:
            raise ValueError("out_features must be greater than 1.")
        if self.k <= 0:
            raise ValueError("k must be a positive integer.")

        self.weight = nn.Parameter(torch.empty(self.out_features * self.k, self.in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features):
        cosine_all = F.linear(F.normalize(features, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine


class ArcFaceLossAdaptiveMargin(nn.Module):
    """Apply ArcFace transform with per-class margins."""

    def __init__(self, margins, out_dim, s=30.0):
        super().__init__()
        self.out_dim = int(out_dim)
        self.s = float(s)
        if self.out_dim <= 1:
            raise ValueError("out_dim must be greater than 1.")
        if self.s <= 0:
            raise ValueError("s must be greater than zero.")

        margins_tensor = torch.as_tensor(margins, dtype=torch.float32)
        if margins_tensor.ndim != 1 or margins_tensor.numel() != self.out_dim:
            raise ValueError("margins must be a 1D tensor/list of length out_dim.")
        if torch.any(margins_tensor <= 0) or torch.any(margins_tensor >= math.pi / 2):
            raise ValueError("Each margin must be in the open interval (0, pi/2).")
        self.register_buffer("margins", margins_tensor)

    def forward(self, logits, labels):
        labels = labels.long()
        margins = self.margins[labels]

        cos_m = torch.cos(margins).view(-1, 1)
        sin_m = torch.sin(margins).view(-1, 1)
        th = torch.cos(math.pi - margins).view(-1, 1)
        mm = (torch.sin(math.pi - margins) * margins).view(-1, 1)

        cosine = logits.clamp(-1.0, 1.0)
        sine = torch.sqrt(torch.clamp(1.0 - cosine * cosine, min=1e-7))
        phi = cosine * cos_m - sine * sin_m
        phi = torch.where(cosine > th, phi, cosine - mm)

        one_hot = F.one_hot(labels, self.out_dim).float()
        output = one_hot * phi + (1.0 - one_hot) * cosine
        return output * self.s


class ArcFaceSubCenterDynamicHead(nn.Module):
    """
    MiewID-style Sub-center ArcFace head with per-class adaptive margins.

    The adaptive part is the per-class margin vector (``margins``).
    If explicit ``margins`` are not provided, margins are initialized uniformly.
    """

    def __init__(
        self,
        in_features,
        out_features,
        s=30.0,
        k=3,
        margins=None,
        margins_source="uniform",
        margin_default=0.3,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.s = float(s)
        self.k = int(k)
        self.margins_source = str(margins_source).lower()
        self.margin_default = float(margin_default)

        if self.in_features <= 0:
            raise ValueError("in_features must be a positive integer.")
        if self.out_features <= 1:
            raise ValueError("out_features must be greater than 1.")
        if self.s <= 0:
            raise ValueError("s must be greater than zero.")
        if self.k <= 0:
            raise ValueError("k must be a positive integer.")
        if not 0 < self.margin_default < math.pi / 2:
            raise ValueError("margin_default must be in the open interval (0, pi/2).")

        margins_tensor = self._resolve_margins(margins)
        self.subcenter = ArcMarginProductSubcenter(self.in_features, self.out_features, k=self.k)
        self.arcface_margin = ArcFaceLossAdaptiveMargin(margins=margins_tensor, out_dim=self.out_features, s=self.s)

    def _resolve_margins(self, margins):
        if margins is not None:
            if hasattr(margins, "items"):
                ordered = [0.0] * self.out_features
                for key, value in dict(margins.items()).items():
                    idx = int(key)
                    if idx < 0 or idx >= self.out_features:
                        raise ValueError("Margin dict keys must be class indices within [0, out_features).")
                    ordered[idx] = float(value)
                margins_tensor = torch.tensor(ordered, dtype=torch.float32)
            else:
                margins_tensor = torch.as_tensor(margins, dtype=torch.float32)
        elif self.margins_source == "uniform":
            margins_tensor = torch.full((self.out_features,), self.margin_default, dtype=torch.float32)
        else:
            raise ValueError(
                f"Unsupported margins_source '{self.margins_source}'. "
                "Use explicit margins or margins_source='uniform'."
            )

        if margins_tensor.ndim != 1 or margins_tensor.numel() != self.out_features:
            raise ValueError("margins must be a 1D list/tensor with length equal to out_features.")
        return margins_tensor

    def predict_logits(self, embeddings):
        cosine = self.subcenter(embeddings.float())
        return cosine * self.s

    def forward(self, embeddings, labels):
        cosine = self.subcenter(embeddings.float())
        return self.arcface_margin(cosine, labels)


__all__ = [
    "ArcMarginProductSubcenter",
    "ArcFaceLossAdaptiveMargin",
    "ArcFaceSubCenterDynamicHead",
]
