import torch
from torch.nn import functional as F

from ...losses.sphereface2 import SphereFace2


class SphereFace2Head(SphereFace2):
    """Repo-compatible adapter for the official SphereFace2 implementation."""

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
    ):
        in_features = int(in_features)
        out_features = int(out_features)
        if in_features <= 0:
            raise ValueError("in_features must be a positive integer.")
        if out_features <= 1:
            raise ValueError("out_features must be greater than 1 for SphereFace2.")

        magn_type = str(magn_type).upper()
        if magn_type not in {"C", "A", "M"}:
            raise ValueError("magn_type must be one of 'C', 'A', or 'M'.")

        alpha = float(alpha)
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in the open interval (0, 1).")

        super().__init__(
            feat_dim=in_features,
            num_class=out_features,
            magn_type=magn_type,
            alpha=alpha,
            r=float(r),
            m=float(m),
            t=float(t),
            lw=float(lw),
        )
        self.in_features = in_features
        self.out_features = out_features

    def predict_logits(self, embeddings):
        """Return non-margin class logits for accuracy/validation metrics."""
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=0)
        cosine = F.normalize(embeddings, dim=1).mm(self.w)
        return self.r * cosine + self.b


__all__ = ["SphereFace2Head"]
