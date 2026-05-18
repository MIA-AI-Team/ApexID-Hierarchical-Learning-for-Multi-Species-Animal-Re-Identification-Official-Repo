import torch
from torch.nn import functional as F

from .sphereface2 import SphereFace2Head


class SphereFace2FocalHead(SphereFace2Head):
    """SphereFace2 head with focal weighting added to the base loss."""

    def __init__(
        self,
        in_features,
        out_features,
        r=40.0,
        m=0.4,
        t=3,
        lam=0.7,
        focal_gamma=2.0,
        alpha=None,
        magn_type=None,
        lw=None,
    ):
        if magn_type not in {None, "C"}:
            raise ValueError(
                "SphereFace2FocalHead reuses SphereFace2Head and supports only "
                "magn_type='C'."
            )
        if alpha is not None:
            lam = alpha
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            r=r,
            m=m,
            t=t,
            lam=lam,
        )
        self.focal_gamma = float(focal_gamma)
        self.magn_type = "C"
        self.alpha = self.lam
        self.lw = lw

    def forward(self, embeddings, labels):
        """Compute the scalar SphereFace2 loss with focal weighting."""
        if self.focal_gamma == 0.0:
            return super().forward(embeddings, labels)

        cosine = F.linear(
            F.normalize(embeddings),
            F.normalize(self.weight),
        )
        g_cos = self._g(cosine)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        pos_logits = -self.r * (g_cos - self.m) - self.bias
        pos_loss = (self.lam / self.r) * F.softplus(pos_logits)
        pos_focal = torch.sigmoid(pos_logits).pow(self.focal_gamma)
        pos_loss = (one_hot * pos_loss * pos_focal).sum(dim=1)

        neg_logits = self.r * (g_cos + self.m) + self.bias
        neg_loss = ((1.0 - self.lam) / self.r) * F.softplus(neg_logits)
        neg_focal = torch.sigmoid(neg_logits).pow(self.focal_gamma)
        neg_loss = ((1.0 - one_hot) * neg_loss * neg_focal).sum(dim=1)

        return (pos_loss + neg_loss).mean()


__all__ = ["SphereFace2FocalHead"]
