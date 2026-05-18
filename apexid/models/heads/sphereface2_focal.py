import torch
from torch.nn import functional as F

from .sphereface2_head import SphereFace2Head


class SphereFace2FocalHead(SphereFace2Head):
    """SphereFace2 head with focal weighting on the official BCE objective."""

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
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            magn_type=magn_type,
            alpha=alpha,
            r=r,
            m=m,
            t=t,
            lw=lw,
        )
        self.focal_gamma = float(focal_gamma)
        if self.focal_gamma < 0.0:
            raise ValueError("focal_gamma must be non-negative.")

    def _logits_and_targets(self, embeddings, labels):
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=0)

        labels = labels.view(-1, 1).long()
        cos_theta = F.normalize(embeddings, dim=1).mm(self.w)

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels, 1.0)

        with torch.no_grad():
            if self.magn_type == "C":
                g_cos_theta = 2.0 * ((cos_theta + 1.0) / 2.0).pow(self.t) - 1.0
                g_cos_theta = g_cos_theta - self.m * (2.0 * one_hot - 1.0)

            elif self.magn_type == "A":
                theta_m = torch.acos(cos_theta.clamp(-1.0 + 1e-5, 1.0 - 1e-5))
                theta_m.scatter_(1, labels, theta_m.gather(1, labels) + self.m)
                theta_m.clamp_(1e-5, 3.14159)
                g_cos_theta = torch.cos(theta_m)
                g_cos_theta = 2.0 * ((g_cos_theta + 1.0) / 2.0).pow(self.t) - 1.0

            elif self.magn_type == "M":
                m_theta = torch.acos(cos_theta.clamp(-1.0 + 1e-5, 1.0 - 1e-5))
                m_theta.scatter_(1, labels, m_theta.gather(1, labels) * self.m)
                m_theta.clamp_(1e-5, 3.14159)
                g_cos_theta = torch.cos(m_theta)
                g_cos_theta = 2.0 * ((g_cos_theta + 1.0) / 2.0).pow(self.t) - 1.0

            else:
                raise NotImplementedError

            d_theta = g_cos_theta - cos_theta

        logits = self.r * (cos_theta + d_theta) + self.b
        return logits, one_hot

    def forward(self, embeddings, labels):
        """Compute the scalar SphereFace2 focal loss."""
        if self.focal_gamma == 0.0:
            return super().forward(embeddings, labels)

        logits, one_hot = self._logits_and_targets(embeddings, labels)
        weight = self.alpha * one_hot + (1.0 - self.alpha) * (1.0 - one_hot)
        weight = self.lw * self.num_class / self.r * weight

        loss = F.binary_cross_entropy_with_logits(
            logits,
            one_hot,
            weight=weight,
            reduction="none",
        )
        probabilities = torch.sigmoid(logits)
        target_probabilities = probabilities * one_hot + (1.0 - probabilities) * (1.0 - one_hot)
        focal_weight = (1.0 - target_probabilities).pow(self.focal_gamma)
        return (loss * focal_weight).mean()


__all__ = ["SphereFace2FocalHead"]
