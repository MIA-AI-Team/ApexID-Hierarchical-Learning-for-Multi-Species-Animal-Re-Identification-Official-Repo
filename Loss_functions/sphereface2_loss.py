import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SphereFace2Loss(nn.Module):
    """
    SphereFace2: Binary Classification is All You Need (Wen & Liu, ICLR 2022).

    Constructs K independent binary classification tasks on the hypersphere.
    Final loss (Eq. 5 from the paper):

        L = (lam/r) * log(1 + exp(-r*(g(cos(theta_y)) - m) - b))
          + ((1-lam)/r) * sum_{i!=y} log(1 + exp(r*(g(cos(theta_i)) + m) + b))

    where g(z) = 2 * ((z+1)/2)^t - 1  is the similarity adjustment.

    Args:
        in_features:  Embedding dimension.
        out_features: Number of classes / identities.
        r:            Radius / hard-sample mining factor. Default 40.0.
        m:            Additive angular margin (used for both pos & neg). Default 0.4.
        t:            Similarity adjustment power. Default 3.
        lam:          Positive/negative balance weight. Default 0.7.
    """

    def __init__(self, in_features, out_features, r=40.0, m=0.4, t=3, lam=0.7):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.m = m
        self.t = t
        self.lam = lam

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # learnable bias (shared across all classes, as per the paper)
        self.bias = nn.Parameter(torch.zeros(1))
        self._init_bias()

    def _init_bias(self):
        """Bias initialisation from Appendix G (Eq. 10)."""
        a_y = self.r * (self._g_scalar(0.0) - self.m)
        a_i = self.r * (self._g_scalar(0.0) + self.m)
        n = self.out_features
        z = self.lam / ((1 - self.lam) * max(n - 1, 1))
        inside = (1 - z) ** 2 + 4 * z * math.exp(a_y - a_i)
        b0 = math.log(2 * z) - a_i - math.log((1 - z) + math.sqrt(inside))
        self.bias.data.fill_(b0)

    def _g_scalar(self, cos_val):
        """Similarity adjustment g(z) for a Python float."""
        return 2.0 * ((cos_val + 1.0) / 2.0) ** self.t - 1.0

    def _g(self, cos_theta):
        """Similarity adjustment g(z) = 2*((z+1)/2)^t - 1, element-wise."""
        return 2.0 * torch.pow((cos_theta + 1.0) / 2.0, self.t) - 1.0

    def predict_logits(self, embeddings):
        """
        Classification scores used for metrics when this head computes the loss
        directly. The argmax matches the nearest class center on the hypersphere.
        """
        return F.linear(F.normalize(embeddings), F.normalize(self.weight))

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        g_cos = self._g(cosine)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # positive term: target class
        pos_logits = -self.r * (g_cos - self.m) - self.bias
        pos_loss = (self.lam / self.r) * torch.log(1.0 + torch.exp(pos_logits))
        pos_loss = (one_hot * pos_loss).sum(dim=1)

        # negative term: all non-target classes
        neg_logits = self.r * (g_cos + self.m) + self.bias
        neg_loss = ((1.0 - self.lam) / self.r) * torch.log(1.0 + torch.exp(neg_logits))
        neg_loss = ((1.0 - one_hot) * neg_loss).sum(dim=1)

        loss = (pos_loss + neg_loss).mean()
        return loss
