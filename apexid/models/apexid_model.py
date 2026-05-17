import torch.nn as nn
from torch.nn import functional as F

from .backbones.registry import build_backbone


class ApexID(nn.Module):
    """Canonical ApexID embedding model.

    The class name remains ApexID while the internal backbone can change across
    experiments. Forward always returns L2-normalized embeddings.
    """

    def __init__(self, backbone_name="osnet_x1_0", embedding_dim=512, backbone_kwargs=None):
        super().__init__()
        self.backbone_name = backbone_name
        self.embedding_dim = embedding_dim

        kwargs = dict(backbone_kwargs or {})
        kwargs["embedding_dim"] = embedding_dim
        self.backbone = build_backbone(backbone_name, **kwargs)

    def forward(self, x):
        embeddings = self.backbone(x)
        return F.normalize(embeddings)
