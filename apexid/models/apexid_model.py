import torch.nn as nn
from torch.nn import functional as F

from .backbones.registry import build_backbone


class ApexID(nn.Module):
    """Canonical ApexID embedding model.

    The class name remains ApexID while the internal backbone can change across
    experiments. Forward always returns L2-normalized embeddings.
    """

    def __init__(
        self,
        backbone_name="convnext_small",
        embedding_dim=2048,
        input_channels=3,
        backbone_kwargs=None,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.embedding_dim = embedding_dim

        kwargs = dict(backbone_kwargs or {})
        kw_input_channels = kwargs.pop("input_channels", None)
        if kw_input_channels is not None:
            if input_channels != 3 and int(kw_input_channels) != int(input_channels):
                raise ValueError("Pass ApexID input_channels once, not with conflicting values.")
            input_channels = kw_input_channels

        self.input_channels = int(input_channels)
        if self.input_channels <= 0:
            raise ValueError("ApexID input_channels must be a positive integer.")

        kwargs["embedding_dim"] = embedding_dim
        kwargs["input_channels"] = self.input_channels
        self.backbone = build_backbone(backbone_name, **kwargs)

    def forward(self, x):
        embeddings = self.backbone(x)
        return F.normalize(embeddings, dim=1)
