import timm
import torch.nn as nn
import torch.nn.functional as F


class TimmEmbeddingBackbone(nn.Module):
    """Wraps any timm model as a feature-extraction backbone that outputs
    L2-normalized embeddings, matching the ``ApexID.forward`` contract."""

    def __init__(self, model_name, embedding_dim, pretrained=True):
        super().__init__()
        self.encoder = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        num_features = self.encoder.num_features
        self.neck = nn.BatchNorm1d(num_features)
        self.embedding = nn.Linear(num_features, embedding_dim, bias=False)

    def forward(self, x):
        x = self.encoder(x)
        x = self.neck(x)
        x = self.embedding(x)
        x = F.normalize(x)
        return x
