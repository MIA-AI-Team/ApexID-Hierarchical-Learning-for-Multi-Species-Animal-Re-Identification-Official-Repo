import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from transformers import AutoModel
except ImportError:
    AutoModel = None
class MiewIdEmbeddingBackbone(nn.Module):
    """Loads a MiewID checkpoint from Hugging Face (EfficientNetV2 + GeM + BN feature
    extractor), then maps features to embedding_dim and L2-normalizes, matching
    the ApexID / TimmEmbeddingBackbone contract."""
    def __init__(
        self,
        embedding_dim,
        hf_model_id="conservationxlabs/miewid-msv3",
        img_size=440,
        trust_remote_code=True,
    ):
        super().__init__()
        if AutoModel is None:
            raise ImportError(
                "MiewIdEmbeddingBackbone requires the transformers package."
            )
        self.extractor = AutoModel.from_pretrained(
            hf_model_id, trust_remote_code=trust_remote_code
        )
        with torch.no_grad():
            probe = torch.zeros(1, 3, img_size, img_size)
            feat = self.extractor(probe)
        in_dim = int(feat.shape[-1])
        self.embedding = nn.Linear(in_dim, embedding_dim, bias=False)
    def forward(self, x):
        x = self.extractor(x)
        x = self.dropout(x)
        x = self.embedding(x)
        x = F.normalize(x)
        return x
