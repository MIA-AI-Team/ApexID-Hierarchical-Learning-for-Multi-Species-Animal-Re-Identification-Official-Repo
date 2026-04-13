import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoModel
except ImportError:
    AutoModel = None


class MiewIdEmbeddingBackbone(nn.Module):
    """Loads a MiewID checkpoint from Hugging Face (EfficientNetV2 + GeM + BN feature
    extractor), then maps features to ``embedding_dim`` and L2-normalizes, matching
    the ``ApexID`` / ``TimmEmbeddingBackbone`` contract."""

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
                "MiewIdEmbeddingBackbone requires the `transformers` package."
            )
        # copy read-only model dir to a writable tmp location
        writable_model_path = hf_model_id
        if os.path.isdir(hf_model_id) and not os.access(hf_model_id, os.W_OK):
            writable_model_path = f"/tmp/miewid_model"
            if not os.path.exists(writable_model_path):
                print(f"Copying read-only model to {writable_model_path} ...")
                shutil.copytree(hf_model_id, writable_model_path)
            hf_model_id = writable_model_path

        self.extractor = AutoModel.from_pretrained(
            hf_model_id,
            trust_remote_code=trust_remote_code,
            local_files_only=True,          # never hit the network on Kaggle
            cache_dir="/tmp/hf_cache",      # writable cache dir
        )
        
        with torch.no_grad():
            probe = torch.zeros(1, 3, img_size, img_size)
            feat = self.extractor(probe)
        in_dim = int(feat.shape[-1])
        self.embedding = nn.Linear(in_dim, embedding_dim, bias=False)

    def forward(self, x):
        x = self.extractor(x)
        x = self.embedding(x)
        x = F.normalize(x)
        return x
