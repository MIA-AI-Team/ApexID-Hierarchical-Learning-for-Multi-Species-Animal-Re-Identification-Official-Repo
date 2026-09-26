"""MiewID-msv3 embedding architecture and pinned wildlife weights.

Matches conservationxlabs/miewid-msv3's EfficientNetV2 + GeM + BN inference
path without executing Hub code or downloading redundant ImageNet weights.
Reference: https://huggingface.co/conservationxlabs/miewid-msv3/blob/
4f1d7f2b521149e5fe34bb85f377248ce9971a7d/modeling_miewid.py
"""

import torch
from torch import nn

from apexid.utils.provenance import sha256_file


MIEWID_REPO = "conservationxlabs/miewid-msv3"
MIEWID_REVISION = "4f1d7f2b521149e5fe34bb85f377248ce9971a7d"
MIEWID_SHA256 = "adff92b39678f37eb74861c6399a741639a8907ec2382738e903d6120727b348"


def _create_model(*args, **kwargs):
    import timm

    return timm.create_model(*args, **kwargs)


def _download_weights(**kwargs):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(**kwargs)


class GeM(nn.Module):
    """Learnable generalized mean pooling, retaining the checkpoint's p."""

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.tensor([3.0]))

    def forward(self, x):
        return x.clamp(min=1e-6).pow(self.p).mean(dim=(-2, -1), keepdim=True).pow(1.0 / self.p)


class _MiewIDEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _create_model("efficientnetv2_rw_m", pretrained=False, num_classes=0)
        self.backbone.global_pool = GeM()
        self.bn = nn.BatchNorm1d(self.backbone.num_features)

    def forward(self, x):
        return self.bn(self.backbone(x).flatten(1))


class MiewIDBackbone(nn.Module):
    def __init__(
        self,
        embedding_dim=2152,
        input_channels=3,
        pretrained=False,
        revision=MIEWID_REVISION,
        expected_sha256=MIEWID_SHA256,
        cache_dir=None,
        local_files_only=False,
    ):
        super().__init__()
        self.embedding_dim = self.feature_dim = int(embedding_dim)
        self.input_channels = int(input_channels)
        self.pretrained = bool(pretrained)
        if self.embedding_dim <= 0:
            raise ValueError("MiewID embedding_dim must be positive.")
        if self.input_channels != 3:
            raise ValueError("MiewID-msv3 requires input_channels=3 and data.wavelet=false.")

        self.model = _MiewIDEmbedding()
        self.pretrained_provenance = None
        if self.pretrained:
            from safetensors.torch import load_file

            path = _download_weights(
                repo_id=MIEWID_REPO,
                filename="model.safetensors",
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            digest = sha256_file(path)
            if expected_sha256 is not None and digest.lower() != str(expected_sha256).lower():
                raise RuntimeError(f"MiewID weight SHA-256 mismatch: expected {expected_sha256}, got {digest}")
            state = load_file(str(path), device="cpu")
            # The provider's original identity classifier is unused by its
            # embedding forward pass. ApexID supplies a new dataset-specific head.
            state.pop("final.weight", None)
            state.pop("final.bias", None)
            self.model.load_state_dict(state, strict=True)
            self.pretrained_provenance = {
                "repo_id": MIEWID_REPO, "revision": revision,
                "filename": "model.safetensors", "sha256": digest,
            }

        native_dim = self.model.bn.num_features
        self.projection = (
            nn.Identity() if self.embedding_dim == native_dim
            else nn.Linear(native_dim, self.embedding_dim)
        )

    def forward(self, x):
        return self.projection(self.model(x))


def miewid_msv3(pretrained=False, **kwargs):
    return MiewIDBackbone(pretrained=pretrained, **kwargs)


def miewid_msv3_pt(pretrained=True, **kwargs):
    return MiewIDBackbone(pretrained=pretrained, **kwargs)
