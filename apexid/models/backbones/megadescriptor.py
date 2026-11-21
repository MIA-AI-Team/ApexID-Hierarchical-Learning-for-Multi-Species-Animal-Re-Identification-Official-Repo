"""MegaDescriptor-L-384 wildlife embeddings using timm's Swin loader."""

from torch import nn


MEGADESCRIPTOR_REPO = "BVRA/MegaDescriptor-L-384"
MEGADESCRIPTOR_REVISION = "33b3c6f4ee0c386a4126cc3dcd23843920613fa1"


def _create_model(*args, **kwargs):
    import timm

    return timm.create_model(*args, **kwargs)


class MegaDescriptorBackbone(nn.Module):
    def __init__(
        self,
        embedding_dim=1536,
        input_channels=3,
        pretrained=False,
        revision=MEGADESCRIPTOR_REVISION,
        cache_dir=None,
        grad_checkpointing=False,
    ):
        super().__init__()
        self.embedding_dim = self.feature_dim = int(embedding_dim)
        self.input_channels = int(input_channels)
        self.pretrained = bool(pretrained)
        if self.embedding_dim <= 0:
            raise ValueError("MegaDescriptor embedding_dim must be positive.")
        if self.input_channels != 3:
            raise ValueError("MegaDescriptor requires input_channels=3 for RGB images.")

        # timm handles the original checkpoint's Swin key conversion. The
        # scratch path stays entirely offline, including checkpoint restoration.
        source = (
            f"hf-hub:{MEGADESCRIPTOR_REPO}@{revision}" if self.pretrained
            else "swin_large_patch4_window12_384"
        )
        self.model = _create_model(source, pretrained=self.pretrained, num_classes=0, cache_dir=cache_dir)
        if grad_checkpointing:
            self.model.set_grad_checkpointing(True)
        self.pretrained_provenance = (
            {"repo_id": MEGADESCRIPTOR_REPO, "revision": revision}
            if self.pretrained else None
        )
        self.projection = (
            nn.Identity() if self.embedding_dim == self.model.num_features
            else nn.Linear(self.model.num_features, self.embedding_dim)
        )

    def forward(self, x):
        if tuple(x.shape[-2:]) != (384, 384):
            raise ValueError("MegaDescriptor-L-384 requires data.image_size: [384, 384].")
        return self.projection(self.model(x))


def megadescriptor_l_384(pretrained=False, **kwargs):
    return MegaDescriptorBackbone(pretrained=pretrained, **kwargs)


def megadescriptor_l_384_pt(pretrained=True, **kwargs):
    return MegaDescriptorBackbone(pretrained=pretrained, **kwargs)
