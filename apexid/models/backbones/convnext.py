from __future__ import annotations

import importlib.metadata
from pathlib import Path
from urllib.parse import urlparse

import torch
from torch import nn
from torchvision import models

from apexid.utils.provenance import sha256_file

def _init_convnext_projection(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ConvNeXtBackbone(nn.Module):
    """TorchVision ConvNeXt wrapper that returns ApexID embedding vectors."""

    def __init__(
        self,
        embedding_dim: int = 2048,
        input_channels: int = 3,
        pretrained: bool = False,
        progress: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.feature_dim = self.embedding_dim
        self.input_channels = int(input_channels)
        self.pretrained = bool(pretrained)

        if self.embedding_dim <= 0:
            raise ValueError("ConvNeXt embedding_dim must be a positive integer.")
        if self.input_channels != 3:
            raise ValueError("ConvNeXt-Small expects three RGB channels.")
        if "weights" in kwargs:
            raise ValueError("Use pretrained=True/False for ConvNeXt weights; do not pass weights directly.")
        if "num_classes" in kwargs:
            raise ValueError("Use embedding_dim for ConvNeXt output size; do not pass num_classes.")

        weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if self.pretrained else None
        self.model = models.convnext_small(weights=weights, progress=progress, **kwargs)
        self.pretrained_provenance = None
        if weights is not None:
            filename = Path(urlparse(weights.url).path).name
            cached_path = Path(torch.hub.get_dir()) / "checkpoints" / filename
            self.pretrained_provenance = {
                "provider": "torchvision",
                "torchvision_version": importlib.metadata.version("torchvision"),
                "weights_enum": f"{type(weights).__name__}.{weights.name}",
                "url": weights.url,
                "cached_file": str(cached_path),
                "sha256": sha256_file(cached_path) if cached_path.is_file() else None,
                "filename_hash_prefix": filename.rsplit("-", 1)[-1].removesuffix(".pth") if "-" in filename else None,
            }

        in_features = self._classifier_in_features()
        self._replace_classifier(in_features)

    def _classifier_in_features(self) -> int:
        classifier = getattr(self.model, "classifier", None)
        if not isinstance(classifier, nn.Sequential) or len(classifier) == 0:
            raise ValueError("TorchVision ConvNeXt classifier layout is not supported.")
        linear = classifier[-1]
        if not isinstance(linear, nn.Linear):
            raise ValueError("TorchVision ConvNeXt final classifier layer is not a Linear layer.")
        return int(linear.in_features)

    def _replace_classifier(self, in_features: int) -> None:
        projection = nn.Linear(in_features, self.embedding_dim)
        _init_convnext_projection(projection)
        self.model.classifier[len(self.model.classifier) - 1] = projection

    def featuremaps(self, x):
        return self.model.features(x)

    def forward(self, x):
        return self.model(x)


def _make_convnext_builder(pretrained_default: bool = False):
    def builder(
        embedding_dim: int = 2048,
        input_channels: int = 3,
        pretrained: bool = pretrained_default,
        progress: bool = True,
        **kwargs,
    ):
        return ConvNeXtBackbone(
            embedding_dim=embedding_dim,
            input_channels=input_channels,
            pretrained=pretrained,
            progress=progress,
            **kwargs,
        )

    return builder


convnext_small = _make_convnext_builder(pretrained_default=False)
convnext_small_pt = _make_convnext_builder(pretrained_default=True)
