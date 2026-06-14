from __future__ import annotations

from torch import nn
from torchvision import models

from .osnet_pt import WaveletStem

__all__ = [
    "ConvNeXtBackbone",
    "convnext_base",
    "convnext_base_pt",
    "convnext_large",
    "convnext_large_pt",
    "convnext_small",
    "convnext_small_pt",
    "convnext_tiny",
    "convnext_tiny_pt",
]


_CONVNEXT_VARIANTS = {
    "tiny": ("convnext_tiny", "ConvNeXt_Tiny_Weights"),
    "small": ("convnext_small", "ConvNeXt_Small_Weights"),
    "base": ("convnext_base", "ConvNeXt_Base_Weights"),
    "large": ("convnext_large", "ConvNeXt_Large_Weights"),
}


def _builder_for_variant(variant: str):
    try:
        builder_name, _ = _CONVNEXT_VARIANTS[str(variant).lower()]
    except KeyError as exc:
        available = ", ".join(sorted(_CONVNEXT_VARIANTS))
        raise ValueError(f"Unknown ConvNeXt variant '{variant}'. Available variants: {available}") from exc
    return getattr(models, builder_name)


def _imagenet_weights_for_variant(variant: str):
    _, weights_name = _CONVNEXT_VARIANTS[str(variant).lower()]
    return getattr(models, weights_name).IMAGENET1K_V1


def _init_convnext_projection(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ConvNeXtBackbone(nn.Module):
    """TorchVision ConvNeXt wrapper that returns ApexID embedding vectors."""

    def __init__(
        self,
        variant: str,
        embedding_dim: int = 512,
        input_channels: int = 3,
        pretrained: bool = False,
        progress: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.variant = str(variant).lower()
        self.embedding_dim = int(embedding_dim)
        self.feature_dim = self.embedding_dim
        self.input_channels = int(input_channels)
        self.pretrained = bool(pretrained)

        if self.embedding_dim <= 0:
            raise ValueError("ConvNeXt embedding_dim must be a positive integer.")
        if self.input_channels <= 0:
            raise ValueError("ConvNeXt input_channels must be a positive integer.")
        if "weights" in kwargs:
            raise ValueError("Use pretrained=True/False for ConvNeXt weights; do not pass weights directly.")
        if "num_classes" in kwargs:
            raise ValueError("Use embedding_dim for ConvNeXt output size; do not pass num_classes.")
        if self.pretrained and self.input_channels not in {3, 6}:
            raise ValueError(
                "Pretrained ConvNeXt supports input_channels 3 or 6, "
                f"got {self.input_channels}."
            )

        weights = _imagenet_weights_for_variant(self.variant) if self.pretrained else None
        builder = _builder_for_variant(self.variant)
        self.stem = WaveletStem() if self.pretrained and self.input_channels == 6 else nn.Identity()
        self.model = builder(weights=weights, progress=progress, **kwargs)

        if not self.pretrained and self.input_channels != 3:
            self._replace_input_conv(self.input_channels)

        in_features = self._classifier_in_features()
        self._replace_classifier(in_features)

    def _first_conv(self) -> nn.Conv2d:
        try:
            conv = self.model.features[0][0]
        except (AttributeError, IndexError, TypeError) as exc:
            raise ValueError("TorchVision ConvNeXt stem layout is not supported.") from exc
        if not isinstance(conv, nn.Conv2d):
            raise ValueError("TorchVision ConvNeXt first stem layer is not a Conv2d.")
        return conv

    def _replace_input_conv(self, input_channels: int) -> None:
        conv = self._first_conv()
        replacement = nn.Conv2d(
            input_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
        )
        _init_convnext_projection(replacement)
        self.model.features[0][0] = replacement

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
        x = self.stem(x)
        return self.model.features(x)

    def forward(self, x):
        x = self.stem(x)
        return self.model(x)


def _make_convnext_builder(variant: str, pretrained_default: bool = False):
    def builder(
        embedding_dim: int = 512,
        input_channels: int = 3,
        pretrained: bool = pretrained_default,
        progress: bool = True,
        **kwargs,
    ):
        return ConvNeXtBackbone(
            variant=variant,
            embedding_dim=embedding_dim,
            input_channels=input_channels,
            pretrained=pretrained,
            progress=progress,
            **kwargs,
        )

    return builder


convnext_tiny = _make_convnext_builder("tiny", pretrained_default=False)
convnext_small = _make_convnext_builder("small", pretrained_default=False)
convnext_base = _make_convnext_builder("base", pretrained_default=False)
convnext_large = _make_convnext_builder("large", pretrained_default=False)

convnext_tiny_pt = _make_convnext_builder("tiny", pretrained_default=True)
convnext_small_pt = _make_convnext_builder("small", pretrained_default=True)
convnext_base_pt = _make_convnext_builder("base", pretrained_default=True)
convnext_large_pt = _make_convnext_builder("large", pretrained_default=True)
