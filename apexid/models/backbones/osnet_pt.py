from __future__ import annotations

import os
import warnings
from collections import OrderedDict

import torch
from torch import nn

from .osnet import (
    osnet_ibn_x1_0,
    osnet_x0_25,
    osnet_x0_5,
    osnet_x0_75,
    osnet_x1_0,
)

__all__ = [
    "PretrainedOSNet",
    "WaveletStem",
    "init_pretrained_weights",
    "osnet_ibn_x1_0_pt",
    "osnet_x0_25_pt",
    "osnet_x0_5_pt",
    "osnet_x0_75_pt",
    "osnet_x1_0_pt",
]

PRETRAINED_URLS = {
    "osnet_x1_0": "https://drive.google.com/uc?id=1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY",
    "osnet_x0_75": "https://drive.google.com/uc?id=1uwA9fElHOk3ZogwbeY5GkLI6QPTX70Hq",
    "osnet_x0_5": "https://drive.google.com/uc?id=16DGLbZukvVYgINws8u8deSaOqjybZ83i",
    "osnet_x0_25": "https://drive.google.com/uc?id=1rb8UN5ZzPKRc_xvtHlyDh-cSz88YX9hs",
    "osnet_ibn_x1_0": "https://drive.google.com/uc?id=1sr90V6irlYYDd4_4ISU2iruoRG8J__6l",
}


class WaveletStem(nn.Module):
    """Project wavelet RGB+LH+HL+HH inputs to 3 channels for pretrained conv1."""

    def __init__(self, in_channels=6, out_channels=3):
        super().__init__()
        if in_channels != 6 or out_channels != 3:
            raise ValueError("WaveletStem expects in_channels=6 and out_channels=3.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self._init_params()

    def _init_params(self):
        with torch.no_grad():
            self.proj.weight.zero_()
            for channel in range(self.out_channels):
                # RGB passthrough: R->0, G->1, B->2
                self.proj.weight[channel, channel, 0, 0] = 1.0
                # Wavelet passthrough: LH->0, HL->1, HH->2
                self.proj.weight[channel, channel + 3, 0, 0] = 1.0
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        x = self.proj(x)
        x = self.bn(x)
        return self.activation(x)


class PretrainedOSNet(nn.Module):
    """OSNet wrapper that keeps conv1 at 3 channels and optionally adapts wavelet input."""

    def __init__(self, backbone, input_channels=3):
        super().__init__()
        self.input_channels = int(input_channels)
        if self.input_channels not in {3, 6}:
            raise ValueError(
                "Pretrained OSNet supports input_channels 3 or 6, "
                f"got {self.input_channels}."
            )
        if backbone.input_channels != 3:
            raise ValueError("Pretrained OSNet backbone must be built with input_channels=3.")

        self.stem = WaveletStem() if self.input_channels == 6 else nn.Identity()
        self.backbone = backbone
        self.embedding_dim = backbone.embedding_dim
        self.feature_dim = backbone.feature_dim

    def featuremaps(self, x):
        x = self.stem(x)
        return self.backbone.featuremaps(x)

    def forward(self, x):
        x = self.stem(x)
        return self.backbone(x)


def init_pretrained_weights(model, key: str):
    """Load ImageNet-pretrained weights where parameter names and shapes match."""
    if key not in PRETRAINED_URLS:
        raise ValueError(f"Unknown pretrained OSNet key '{key}'.")

    try:
        import gdown
    except ImportError as exc:
        raise ImportError(
            "Loading pretrained OSNet weights requires the 'gdown' package."
        ) from exc

    cache_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    os.makedirs(cache_dir, exist_ok=True)
    cached_file = os.path.join(cache_dir, f"{key}_imagenet.pth")

    if not os.path.exists(cached_file):
        gdown.download(PRETRAINED_URLS[key], cached_file, quiet=False)

    state_dict = torch.load(cached_file, map_location="cpu", weights_only=False)
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers = []
    discarded_layers = []

    for key_name, value in state_dict.items():
        if key_name.startswith("module."):
            key_name = key_name[7:]
        if key_name in model_dict and model_dict[key_name].size() == value.size():
            new_state_dict[key_name] = value
            matched_layers.append(key_name)
        else:
            discarded_layers.append(key_name)

    if not matched_layers:
        warnings.warn(
            f'The pretrained weights from "{cached_file}" cannot be loaded; '
            "no parameter shapes matched."
        )
        return {
            "cached_file": cached_file,
            "matched_layers": matched_layers,
            "discarded_layers": discarded_layers,
        }

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    return {
        "cached_file": cached_file,
        "matched_layers": matched_layers,
        "discarded_layers": discarded_layers,
    }


def _make_pretrained_builder(base_builder, weight_key: str):
    def builder(embedding_dim=512, input_channels=3, pretrained=True, **kwargs):
        core = base_builder(embedding_dim=embedding_dim, input_channels=3, **kwargs)
        if pretrained:
            init_pretrained_weights(core, key=weight_key)
        return PretrainedOSNet(core, input_channels=input_channels)

    return builder


osnet_x1_0_pt = _make_pretrained_builder(osnet_x1_0, "osnet_x1_0")
osnet_x0_75_pt = _make_pretrained_builder(osnet_x0_75, "osnet_x0_75")
osnet_x0_5_pt = _make_pretrained_builder(osnet_x0_5, "osnet_x0_5")
osnet_x0_25_pt = _make_pretrained_builder(osnet_x0_25, "osnet_x0_25")
osnet_ibn_x1_0_pt = _make_pretrained_builder(osnet_ibn_x1_0, "osnet_ibn_x1_0")
