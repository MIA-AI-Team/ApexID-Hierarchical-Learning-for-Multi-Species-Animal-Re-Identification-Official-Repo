from __future__ import annotations

import warnings
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

from apexid.utils.provenance import sha256_file

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

OSNET_HF_REPO = "kaiyangzhou/osnet"
OSNET_HF_REVISION = "a5c5cc037c24235cda3b21085b93ad77c9616224"
PRETRAINED_ARTIFACTS = {
    "osnet_x1_0": ("osnet_x1_0_imagenet.pth", "fe2d63f9157c28a4a8d8ca29bec12d5b2988ac0346d712025789ea9174968e79"),
    "osnet_x0_75": ("osnet_x0_75_imagenet.pth", "37add23d2ebc7a22584203d6c22976fdda0f7f29c7b7d79257ac22ae20aec809"),
    "osnet_x0_5": ("osnet_x0_5_imagenet.pth", "f10ea9e26ee55b1a65cee3c6e88aa1ae105fe981af3e37692fa68a72fc64bf8e"),
    "osnet_x0_25": ("osnet_x0_25_imagenet.pth", "f54941a66bad4ddd07f2907f498c810ce639ce7a1abeaf2a151f8da118d84693"),
    "osnet_ibn_x1_0": ("osnet_ibn_x1_0_imagenet.pth", "4e6fbaa3f5e31170aae76372a9ebbe36551e4011a4ed88b139d50bc98729e23a"),
}


def _hf_hub_download(**kwargs):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("Loading pinned pretrained OSNet weights requires 'huggingface-hub'.") from exc
    return hf_hub_download(**kwargs)


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


def init_pretrained_weights(
    model,
    key: str,
    *,
    repo_id: str = OSNET_HF_REPO,
    revision: str = OSNET_HF_REVISION,
    expected_sha256: str | None = None,
):
    """Load ImageNet-pretrained weights where parameter names and shapes match."""
    if key not in PRETRAINED_ARTIFACTS:
        raise ValueError(f"Unknown pretrained OSNet key '{key}'.")
    filename, pinned_sha256 = PRETRAINED_ARTIFACTS[key]
    expected_sha256 = pinned_sha256 if expected_sha256 is None else expected_sha256
    cached_file = Path(_hf_hub_download(repo_id=repo_id, filename=filename, revision=revision))
    actual_sha256 = sha256_file(cached_file)
    if actual_sha256.lower() != str(expected_sha256).lower():
        raise RuntimeError(
            f"OSNet weight SHA-256 mismatch for {repo_id}@{revision}/{filename}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )

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
            "cached_file": str(cached_file),
            "repo_id": repo_id,
            "revision": revision,
            "sha256": actual_sha256,
            "matched_layers": matched_layers,
            "discarded_layers": discarded_layers,
        }

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    return {
        "cached_file": str(cached_file),
        "repo_id": repo_id,
        "revision": revision,
        "sha256": actual_sha256,
        "matched_layers": matched_layers,
        "discarded_layers": discarded_layers,
    }


def _make_pretrained_builder(base_builder, weight_key: str):
    def builder(embedding_dim=512, input_channels=3, pretrained=True, **kwargs):
        core = base_builder(embedding_dim=embedding_dim, input_channels=3, **kwargs)
        provenance = None
        if pretrained:
            provenance = init_pretrained_weights(core, key=weight_key)
        wrapped = PretrainedOSNet(core, input_channels=input_channels)
        wrapped.pretrained_provenance = provenance
        return wrapped

    return builder


osnet_x1_0_pt = _make_pretrained_builder(osnet_x1_0, "osnet_x1_0")
osnet_x0_75_pt = _make_pretrained_builder(osnet_x0_75, "osnet_x0_75")
osnet_x0_5_pt = _make_pretrained_builder(osnet_x0_5, "osnet_x0_5")
osnet_x0_25_pt = _make_pretrained_builder(osnet_x0_25, "osnet_x0_25")
osnet_ibn_x1_0_pt = _make_pretrained_builder(osnet_ibn_x1_0, "osnet_ibn_x1_0")
