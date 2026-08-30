from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from apexid.utils.provenance import sha256_file

from .osnet_pt import WaveletStem

__all__ = [
    "DINOv3ConvNeXtBackbone",
    "dinov3_convnext_base",
    "dinov3_convnext_base_pt",
    "dinov3_convnext_large",
    "dinov3_convnext_large_pt",
    "dinov3_convnext_small",
    "dinov3_convnext_small_pt",
    "dinov3_convnext_tiny",
    "dinov3_convnext_tiny_pt",
]


_DINOV3_CONVNEXT_REPOS = {
    "tiny": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    "small": "facebook/dinov3-convnext-small-pretrain-lvd1689m",
    "base": "facebook/dinov3-convnext-base-pretrain-lvd1689m",
    "large": "facebook/dinov3-convnext-large-pretrain-lvd1689m",
}
DINO_SMALL_REVISION = "421020015aa4266363db803b6049a529999afb1e"
DINO_SMALL_WEIGHT_SHA256 = "8182492b73345ceb449ed6be0de486d778a73afb273ba2684de785a71ef1da9d"


def _transformers_classes():
    try:
        from transformers import AutoConfig, AutoModel
    except ImportError as exc:
        raise ImportError(
            "DINOv3 ConvNeXt backbones require the 'transformers' package. "
            "Install the dependencies from requirements.txt."
        ) from exc
    return AutoConfig, AutoModel


def _hf_hub_download(**kwargs):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("Pinned DINOv3 weights require the 'huggingface-hub' package.") from exc
    return hf_hub_download(**kwargs)


def _verify_pretrained_artifact(
    repo_id: str,
    revision: str | None,
    expected_sha256: str | None,
    filename: str,
    download_kwargs: Mapping[str, object],
) -> dict[str, str | None]:
    if expected_sha256 is None:
        return {"repo_id": repo_id, "revision": revision, "filename": filename, "sha256": None, "path": None}
    if not revision:
        raise ValueError("An immutable Hugging Face revision is required when expected_sha256 is set")
    supported = {"cache_dir", "force_download", "local_files_only", "token"}
    kwargs = {key: value for key, value in download_kwargs.items() if key in supported}
    path = _hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, **kwargs)
    actual_sha256 = sha256_file(path)
    if actual_sha256.lower() != str(expected_sha256).lower():
        raise RuntimeError(
            f"DINOv3 weight SHA-256 mismatch for {repo_id}@{revision}/{filename}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    return {"repo_id": repo_id, "revision": revision, "filename": filename, "sha256": actual_sha256, "path": str(path)}


def _repo_for_variant(variant: str) -> str:
    try:
        return _DINOV3_CONVNEXT_REPOS[str(variant).lower()]
    except KeyError as exc:
        available = ", ".join(sorted(_DINOV3_CONVNEXT_REPOS))
        raise ValueError(
            f"Unknown DINOv3 ConvNeXt variant '{variant}'. Available variants: {available}"
        ) from exc


def _pooled_feature_dim(config) -> int:
    hidden_sizes = getattr(config, "hidden_sizes", None)
    if hidden_sizes:
        return int(hidden_sizes[-1])

    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is not None:
        return int(hidden_size)

    raise ValueError(
        "Could not determine the DINOv3 ConvNeXt pooled feature dimension from its config."
    )


def _init_projection(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class DINOv3ConvNeXtBackbone(nn.Module):
    """Hugging Face DINOv3 ConvNeXt wrapper returning ApexID embeddings."""

    def __init__(
        self,
        variant: str,
        embedding_dim: int = 512,
        input_channels: int = 3,
        pretrained: bool = False,
        model_name: str | None = None,
        repo_id: str | None = None,
        revision: str | None = None,
        expected_sha256: str | None = None,
        weight_filename: str = "model.safetensors",
        **kwargs,
    ):
        super().__init__()
        self.variant = str(variant).lower()
        self.embedding_dim = int(embedding_dim)
        self.feature_dim = self.embedding_dim
        self.input_channels = int(input_channels)
        self.pretrained = bool(pretrained)
        self.model_name = str(repo_id or model_name or _repo_for_variant(self.variant))
        self.revision = revision
        self.expected_sha256 = expected_sha256
        self.weight_filename = str(weight_filename)

        if self.embedding_dim <= 0:
            raise ValueError("DINOv3 ConvNeXt embedding_dim must be a positive integer.")
        if self.input_channels not in {3, 6}:
            raise ValueError(
                "DINOv3 ConvNeXt supports input_channels 3 or 6, "
                f"got {self.input_channels}."
            )

        AutoConfig, AutoModel = _transformers_classes()
        if self.pretrained:
            self.pretrained_provenance = _verify_pretrained_artifact(
                self.model_name, self.revision, self.expected_sha256, self.weight_filename, kwargs,
            )
            pretrained_kwargs = dict(kwargs)
            if self.revision is not None:
                pretrained_kwargs["revision"] = self.revision
            self.model = AutoModel.from_pretrained(self.model_name, **pretrained_kwargs)
        else:
            config_kwargs = dict(kwargs)
            if self.revision is not None:
                config_kwargs["revision"] = self.revision
            config = AutoConfig.from_pretrained(self.model_name, **config_kwargs)
            self.model = AutoModel.from_config(config)
            self.pretrained_provenance = None

        self.stem = WaveletStem() if self.input_channels == 6 else nn.Identity()
        pooled_dim = _pooled_feature_dim(self.model.config)
        self.projection = nn.Linear(pooled_dim, self.embedding_dim)
        _init_projection(self.projection)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        outputs = self.model(pixel_values=x)
        pooled_output = getattr(outputs, "pooler_output", None)
        if pooled_output is None and isinstance(outputs, Mapping):
            pooled_output = outputs.get("pooler_output")
        if pooled_output is None:
            raise ValueError("DINOv3 ConvNeXt output does not contain pooler_output.")
        return self.projection(pooled_output)


def _make_dinov3_convnext_builder(variant: str, pretrained_default: bool = False):
    def builder(
        embedding_dim: int = 512,
        input_channels: int = 3,
        pretrained: bool = pretrained_default,
        **kwargs,
    ):
        return DINOv3ConvNeXtBackbone(
            variant=variant,
            embedding_dim=embedding_dim,
            input_channels=input_channels,
            pretrained=pretrained,
            **kwargs,
        )

    return builder


dinov3_convnext_tiny = _make_dinov3_convnext_builder("tiny")
dinov3_convnext_small = _make_dinov3_convnext_builder("small")
dinov3_convnext_base = _make_dinov3_convnext_builder("base")
dinov3_convnext_large = _make_dinov3_convnext_builder("large")

dinov3_convnext_tiny_pt = _make_dinov3_convnext_builder("tiny", pretrained_default=True)
dinov3_convnext_small_pt = _make_dinov3_convnext_builder("small", pretrained_default=True)
dinov3_convnext_base_pt = _make_dinov3_convnext_builder("base", pretrained_default=True)
dinov3_convnext_large_pt = _make_dinov3_convnext_builder("large", pretrained_default=True)
