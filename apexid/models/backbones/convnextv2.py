from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from apexid.utils.provenance import sha256_file

from .osnet_pt import WaveletStem

__all__ = [
    "ConvNeXtV2Backbone",
    "convnextv2_base",
    "convnextv2_base_pt",
    "convnextv2_large",
    "convnextv2_large_pt",
    "convnextv2_tiny",
    "convnextv2_tiny_pt",
]


_CONVNEXTV2_REPOS = {
    "tiny": "facebook/convnextv2-tiny-22k-384",
    "base": "facebook/convnextv2-base-22k-384",
    "large": "facebook/convnextv2-large-22k-384",
}
CONVNEXTV2_BASE_REVISION = "53ec0af6e1bd11bf7eba3e9131d29b69625accf3"
CONVNEXTV2_BASE_WEIGHT_SHA256 = "70dd4eb0d73cbad0f93236564e093cec20ab76a877d2e4d155512097ebf06fd9"


def _transformers_classes():
    try:
        from transformers import AutoConfig, AutoModel
    except ImportError as exc:
        raise ImportError(
            "ConvNeXt V2 backbones require the 'transformers' package. "
            "Install the dependencies from requirements.txt."
        ) from exc
    return AutoConfig, AutoModel


def _hf_hub_download(**kwargs):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("Pinned ConvNeXt V2 weights require the 'huggingface-hub' package.") from exc
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
            f"ConvNeXt V2 weight SHA-256 mismatch for {repo_id}@{revision}/{filename}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    return {"repo_id": repo_id, "revision": revision, "filename": filename, "sha256": actual_sha256, "path": str(path)}


def _repo_for_variant(variant: str) -> str:
    try:
        return _CONVNEXTV2_REPOS[str(variant).lower()]
    except KeyError as exc:
        available = ", ".join(sorted(_CONVNEXTV2_REPOS))
        raise ValueError(
            f"Unknown ConvNeXt V2 variant '{variant}'. Available variants: {available}"
        ) from exc


def _pooled_feature_dim(config) -> int:
    hidden_sizes = getattr(config, "hidden_sizes", None)
    if hidden_sizes:
        return int(hidden_sizes[-1])

    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is not None:
        return int(hidden_size)

    raise ValueError(
        "Could not determine the ConvNeXt V2 pooled feature dimension from its config."
    )


def _pooler_output(outputs) -> torch.Tensor:
    pooled_output = getattr(outputs, "pooler_output", None)
    if pooled_output is None and isinstance(outputs, Mapping):
        pooled_output = outputs.get("pooler_output")
    if pooled_output is None:
        raise ValueError("ConvNeXt V2 output does not contain pooler_output.")
    return pooled_output


def _init_projection(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ConvNeXtV2Backbone(nn.Module):
    """Hugging Face ConvNeXt V2 wrapper returning ApexID embeddings."""

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
            raise ValueError("ConvNeXt V2 embedding_dim must be a positive integer.")
        if self.input_channels not in {3, 6}:
            raise ValueError(
                "ConvNeXt V2 supports input_channels 3 or 6, "
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
        if self.embedding_dim == pooled_dim:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Linear(pooled_dim, self.embedding_dim)
            _init_projection(self.projection)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        outputs = self.model(pixel_values=x)
        return self.projection(_pooler_output(outputs))


def _make_convnextv2_builder(variant: str, pretrained_default: bool = False):
    def builder(
        embedding_dim: int = 512,
        input_channels: int = 3,
        pretrained: bool = pretrained_default,
        **kwargs,
    ):
        return ConvNeXtV2Backbone(
            variant=variant,
            embedding_dim=embedding_dim,
            input_channels=input_channels,
            pretrained=pretrained,
            **kwargs,
        )

    return builder


convnextv2_tiny = _make_convnextv2_builder("tiny")
convnextv2_base = _make_convnextv2_builder("base")
convnextv2_large = _make_convnextv2_builder("large")

convnextv2_tiny_pt = _make_convnextv2_builder("tiny", pretrained_default=True)
convnextv2_base_pt = _make_convnextv2_builder("base", pretrained_default=True)
convnextv2_large_pt = _make_convnextv2_builder("large", pretrained_default=True)
