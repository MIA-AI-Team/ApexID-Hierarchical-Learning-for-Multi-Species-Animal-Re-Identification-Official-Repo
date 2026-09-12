from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from apexid.utils.provenance import sha256_file

from .osnet_pt import WaveletStem

__all__ = [
    "DINOv3ViTBackbone",
    "dinov3_vitb16",
    "dinov3_vitb16_pt",
    "dinov3_vitl16",
    "dinov3_vitl16_pt",
    "dinov3_vits16",
    "dinov3_vits16_pt",
    "dinov3_vits16plus",
    "dinov3_vits16plus_pt",
]


_DINOV3_VIT_REPOS = {
    "s16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "s16plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "b16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "l16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}
DINO_VITB16_REVISION = "5931719e67bbdb9737e363e781fb0c67687896bc"
DINO_VITB16_WEIGHT_SHA256 = "9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b"
_FROM_PRETRAINED_ONLY = {
    "attn_implementation",
    "device_map",
    "quantization_config",
    "low_cpu_mem_usage",
}


def _transformers_classes():
    try:
        from transformers import AutoConfig, AutoModel
    except ImportError as exc:
        raise ImportError(
            "DINOv3 ViT backbones require the 'transformers' package. "
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
        return _DINOV3_VIT_REPOS[str(variant).lower()]
    except KeyError as exc:
        available = ", ".join(sorted(_DINOV3_VIT_REPOS))
        raise ValueError(
            f"Unknown DINOv3 ViT variant '{variant}'. Available variants: {available}"
        ) from exc


def _hidden_size(config) -> int:
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is not None:
        return int(hidden_size)
    raise ValueError("Could not determine the DINOv3 ViT hidden size from its config.")


def _patch_size(config) -> int:
    patch_size = getattr(config, "patch_size", 16)
    if isinstance(patch_size, (list, tuple)):
        if not patch_size:
            raise ValueError("DINOv3 ViT config.patch_size is empty.")
        return int(patch_size[0])
    return int(patch_size)


def _cls_token(outputs) -> torch.Tensor:
    last_hidden_state = getattr(outputs, "last_hidden_state", None)
    if last_hidden_state is None and isinstance(outputs, Mapping):
        last_hidden_state = outputs.get("last_hidden_state")
    if last_hidden_state is None:
        raise ValueError("DINOv3 ViT output does not contain last_hidden_state.")
    if last_hidden_state.ndim != 3 or last_hidden_state.shape[1] < 1:
        raise ValueError(
            "DINOv3 ViT last_hidden_state must have shape (batch, sequence, hidden), "
            f"got {tuple(last_hidden_state.shape)}."
        )
    return last_hidden_state[:, 0, :]


def _init_projection(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class DINOv3ViTBackbone(nn.Module):
    """Hugging Face DINOv3 ViT wrapper returning native CLS embeddings."""

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
            raise ValueError("DINOv3 ViT embedding_dim must be a positive integer.")
        if self.input_channels not in {3, 6}:
            raise ValueError(
                "DINOv3 ViT supports input_channels 3 or 6, "
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
            pretrained_kwargs.setdefault("attn_implementation", "sdpa")
            self.model = AutoModel.from_pretrained(self.model_name, **pretrained_kwargs)
        else:
            config_kwargs = {
                key: value for key, value in kwargs.items() if key not in _FROM_PRETRAINED_ONLY
            }
            if self.revision is not None:
                config_kwargs["revision"] = self.revision
            config = AutoConfig.from_pretrained(self.model_name, **config_kwargs)
            self.model = AutoModel.from_config(config)
            self.pretrained_provenance = None

        self.stem = WaveletStem() if self.input_channels == 6 else nn.Identity()
        hidden_size = _hidden_size(self.model.config)
        if self.embedding_dim == hidden_size:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Linear(hidden_size, self.embedding_dim)
            _init_projection(self.projection)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        if x.ndim != 4:
            raise ValueError(
                "DINOv3 ViT expects a 4-D image tensor (batch, channels, height, width), "
                f"got shape {tuple(x.shape)}."
            )
        patch_size = _patch_size(self.model.config)
        _, _, height, width = x.shape
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                "DINOv3 ViT input spatial size must be a multiple of "
                f"patch_size={patch_size}, got {height}x{width}."
            )
        outputs = self.model(pixel_values=x)
        return self.projection(_cls_token(outputs))


def _make_dinov3_vit_builder(variant: str, pretrained_default: bool = False):
    def builder(
        embedding_dim: int = 512,
        input_channels: int = 3,
        pretrained: bool = pretrained_default,
        **kwargs,
    ):
        return DINOv3ViTBackbone(
            variant=variant,
            embedding_dim=embedding_dim,
            input_channels=input_channels,
            pretrained=pretrained,
            **kwargs,
        )

    return builder


dinov3_vits16 = _make_dinov3_vit_builder("s16")
dinov3_vits16plus = _make_dinov3_vit_builder("s16plus")
dinov3_vitb16 = _make_dinov3_vit_builder("b16")
dinov3_vitl16 = _make_dinov3_vit_builder("l16")

dinov3_vits16_pt = _make_dinov3_vit_builder("s16", pretrained_default=True)
dinov3_vits16plus_pt = _make_dinov3_vit_builder("s16plus", pretrained_default=True)
dinov3_vitb16_pt = _make_dinov3_vit_builder("b16", pretrained_default=True)
dinov3_vitl16_pt = _make_dinov3_vit_builder("l16", pretrained_default=True)
