"""Load ApexID embedding checkpoints and read frozen evaluation images."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from apexid.models import build_model
from apexid.utils.provenance import sha256_file


LOGGER = logging.getLogger(__name__)


def read_path_map(path):
    """Load old-to-new image prefix mappings, longest prefix first."""
    if not path:
        return []
    mapping = json.loads(Path(path).read_text(encoding="utf-8"))
    return sorted(
        ((prefix.replace("\\", "/").rstrip("/"), Path(root)) for prefix, root in mapping.items()),
        key=lambda item: len(item[0]), reverse=True,
    )


def remap_path(path, path_map):
    path = path.replace("\\", "/")
    for prefix, root in path_map:
        if path.startswith(prefix + "/"):
            return str(root / path[len(prefix) + 1:])
    return path


class BenchmarkImageDataset(Dataset):
    """Public-manifest image loader using ApexID's evaluation transform."""

    def __init__(self, frame: pd.DataFrame, data_root: Path, transform: Any):
        self.frame = frame.reset_index(drop=True).copy()
        self.data_root = data_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str, str]:
        row = self.frame.iloc[index]
        candidate = Path(str(row.path))
        if not candidate.is_absolute():
            candidate = self.data_root / candidate
        image = cv2.imread(str(candidate), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read benchmark image: {candidate}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=image)["image"]
        return transformed, str(row.image_id), str(row.path)


class ApexIDModelAdapter:
    """Load the repository's embedding model without constructing a surrogate."""

    def __init__(self, config: Mapping[str, Any], device: torch.device, amp: bool = True):
        self.config = copy.deepcopy(dict(config))
        self.device = device
        self.model = build_model(self.config).to(device).eval()
        self.amp = bool(amp and device.type == "cuda")
        self.checkpoint_metadata: dict[str, Any] = {}

    @classmethod
    def from_checkpoint(cls, config, weights, device, *, amp=True):
        """Restore the saved encoder without downloading its initializer."""
        model_config = copy.deepcopy(config)
        model_config["model"]["backbone"].setdefault("kwargs", {})["pretrained"] = False
        adapter = cls(model_config, device, amp=amp)
        adapter.load_weights(weights)
        return adapter

    @staticmethod
    def _checkpoint_state(payload: Any) -> tuple[Mapping[str, torch.Tensor], dict[str, Any]]:
        if not isinstance(payload, Mapping):
            raise TypeError("Checkpoint must be a state dictionary or a mapping containing one")
        for key in ("model_state_dict", "state_dict"):
            value = payload.get(key)
            if isinstance(value, Mapping):
                return value, {key_: value_ for key_, value_ in payload.items() if key_ != key}
        if all(torch.is_tensor(value) for value in payload.values()):
            return payload, {}
        raise KeyError("Checkpoint has no model_state_dict/state_dict and is not a raw tensor state dict")

    def load_weights(self, weights: Path) -> dict[str, Any]:
        weights = Path(weights)
        try:
            payload = torch.load(weights, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(weights, map_location="cpu")
        state, metadata = self._checkpoint_state(payload)
        normalized = {str(key).removeprefix("module."): value for key, value in state.items()}
        discarded_head_tensors = sum(key.startswith("head.") for key in normalized)
        # Trainer checkpoints save ModuleDict({"backbone": ApexID, "head": ...}).
        if any(key.startswith("backbone.backbone.") for key in normalized):
            normalized = {
                key.removeprefix("backbone."): value
                for key, value in normalized.items()
                if key.startswith("backbone.")
            }
        expected = self.model.state_dict()
        matched = {key: value for key, value in normalized.items() if key in expected}
        unexpected = [key for key in normalized if key not in expected and not key.startswith("head.")]
        missing = [key for key in expected if key not in matched]
        shape_errors = [
            key for key, value in matched.items()
            if tuple(value.shape) != tuple(expected[key].shape)
        ]
        if missing or unexpected or shape_errors:
            raise RuntimeError(
                "Incompatible ApexID embedding weights: "
                f"missing={missing[:5]} ({len(missing)}), unexpected={unexpected[:5]} ({len(unexpected)}), "
                f"shape_mismatches={shape_errors[:5]} ({len(shape_errors)})"
            )
        self.model.load_state_dict(matched, strict=True)
        self.model.eval()
        self.checkpoint_metadata = {
            "checkpoint_path": str(weights), "weights_sha256": sha256_file(weights),
            "epoch": metadata.get("epoch"), "next_epoch": metadata.get("next_epoch"),
            "discarded_head_tensors": discarded_head_tensors,
            "embedding_dim": int(self.model.embedding_dim),
            "parameter_count": int(sum(parameter.numel() for parameter in self.model.parameters())),
            "device": str(self.device),
            "pretrained_artifact": getattr(self.model.backbone, "pretrained_provenance", None),
        }
        return self.checkpoint_metadata

    def extract(self, loader: DataLoader) -> tuple[np.ndarray, list[str], list[str]]:
        embeddings: list[np.ndarray] = []
        image_ids: list[str] = []
        paths: list[str] = []
        with torch.inference_mode():
            for images, batch_ids, batch_paths in loader:
                images = images.to(self.device, non_blocking=True)
                with torch.autocast(device_type="cuda", enabled=self.amp):
                    output = self.model(images)
                output = torch.nn.functional.normalize(output.float(), dim=1)
                if not torch.isfinite(output).all():
                    raise ValueError("Model produced NaN or infinite embeddings")
                embeddings.append(output.cpu().numpy().astype(np.float32, copy=False))
                image_ids.extend(map(str, batch_ids))
                paths.extend(map(str, batch_paths))
        result = np.concatenate(embeddings, axis=0) if embeddings else np.empty((0, self.model.embedding_dim), dtype=np.float32)
        norms = np.linalg.norm(result, axis=1)
        if result.size and not np.allclose(norms, 1.0, atol=1e-3):
            LOGGER.warning("Embedding norms were renormalized; range=[%.5f, %.5f]", norms.min(), norms.max())
            result = l2_normalize(result)
        return result, image_ids, paths


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
        raise ValueError("Embeddings must be a finite 2D matrix")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Embeddings contain zero-norm rows")
    return embeddings / norms
