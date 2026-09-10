"""Portable, best-effort run provenance for ApexID training and evaluation."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


PROVENANCE_SCHEMA_VERSION = 1
DEFAULT_DEPENDENCIES = (
    "torch", "torchvision", "timm", "huggingface-hub", "transformers",
    "numpy", "pandas", "scikit-learn", "opencv-python-headless", "Pillow",
    "albumentations", "PyYAML", "numba", "llvmlite", "umap-learn",
    "matplotlib", "safetensors", "wandb",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize provenance value {type(value)!r}")


def _atomic_write(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_json_atomic(path: str | Path, value: Any) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, default=_json_default, allow_nan=True) + "\n"
    _atomic_write(Path(path), text)


def write_yaml_atomic(path: str | Path, value: Any) -> None:
    serializable = json.loads(json.dumps(value, default=_json_default))
    text = yaml.safe_dump(serializable, sort_keys=False, allow_unicode=True)
    _atomic_write(Path(path), text)


def dependency_versions(names: Sequence[str] = DEFAULT_DEPENDENCIES) -> dict[str, str | None]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def capture_environment() -> dict[str, Any]:
    """Capture a CPU-safe environment record; CUDA fields remain explicit when absent."""
    record: dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "captured_at_utc": utc_now(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "dependencies": dependency_versions(),
        "cuda": {"available": False, "runtime": None, "cudnn": None, "device_count": 0, "devices": []},
    }
    try:
        import torch

        record["torch"] = torch.__version__
        cuda_available = bool(torch.cuda.is_available())
        devices = []
        if cuda_available:
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                devices.append({
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": int(properties.total_memory),
                    "compute_capability": f"{properties.major}.{properties.minor}",
                })
        record["cuda"] = {
            "available": cuda_available,
            "runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device_count": len(devices),
            "devices": devices,
        }
    except Exception as exc:  # environment capture must remain usable without torch/CUDA
        record["torch_error"] = f"{type(exc).__name__}: {exc}"
    return record


def capture_git_state(path: str | Path = ".") -> dict[str, Any]:
    """Return commit and dirty paths, or an auditable unavailable record."""
    path = Path(path)
    try:
        root = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        status_lines = subprocess.run(
            ["git", "-C", root, "status", "--porcelain=v1", "--untracked-files=all"],
            check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        changed_paths = [line[3:] if len(line) > 3 else line for line in status_lines]
        return {
            "available": True, "root": root, "commit": commit,
            "dirty": bool(status_lines), "changed_paths": changed_paths, "error": None,
        }
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        detail = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) and exc.stderr else str(exc)
        return {
            "available": False, "root": None, "commit": None, "dirty": None,
            "changed_paths": [], "error": detail,
        }


def require_clean_git(state: Mapping[str, Any]) -> None:
    if not state.get("available"):
        raise RuntimeError(f"Clean Git state is required but unverifiable: {state.get('error')}")
    if state.get("dirty"):
        paths = ", ".join(map(str, state.get("changed_paths", [])[:20]))
        raise RuntimeError(f"Clean Git state is required but the worktree is dirty: {paths}")


def command_record(command: Sequence[str] | None = None) -> dict[str, Any]:
    command = list(sys.argv if command is None else command)
    return {"argv": [str(value) for value in command], "cwd": str(Path.cwd())}


def artifact_hashes(paths: Mapping[str, str | Path | None]) -> dict[str, Any]:
    records = {}
    for name, value in paths.items():
        if value in {None, ""}:
            records[name] = {"path": None, "exists": False, "sha256": None, "size_bytes": None}
            continue
        path = Path(value)
        records[name] = {
            "path": str(path), "exists": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
            "size_bytes": path.stat().st_size if path.is_file() else None,
        }
    return records


def validate_normalization_stats(
    path: str | Path,
    *,
    train_csv: str | Path,
    split_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Validate normalization statistics against the train CSV and optional manifest."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Normalization statistics file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_train_hash = payload.get("train_csv_sha256")
    actual_train_hash = sha256_file(train_csv)
    if expected_train_hash and expected_train_hash != actual_train_hash:
        raise ValueError("normalization_stats.json train CSV hash does not match the configured training split")
    if split_manifest is not None and Path(split_manifest).is_file():
        expected_manifest_hash = payload.get("manifest_sha256")
        actual_manifest_hash = sha256_file(split_manifest)
        if expected_manifest_hash and expected_manifest_hash != actual_manifest_hash:
            raise ValueError("normalization_stats.json manifest hash does not match split_manifest.csv")
    std = payload.get("std")
    if std is None:
        std = payload.get("standard_deviation", [])
    if len(payload.get("mean", [])) != 3 or len(std) != 3:
        raise ValueError("normalization_stats.json must contain three-channel mean and standard deviation")
    payload = dict(payload)
    payload["std"] = std
    return payload


__all__ = [
    "PROVENANCE_SCHEMA_VERSION", "artifact_hashes", "capture_environment", "capture_git_state",
    "command_record", "dependency_versions", "require_clean_git", "sha256_file",
    "utc_now", "validate_normalization_stats", "write_json_atomic", "write_yaml_atomic",
]
