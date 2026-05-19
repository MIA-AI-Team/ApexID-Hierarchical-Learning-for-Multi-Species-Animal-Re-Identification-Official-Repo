from __future__ import annotations

import colorsys
import contextlib
import json
import os
import shutil
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


@contextlib.contextmanager
def _suppress_native_stderr():
    """Temporarily silence C/C++ libraries that write directly to stderr."""
    sys.stderr.flush()
    try:
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        yield
        return
    saved_stderr_fd = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)


def _import_umap():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("ABSL_LOGGING_MIN_LOG_LEVEL", "3")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with _suppress_native_stderr():
                import umap
    except ImportError as exc:
        raise ImportError(
            "Embedding UMAP plotting requires umap-learn. "
            "Install project requirements or run: pip install umap-learn matplotlib"
        ) from exc
    return umap


def _import_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Embedding UMAP plotting requires matplotlib. "
            "Install project requirements or run: pip install matplotlib"
        ) from exc
    return plt


def _as_numpy(value, name):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)
    if value.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return value


def _validate_inputs(embeddings, labels):
    embeddings = _as_numpy(embeddings, "embeddings").astype(np.float32, copy=False)
    labels = _as_numpy(labels, "labels").reshape(-1)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape (num_samples, embedding_dim).")
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional.")
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError("embeddings and labels must contain the same number of samples.")
    if embeddings.shape[0] < 3:
        raise ValueError("At least 3 samples are required to fit UMAP.")
    return embeddings, labels


def _label_key(label):
    if isinstance(label, np.generic):
        label = label.item()
    return str(label)


def _label_sort_key(label):
    if isinstance(label, np.generic):
        label = label.item()
    if isinstance(label, (int, float)):
        return (0, float(label), "")
    return (1, 0.0, str(label))


def _color_map(labels):
    unique_labels = sorted(set(labels.tolist()), key=_label_sort_key)
    total = max(len(unique_labels), 1)
    colors = {}
    for idx, label in enumerate(unique_labels):
        hue = idx / total
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.70, 0.90)
        colors[_label_key(label)] = "#{:02x}{:02x}{:02x}".format(
            int(red * 255),
            int(green * 255),
            int(blue * 255),
        )
    return colors


def _stratified_indices(labels, max_points, random_state):
    if max_points is None:
        return np.arange(labels.shape[0], dtype=np.int64)

    max_points = int(max_points)
    if max_points <= 0:
        raise ValueError("max_points must be positive when provided.")
    if labels.shape[0] <= max_points:
        return np.arange(labels.shape[0], dtype=np.int64)

    groups = defaultdict(list)
    for idx, label in enumerate(labels.tolist()):
        groups[_label_key(label)].append(idx)

    rng = np.random.default_rng(random_state)
    label_keys = sorted(groups.keys())

    if max_points < len(label_keys):
        selected_labels = label_keys[:max_points]
        selected = [int(rng.choice(groups[label])) for label in selected_labels]
        return np.asarray(sorted(selected), dtype=np.int64)

    total = float(labels.shape[0])
    allocations = {}
    remainders = []
    remaining = max_points
    for label in label_keys:
        exact = max_points * (len(groups[label]) / total)
        count = min(len(groups[label]), max(1, int(np.floor(exact))))
        allocations[label] = count
        remaining -= count
        remainders.append((exact - np.floor(exact), label))

    for _, label in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        if allocations[label] < len(groups[label]):
            allocations[label] += 1
            remaining -= 1

    selected = []
    for label in label_keys:
        group_indices = np.asarray(groups[label], dtype=np.int64)
        count = min(allocations[label], len(group_indices))
        selected.extend(rng.choice(group_indices, size=count, replace=False).tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def _json_path_for(plot_path):
    return plot_path.with_suffix(".json")


def plot_umap_embeddings(
    embeddings,
    labels,
    output_path,
    epoch=None,
    split="val",
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
    random_state=42,
    point_size=8,
    alpha=0.85,
    dpi=200,
    max_points=None,
):
    """Fit UMAP on embeddings and save a color-coded identity plot."""
    embeddings, labels = _validate_inputs(embeddings, labels)
    sample_indices = _stratified_indices(labels, max_points, random_state)
    sampled_embeddings = embeddings[sample_indices]
    sampled_labels = labels[sample_indices]
    if sampled_embeddings.shape[0] < 3:
        raise ValueError("At least 3 sampled points are required to fit UMAP.")

    effective_neighbors = min(int(n_neighbors), sampled_embeddings.shape[0] - 1)
    effective_neighbors = max(2, effective_neighbors)

    umap_module = _import_umap()
    plt = _import_pyplot()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with _suppress_native_stderr():
            reducer = umap_module.UMAP(
                n_neighbors=effective_neighbors,
                min_dist=float(min_dist),
                metric=str(metric),
                random_state=random_state,
            )
            coordinates = reducer.fit_transform(sampled_embeddings)

    color_map = _color_map(sampled_labels)
    point_colors = [color_map[_label_key(label)] for label in sampled_labels.tolist()]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = _json_path_for(output_path)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=point_colors,
        s=float(point_size),
        alpha=float(alpha),
        linewidths=0,
    )
    title = f"{split} embedding UMAP"
    if epoch is not None:
        title = f"{title} - epoch {int(epoch) + 1}"
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(True, linewidth=0.3, alpha=0.25)
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)

    label_counts = Counter(_label_key(label) for label in sampled_labels.tolist())
    metadata = {
        "epoch": None if epoch is None else int(epoch),
        "epoch_number": None if epoch is None else int(epoch) + 1,
        "split": str(split),
        "output_path": str(output_path),
        "metadata_path": str(metadata_path),
        "num_samples": int(sampled_embeddings.shape[0]),
        "num_original_samples": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "max_points": None if max_points is None else int(max_points),
        "umap": {
            "n_neighbors": effective_neighbors,
            "requested_n_neighbors": int(n_neighbors),
            "min_dist": float(min_dist),
            "metric": str(metric),
            "random_state": random_state,
        },
        "label_counts": dict(sorted(label_counts.items())),
        "label_to_color": color_map,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    return {"png": output_path, "json": metadata_path, "metadata": metadata}


class EmbeddingUMAPPlotter:
    """Configurable writer for epoch and best embedding-space plots."""

    def __init__(
        self,
        output_dir,
        best_output_dir=None,
        split="val",
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
        point_size=8,
        alpha=0.85,
        dpi=200,
        max_points=None,
        filename_template="epoch_{epoch_number:03d}_umap.png",
        best_filename="best_embedding_space.png",
    ):
        self.output_dir = Path(output_dir)
        self.best_output_dir = Path(best_output_dir) if best_output_dir is not None else None
        self.split = str(split)
        self.n_neighbors = int(n_neighbors)
        self.min_dist = float(min_dist)
        self.metric = str(metric)
        self.random_state = random_state
        self.point_size = float(point_size)
        self.alpha = float(alpha)
        self.dpi = int(dpi)
        self.max_points = None if max_points is None else int(max_points)
        self.filename_template = str(filename_template)
        self.best_filename = str(best_filename)

    def _epoch_output_path(self, epoch):
        epoch = int(epoch)
        filename = self.filename_template.format(
            epoch=epoch,
            epoch_index=epoch,
            epoch_number=epoch + 1,
        )
        return self.output_dir / filename

    def plot_epoch(self, embeddings, labels, epoch):
        """Save one epoch plot and metadata."""
        return plot_umap_embeddings(
            embeddings=embeddings,
            labels=labels,
            output_path=self._epoch_output_path(epoch),
            epoch=epoch,
            split=self.split,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
            point_size=self.point_size,
            alpha=self.alpha,
            dpi=self.dpi,
            max_points=self.max_points,
        )

    def save_best(self, epoch_artifacts):
        """Copy an epoch plot into the configured best-model artifact location."""
        if self.best_output_dir is None:
            return None

        best_png = self.best_output_dir / self.best_filename
        best_json = _json_path_for(best_png)
        best_png.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(epoch_artifacts["png"], best_png)

        with open(epoch_artifacts["json"], "r", encoding="utf-8") as f:
            metadata = json.load(f)
        metadata["source_output_path"] = str(epoch_artifacts["png"])
        metadata["source_metadata_path"] = str(epoch_artifacts["json"])
        metadata["output_path"] = str(best_png)
        metadata["metadata_path"] = str(best_json)
        with open(best_json, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

        return {"png": best_png, "json": best_json, "metadata": metadata}


__all__ = ["EmbeddingUMAPPlotter", "plot_umap_embeddings"]
