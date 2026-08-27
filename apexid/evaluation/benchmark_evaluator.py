"""Reusable implementation for the sealed ApexID benchmark evaluator.

The public-only inference entrypoints in this module intentionally accept no
final ground-truth dataframe.  Final labels are consumed by ``evaluate_final``
only after callers have persisted predictions.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import logging
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, f1_score,
    precision_recall_curve, precision_recall_fscore_support, roc_auc_score, roc_curve,
)
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from apexid.data import build_eval_transforms
from apexid.evaluation.benchmark_metrics import discovery_metrics, clustering_metrics
from apexid.evaluation.open_set_metrics import compute_known_retrieval_metrics, compute_oscr_auc
from apexid.models import build_model
from apexid.utils.config import DEFAULT_CONFIG_PATH, load_config


LOGGER = logging.getLogger(__name__)
REQUIRED_BENCHMARK_FILES = (
    "database.csv", "development.csv", "development_ground_truth.csv", "final_test.csv",
    "final_test_no_species.csv", "final_test_ground_truth.csv", "known_identity_registry.csv",
    "private_identity_mapping.csv",
)
PUBLIC_FINAL_COLUMNS = ("image_id", "arrival_index", "path", "species", "dataset")
PUBLIC_DEVELOPMENT_COLUMNS = ("image_id", "path", "species", "dataset")
SCENARIO_KNOWN = "known_identity"
SCENARIO_NOVEL = "novel_identity_known_species"
SCENARIO_DISCOVERY = "discovery_only_species"


@dataclass(frozen=True)
class EvaluationConfig:
    weights: Path
    benchmark_dir: Path
    data_root: Path
    output_dir: Path
    device: str = "cuda"
    batch_size: int = 64
    num_workers: int = 4
    image_size: tuple[int, int] | None = None
    seed: int = 42
    model_config: Path = DEFAULT_CONFIG_PATH
    evaluation_mode: str = "both"
    gallery_scoring: str = "centroid"
    threshold_objective: str = "f1"
    discovery_threshold_objective: str = "ari"
    retrieval_chunk_size: int = 1024
    bootstrap_samples: int = 1000
    force_recompute: bool = False
    save_embeddings: bool = True
    update_known_prototypes: bool = False
    amp: bool = True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_benchmark_directory(benchmark_dir: Path) -> dict[str, pd.DataFrame]:
    missing = [name for name in REQUIRED_BENCHMARK_FILES if not (benchmark_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Benchmark directory is missing required files: {missing}")
    frames = {name.removesuffix(".csv"): pd.read_csv(benchmark_dir / name) for name in REQUIRED_BENCHMARK_FILES}
    for name, expected in (("development", PUBLIC_DEVELOPMENT_COLUMNS), ("final_test", PUBLIC_FINAL_COLUMNS)):
        actual = tuple(frames[name].columns)
        if actual != expected:
            raise ValueError(f"{name}.csv must have columns {expected}; got {actual}")
    database = frames["database"]
    needed_db = {"path", "database_id", "species", "dataset", "encounter_id"}
    if not needed_db <= set(database.columns):
        raise ValueError(f"database.csv missing {sorted(needed_db - set(database.columns))}")
    if database["database_id"].isna().any() or database["database_id"].duplicated().all():
        raise ValueError("database.csv must contain valid database IDs")
    for public_name in ("development", "final_test"):
        sensitive = {"identity", "original_identity", "true_identity", "scenario", "known_database_id"}
        if sensitive & set(frames[public_name].columns):
            raise ValueError(f"Public {public_name}.csv exposes private labels")
    return frames


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
        self.model = build_model(self.config).to(device)
        self.amp = bool(amp and device.type == "cuda")
        self.checkpoint_metadata: dict[str, Any] = {}

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


def _cache_metadata(frame: pd.DataFrame, adapter: ApexIDModelAdapter, config: EvaluationConfig) -> dict[str, Any]:
    return {
        "csv_sha256": _frame_hash(frame), "weights_sha256": adapter.checkpoint_metadata["weights_sha256"],
        "model_config": adapter.config, "image_size": config.image_size,
        "embedding_dim": adapter.checkpoint_metadata["embedding_dim"],
    }


def extract_with_cache(
    split: str, frame: pd.DataFrame, adapter: ApexIDModelAdapter, config: EvaluationConfig, transform: Any
) -> tuple[np.ndarray, pd.DataFrame]:
    cache_path = config.output_dir / "cache" / f"{split}_embeddings.npz"
    metadata = _cache_metadata(frame, adapter, config)
    if config.save_embeddings and cache_path.is_file() and not config.force_recompute:
        cached = np.load(cache_path, allow_pickle=False)
        if json.loads(str(cached["metadata"])) == metadata:
            cached_ids = cached["image_ids"].astype(str)
            if np.array_equal(cached_ids, frame["image_id"].astype(str).to_numpy()):
                return l2_normalize(cached["embeddings"]), frame.reset_index(drop=True)
    dataset = BenchmarkImageDataset(frame, config.data_root, transform)
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers,
        pin_memory=adapter.device.type == "cuda", persistent_workers=config.num_workers > 0,
    )
    embeddings, image_ids, paths = adapter.extract(loader)
    expected_ids = frame["image_id"].astype(str).tolist()
    if image_ids != expected_ids or paths != frame["path"].astype(str).tolist():
        raise RuntimeError("Embedding extraction order no longer matches the public benchmark manifest")
    if config.save_embeddings:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, embeddings=embeddings, image_ids=np.asarray(image_ids), metadata=json.dumps(metadata, sort_keys=True))
    return embeddings, frame.reset_index(drop=True)


@dataclass
class Gallery:
    image_embeddings: np.ndarray
    image_database_ids: np.ndarray
    identity_ids: np.ndarray
    centroids: np.ndarray


def build_gallery(database: pd.DataFrame, embeddings: np.ndarray) -> Gallery:
    if len(database) != len(embeddings):
        raise ValueError("Database embedding count does not match database manifest")
    ids = database["database_id"].astype(str).to_numpy()
    identity_ids = np.asarray(sorted(set(ids)), dtype=str)
    centroids = np.vstack([l2_normalize(embeddings[ids == identity]).mean(axis=0) for identity in identity_ids])
    return Gallery(l2_normalize(embeddings), ids, identity_ids, l2_normalize(centroids))


def score_gallery(
    queries: np.ndarray, gallery: Gallery, strategy: str = "centroid", chunk_size: int = 1024, top_k: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return identity score matrix, ranked identity indices, and top scores."""
    queries = l2_normalize(queries)
    if strategy not in {"centroid", "max_image", "mean_top_k"}:
        raise ValueError("gallery_scoring must be centroid, max_image, or mean_top_k")
    all_scores = np.empty((len(queries), len(gallery.identity_ids)), dtype=np.float32)
    for start in range(0, len(queries), max(1, int(chunk_size))):
        stop = min(start + max(1, int(chunk_size)), len(queries))
        block = queries[start:stop]
        if strategy == "centroid":
            all_scores[start:stop] = block @ gallery.centroids.T
        else:
            image_scores = block @ gallery.image_embeddings.T
            for index, identity in enumerate(gallery.identity_ids):
                values = image_scores[:, gallery.image_database_ids == identity]
                if strategy == "max_image":
                    all_scores[start:stop, index] = values.max(axis=1)
                else:
                    take = min(3, values.shape[1])
                    all_scores[start:stop, index] = np.partition(values, values.shape[1] - take, axis=1)[:, -take:].mean(axis=1)
    ranks = np.argsort(-all_scores, axis=1)[:, : min(top_k, all_scores.shape[1])]
    return all_scores, ranks, np.take_along_axis(all_scores, ranks, axis=1)


def retrieval_rows(frame: pd.DataFrame, gallery: Gallery, ranks: np.ndarray, ranked_scores: np.ndarray) -> pd.DataFrame:
    records = []
    for position, image_id in enumerate(frame["image_id"].astype(str)):
        for rank, (identity_index, score) in enumerate(zip(ranks[position], ranked_scores[position]), start=1):
            records.append({"image_id": image_id, "rank": rank, "database_id": gallery.identity_ids[identity_index], "similarity": float(score)})
    return pd.DataFrame.from_records(records, columns=["image_id", "rank", "database_id", "similarity"])


def _threshold_candidates(scores: np.ndarray, limit: int | None = None) -> np.ndarray:
    values = np.unique(np.clip(np.asarray(scores, dtype=float)[np.isfinite(scores)], -1.0, 1.0))
    if not len(values):
        raise ValueError("Cannot tune threshold without finite development scores")
    values = np.unique(np.concatenate(([np.nextafter(values[0], -np.inf)], values, [np.nextafter(values[-1], np.inf)])))
    if limit and len(values) > limit:
        values = np.unique(np.quantile(values, np.linspace(0, 1, limit)))
    return values


def tune_known_threshold(scores: np.ndarray, scenarios: Sequence[str], objective: str) -> tuple[float, pd.DataFrame]:
    scenarios = np.asarray(scenarios, dtype=str)
    mask = np.isin(scenarios, [SCENARIO_KNOWN, SCENARIO_NOVEL])
    if not np.any(mask & (scenarios == SCENARIO_KNOWN)) or not np.any(mask & (scenarios == SCENARIO_NOVEL)):
        raise ValueError("Development set needs both known and novel-known-species labels to tune T_known")
    known = scenarios[mask] == SCENARIO_KNOWN
    rows = []
    best_threshold, best_value = float("nan"), -np.inf
    for threshold in _threshold_candidates(scores[mask]):
        accepted = scores[mask] >= threshold
        predicted_novel = ~accepted
        if objective == "f1":
            value = f1_score(known, accepted, average="macro", zero_division=0)
        elif objective == "balanced_accuracy":
            value = balanced_accuracy_score(known, accepted)
        elif objective == "youden":
            tpr = accepted[known].mean(); fpr = accepted[~known].mean()
            value = tpr - fpr
        else:
            raise ValueError("threshold_objective must be f1, balanced_accuracy, or youden")
        precision, recall, f1, _ = precision_recall_fscore_support(~known, predicted_novel, average="binary", zero_division=0)
        rows.append({"threshold": float(threshold), "precision": precision, "recall": recall, "F1": f1, "balanced_accuracy": balanced_accuracy_score(known, accepted), "TPR": accepted[known].mean(), "FPR": accepted[~known].mean(), "objective": float(value)})
        if value > best_value or (value == best_value and threshold > best_threshold):
            best_threshold, best_value = float(threshold), float(value)
    return best_threshold, pd.DataFrame(rows)


def _online_assign(embeddings: np.ndarray, threshold: float) -> tuple[list[str], list[float], list[bool]]:
    centroids: list[np.ndarray] = []
    counts: list[int] = []
    labels: list[str] = []
    scores: list[float] = []
    created: list[bool] = []
    for embedding in l2_normalize(embeddings):
        if centroids:
            similarities = np.asarray(centroids) @ embedding
            index = int(np.argmax(similarities)); score = float(similarities[index])
        else:
            index, score = -1, float("nan")
        if index >= 0 and score >= threshold:
            label = f"NEW_{index + 1:06d}"
            centroids[index] = l2_normalize((centroids[index] * counts[index] + embedding)[None, :])[0]
            counts[index] += 1
            created.append(False)
        else:
            centroids.append(embedding.copy()); counts.append(1); label = f"NEW_{len(centroids):06d}"; created.append(True)
        labels.append(label); scores.append(score)
    return labels, scores, created


def tune_discovery_threshold(
    embeddings: np.ndarray, labels: Sequence[str], objective: str, seed: int, permutations: int = 10
) -> tuple[float, pd.DataFrame]:
    labels = np.asarray(labels, dtype=str)
    candidates = np.linspace(0.0, 1.0, 51)
    rows = []
    best_threshold, best_value = 0.5, -np.inf
    for threshold in candidates:
        values = []
        for order_seed in range(seed, seed + permutations):
            order = np.random.default_rng(order_seed).permutation(len(labels))
            predicted, _, _ = _online_assign(embeddings[order], float(threshold))
            metrics = discovery_metrics(labels[order], predicted)
            values.append(metrics["ari"] if objective == "ari" else metrics["b3_f1"])
        value = float(np.nanmean(values))
        rows.append({"threshold": float(threshold), "objective": value, "mean_ari": float(np.nanmean(values))})
        if value > best_value or (value == best_value and threshold > best_threshold):
            best_threshold, best_value = float(threshold), value
    return best_threshold, pd.DataFrame(rows)


def cluster_rejected(embeddings: np.ndarray, threshold: float) -> np.ndarray:
    embeddings = l2_normalize(embeddings)
    if len(embeddings) == 0:
        return np.empty(0, dtype=int)
    if len(embeddings) == 1:
        return np.zeros(1, dtype=int)
    model = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average", distance_threshold=float(1.0 - threshold))
    return model.fit_predict(embeddings).astype(int)


def tune_cluster_threshold(embeddings: np.ndarray, labels: Sequence[str], objective: str) -> tuple[float, pd.DataFrame]:
    candidates = np.linspace(0.0, 1.0, 31)
    rows = []
    best_threshold, best_value = 0.5, -np.inf
    for threshold in candidates:
        predicted = cluster_rejected(embeddings, float(threshold))
        metrics = discovery_metrics(labels, predicted)
        value = metrics["ari"] if objective == "ari" else metrics["b3_f1"]
        rows.append({"threshold": float(threshold), "objective": value, "ARI": metrics["ari"], "B3_F1": metrics["b3_f1"]})
        if value > best_value or (value == best_value and threshold > best_threshold):
            best_threshold, best_value = float(threshold), float(value)
    return best_threshold, pd.DataFrame(rows)


def tune_thresholds(
    development_embeddings: np.ndarray, development_gt: pd.DataFrame, gallery: Gallery, config: EvaluationConfig
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    scores, _ranks, _ranked = score_gallery(development_embeddings, gallery, config.gallery_scoring, config.retrieval_chunk_size)
    top_scores = scores.max(axis=1)
    known, known_curve = tune_known_threshold(top_scores, development_gt["scenario"], config.threshold_objective)
    novel_mask = development_gt["scenario"].eq(SCENARIO_NOVEL).to_numpy()
    discovered, discovered_curve = tune_discovery_threshold(development_embeddings[novel_mask], development_gt.loc[novel_mask, "true_identity"], config.discovery_threshold_objective, config.seed)
    cluster, cluster_curve = tune_cluster_threshold(development_embeddings[novel_mask], development_gt.loc[novel_mask, "true_identity"], config.discovery_threshold_objective)
    curve = known_curve.assign(kind="known")
    curve = pd.concat([curve, discovered_curve.assign(kind="discovered"), cluster_curve.assign(kind="cluster")], ignore_index=True, sort=False)
    return {"known_similarity_threshold": known, "discovered_identity_threshold": discovered, "clustering_threshold": cluster}, curve, scores


def infer_batch(public_final: pd.DataFrame, embeddings: np.ndarray, gallery: Gallery, thresholds: Mapping[str, float], config: EvaluationConfig) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    scores, ranks, ranked_scores = score_gallery(embeddings, gallery, config.gallery_scoring, config.retrieval_chunk_size)
    best_indices = ranks[:, 0]; top_ids = gallery.identity_ids[best_indices]; top_scores = ranked_scores[:, 0]
    accepted = top_scores >= float(thresholds["known_similarity_threshold"])
    rejected_positions = np.flatnonzero(~accepted)
    clusters = cluster_rejected(embeddings[rejected_positions], float(thresholds["clustering_threshold"]))
    cluster_names = {cluster: f"NEW_{index:06d}" for index, cluster in enumerate(sorted(set(clusters.tolist())), start=1)}
    records = []
    for index, row in public_final.reset_index(drop=True).iterrows():
        if accepted[index]:
            prediction, status, discovered, discovered_score = top_ids[index], "known", "", float("nan")
        else:
            prediction, status = cluster_names[int(clusters[np.where(rejected_positions == index)[0][0]])], "novel"
            discovered, discovered_score = prediction, float("nan")
        records.append({
            "image_id": str(row.image_id), "arrival_index": int(row.arrival_index), "predicted_identity": prediction,
            "predicted_status": status, "top1_database_id": top_ids[index], "top1_similarity": float(top_scores[index]),
            "top2_database_id": gallery.identity_ids[ranks[index, 1]] if ranks.shape[1] > 1 else "",
            "top2_similarity": float(ranked_scores[index, 1]) if ranks.shape[1] > 1 else float("nan"),
            "novelty_score": float(1.0 - top_scores[index]), "matched_discovered_id": discovered,
            "discovered_similarity": discovered_score, "created_new_identity": bool(not accepted[index]),
        })
    return pd.DataFrame(records), retrieval_rows(public_final, gallery, ranks, ranked_scores), scores


def infer_online(public_stream: pd.DataFrame, embeddings_by_id: Mapping[str, np.ndarray], gallery: Gallery, thresholds: Mapping[str, float], config: EvaluationConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    stream = public_stream.sort_values("arrival_index").reset_index(drop=True)
    embeddings = np.vstack([embeddings_by_id[str(image_id)] for image_id in stream["image_id"]])
    scores, ranks, ranked_scores = score_gallery(embeddings, gallery, config.gallery_scoring, config.retrieval_chunk_size)
    top_ids, top_scores = gallery.identity_ids[ranks[:, 0]], ranked_scores[:, 0]
    novel_mask = top_scores < float(thresholds["known_similarity_threshold"])
    novel_labels, novel_scores, created = _online_assign(embeddings[novel_mask], float(thresholds["discovered_identity_threshold"]))
    iterator = iter(zip(novel_labels, novel_scores, created))
    rows = []
    for index, row in stream.iterrows():
        if not novel_mask[index]:
            predicted, status, discovered, dscore, created_flag = top_ids[index], "known", "", float("nan"), False
        else:
            predicted, dscore, created_flag = next(iterator); status, discovered = "novel", predicted
        rows.append({
            "image_id": str(row.image_id), "arrival_index": int(row.arrival_index), "predicted_identity": predicted,
            "predicted_status": status, "top1_database_id": top_ids[index], "top1_similarity": float(top_scores[index]),
            "top2_database_id": gallery.identity_ids[ranks[index, 1]] if ranks.shape[1] > 1 else "",
            "top2_similarity": float(ranked_scores[index, 1]) if ranks.shape[1] > 1 else float("nan"),
            "novelty_score": float(1.0 - top_scores[index]), "matched_discovered_id": discovered,
            "discovered_similarity": dscore, "created_new_identity": created_flag,
        })
    return pd.DataFrame(rows), retrieval_rows(stream, gallery, ranks, ranked_scores)


def assert_prediction_integrity(predictions: pd.DataFrame, final_public: pd.DataFrame, gallery: Gallery) -> None:
    required = {"image_id", "predicted_identity", "predicted_status", "top1_database_id", "top1_similarity", "novelty_score"}
    if not required <= set(predictions):
        raise ValueError(f"Prediction file missing columns {sorted(required - set(predictions))}")
    if predictions.image_id.duplicated().any() or set(predictions.image_id.astype(str)) != set(final_public.image_id.astype(str)):
        raise ValueError("Predictions must contain each final public image ID exactly once")
    if not np.isfinite(predictions[["top1_similarity", "novelty_score"]].to_numpy(dtype=float)).all():
        raise ValueError("Predictions contain non-finite similarity scores")
    known = predictions.loc[predictions.predicted_status.eq("known"), "predicted_identity"]
    if not set(known.astype(str)) <= set(gallery.identity_ids):
        raise ValueError("A known prediction references a database ID not present in gallery")


def _fpr95(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    eligible = fpr[tpr >= 0.95]
    return float(eligible.min()) if len(eligible) else float("nan")


def _known_query_metrics(merged: pd.DataFrame, embeddings_by_id: Mapping[str, np.ndarray], gallery: Gallery) -> dict[str, float]:
    known = merged[merged.scenario.eq(SCENARIO_KNOWN)].reset_index(drop=True)
    if known.empty:
        return {"Known_R1": float("nan"), "Known_R5": float("nan"), "Known_R10": float("nan"), "Known_mAP": float("nan")}
    embeddings = np.vstack([embeddings_by_id[str(image_id)] for image_id in known.image_id])
    metrics = compute_known_retrieval_metrics(embeddings, known.known_database_id.astype(str).to_numpy(), gallery.image_embeddings, gallery.image_database_ids, top_k=(1, 5, 10), cmc_ranks=(), chunk_size=1024)
    return {"Known_R1": metrics["known/top1"], "Known_R5": metrics["known/top5"], "Known_R10": metrics["known/top10"], "Known_mAP": metrics["known/mAP"]}


def _online_metrics(merged: pd.DataFrame) -> dict[str, float]:
    novel = merged[~merged.scenario.eq(SCENARIO_KNOWN)].sort_values("arrival_index")
    if novel.empty:
        return {key: float("nan") for key in ("first_observation_novel_detection_accuracy", "identity_creation_precision", "identity_creation_recall", "novel_reassociation_accuracy", "observations_to_stable_identity_mean", "observations_to_stable_identity_median")}
    first = novel.groupby("true_identity", sort=False).head(1)
    first_accuracy = float(first.predicted_status.eq("novel").mean())
    seen: set[str] = set(); creations = []; creation_correct = []; reassociation = []
    assigned_first: dict[str, str] = {}
    for row in novel.itertuples(index=False):
        true_id = str(row.true_identity)
        if bool(row.created_new_identity):
            creations.append(row); creation_correct.append(true_id not in seen)
        if true_id in seen and true_id in assigned_first:
            reassociation.append(row.predicted_identity == assigned_first[true_id])
        if true_id not in assigned_first and row.predicted_status == "novel":
            assigned_first[true_id] = row.predicted_identity
        seen.add(true_id)
    stable = []
    for true_id, group in novel.groupby("true_identity", sort=False):
        labels = group.predicted_identity.astype(str).tolist()
        value = float("nan")
        for index in range(len(labels)):
            if len(set(labels[index:])) == 1 and labels[index].startswith("NEW_"):
                value = float(index + 1); break
        stable.append(value)
    return {
        "first_observation_novel_detection_accuracy": first_accuracy,
        "identity_creation_precision": float(np.mean(creation_correct)) if creation_correct else float("nan"),
        "identity_creation_recall": float(len(assigned_first) / novel.true_identity.nunique()),
        "novel_reassociation_accuracy": float(np.mean(reassociation)) if reassociation else float("nan"),
        "observations_to_stable_identity_mean": float(np.nanmean(stable)) if np.isfinite(stable).any() else float("nan"),
        "observations_to_stable_identity_median": float(np.nanmedian(stable)) if np.isfinite(stable).any() else float("nan"),
    }


def evaluate_final(predictions: pd.DataFrame, final_ground_truth_path: Path, embeddings_by_id: Mapping[str, np.ndarray], gallery: Gallery, include_online: bool = False) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate finalized predictions; this is the only API that opens final labels."""
    ground_truth = pd.read_csv(final_ground_truth_path)
    merged = predictions.merge(ground_truth, on="image_id", validate="one_to_one", how="inner")
    if len(merged) != len(predictions):
        raise ValueError("Final ground truth does not align one-to-one with finalized predictions")
    known_mask = merged.scenario.eq(SCENARIO_KNOWN).to_numpy()
    novelty_true = ~known_mask
    novelty_score = merged.novelty_score.to_numpy(dtype=float)
    novelty = {
        "Novelty_AUROC": float(roc_auc_score(novelty_true, novelty_score)),
        "Novelty_AUPR": float(average_precision_score(novelty_true, novelty_score)),
        "Novelty_FPR95": _fpr95(novelty_true, novelty_score),
    }
    for label, mask in (("NovelKnownSpecies", merged.scenario.eq(SCENARIO_NOVEL).to_numpy()), ("DiscoveryOnly", merged.scenario.eq(SCENARIO_DISCOVERY).to_numpy())):
        pair_mask = known_mask | mask
        novelty[f"Known_vs_{label}_AUROC"] = float(roc_auc_score(mask[pair_mask], novelty_score[pair_mask])) if pair_mask.sum() and mask.any() else float("nan")
    predicted_novel = merged.predicted_status.eq("novel").to_numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(novelty_true, predicted_novel, average="binary", zero_division=0)
    known = merged[known_mask]
    known_e2e = known.predicted_identity.astype(str).eq(known.known_database_id.astype(str)) & known.predicted_status.eq("known")
    known_precision, known_recall, known_f1, _ = precision_recall_fscore_support(known_mask, ~predicted_novel, average="binary", zero_division=0)
    known_raw = _known_query_metrics(merged, embeddings_by_id, gallery)
    known_scores = known.top1_similarity.to_numpy(dtype=float)
    unknown_scores = merged.loc[~known_mask, "top1_similarity"].to_numpy(dtype=float)
    known_correct = known.top1_database_id.astype(str).eq(known.known_database_id.astype(str)).to_numpy()
    scenarios = {}
    for name, scenario in (("novel_known_species", SCENARIO_NOVEL), ("discovery_only", SCENARIO_DISCOVERY)):
        subset = merged[merged.scenario.eq(scenario)]
        scenarios[name] = discovery_metrics(subset.true_identity.astype(str), subset.predicted_identity.astype(str))
    overall = discovery_metrics(merged.true_identity.astype(str), merged.predicted_identity.astype(str))
    results = {
        "overall": overall,
        "retrieval": known_raw,
        "known_identity": {"known_end_to_end_accuracy": float(known_e2e.mean()) if len(known) else float("nan"), "known_precision": float(known_precision), "known_recall": float(known_recall), "known_f1": float(known_f1)},
        "novelty_detection": novelty | {"novel_precision": float(precision), "novel_recall": float(recall), "novel_f1": float(f1), "balanced_accuracy": float(balanced_accuracy_score(novelty_true, predicted_novel)), "specificity": float((~predicted_novel[known_mask]).mean()) if known_mask.any() else float("nan"), "sensitivity": float(predicted_novel[~known_mask].mean()) if (~known_mask).any() else float("nan"), "OSCR_AUC": float(compute_oscr_auc(known_scores, unknown_scores, known_correct))},
        "novel_known_species": scenarios["novel_known_species"], "discovery_only": scenarios["discovery_only"],
    }
    if include_online:
        results["online"] = _online_metrics(merged)
    return results, merged


def group_metrics(merged: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = []
    for group, subset in merged.groupby(group_column, dropna=False):
        values = discovery_metrics(subset.true_identity.astype(str), subset.predicted_identity.astype(str))
        known = subset[subset.scenario.eq(SCENARIO_KNOWN)]
        rows.append({group_column.title(): group, "Images": len(subset), "True_IDs": subset.true_identity.nunique(), "Predicted_IDs": subset.predicted_identity.nunique(), "ARI": values["ari"], "AMI": values["ami"], "NMI": values["nmi"], "B3_F1": values["b3_f1"], "Pairwise_F1": values["pairwise_f1"], "Known_E2E_Accuracy": float((known.predicted_identity.astype(str).eq(known.known_database_id.astype(str)) & known.predicted_status.eq("known")).mean()) if len(known) else float("nan")})
    return pd.DataFrame(rows)


def bootstrap_headlines(merged: pd.DataFrame, samples: int, seed: int) -> pd.DataFrame:
    if samples <= 0:
        return pd.DataFrame(columns=["metric", "estimate", "ci_lower", "ci_upper", "bootstrap_samples"])
    rng = np.random.default_rng(seed)
    identities = np.asarray(sorted(merged.true_identity.astype(str).unique()))
    estimates = {"Overall_ARI": clustering_metrics(merged.true_identity, merged.predicted_identity)["ari"]}
    known = merged[merged.scenario.eq(SCENARIO_KNOWN)]
    if len(known): estimates["Known_E2E_Accuracy"] = float((known.predicted_identity.astype(str).eq(known.known_database_id.astype(str)) & known.predicted_status.eq("known")).mean())
    values: dict[str, list[float]] = {key: [] for key in estimates}
    for _ in range(samples):
        chosen = rng.choice(identities, size=len(identities), replace=True)
        sample = pd.concat([merged[merged.true_identity.astype(str).eq(identity)] for identity in chosen], ignore_index=True)
        values["Overall_ARI"].append(clustering_metrics(sample.true_identity, sample.predicted_identity)["ari"])
        if "Known_E2E_Accuracy" in values:
            sample_known = sample[sample.scenario.eq(SCENARIO_KNOWN)]
            values["Known_E2E_Accuracy"].append(float((sample_known.predicted_identity.astype(str).eq(sample_known.known_database_id.astype(str)) & sample_known.predicted_status.eq("known")).mean()) if len(sample_known) else float("nan"))
    return pd.DataFrame([{"metric": key, "estimate": estimate, "ci_lower": float(np.nanquantile(values[key], 0.025)), "ci_upper": float(np.nanquantile(values[key], 0.975)), "bootstrap_samples": samples} for key, estimate in estimates.items()])


def save_plots(output_dir: Path, merged: pd.DataFrame, known_curve: pd.DataFrame, retrieval: pd.DataFrame) -> None:
    plots, curves = output_dir / "plots", output_dir / "curves"
    plots.mkdir(parents=True, exist_ok=True); curves.mkdir(parents=True, exist_ok=True)
    known = merged.scenario.eq(SCENARIO_KNOWN).to_numpy(); novel = ~known
    score = merged.novelty_score.to_numpy(dtype=float)
    fpr, tpr, _ = roc_curve(novel, score); precision, recall, _ = precision_recall_curve(novel, score)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(curves / "roc_curve.csv", index=False)
    pd.DataFrame({"precision": precision, "recall": recall}).to_csv(curves / "pr_curve.csv", index=False)
    for x, y, xlabel, ylabel, stem in ((fpr, tpr, "False positive rate", "True positive rate", "novelty_roc"), (recall, precision, "Recall", "Precision", "novelty_pr")):
        fig, ax = plt.subplots(); ax.plot(x, y); ax.set(xlabel=xlabel, ylabel=ylabel, title=stem.replace("_", " ")); fig.tight_layout(); fig.savefig(plots / f"{stem}.png", dpi=180); fig.savefig(plots / f"{stem}.pdf"); plt.close(fig)
    threshold = known_curve[known_curve.kind.eq("known")]
    threshold.to_csv(curves / "development_threshold_curve.csv", index=False)
    fig, ax = plt.subplots(); ax.plot(threshold.threshold, threshold.objective); ax.set(xlabel="Known similarity threshold", ylabel="Selection objective"); fig.tight_layout(); fig.savefig(plots / "development_threshold_selection.png", dpi=180); fig.savefig(plots / "development_threshold_selection.pdf"); plt.close(fig)
    fig, ax = plt.subplots()
    for label, group in merged.groupby("scenario"):
        ax.hist(group.top1_similarity, bins=40, alpha=.45, density=True, label=label)
    ax.legend(); ax.set(xlabel="Top-1 database similarity", ylabel="Density"); fig.tight_layout(); fig.savefig(plots / "similarity_distribution.png", dpi=180); fig.savefig(plots / "similarity_distribution.pdf"); plt.close(fig)
    known_truth = merged.loc[merged.scenario.eq(SCENARIO_KNOWN), ["image_id", "known_database_id"]]
    ranked = retrieval.merge(known_truth, on="image_id", how="inner")
    cmc_rows = []
    for rank in range(1, 11):
        hits = ranked[ranked["rank"] <= rank].groupby("image_id").apply(
            lambda group: bool((group.database_id.astype(str) == group.known_database_id.astype(str)).any()),
            include_groups=False,
        )
        cmc_rows.append({"rank": rank, "CMC": float(hits.mean()) if len(hits) else float("nan")})
    cmc = pd.DataFrame(cmc_rows); cmc.to_csv(curves / "cmc_curve.csv", index=False)
    fig, ax = plt.subplots(); ax.plot(cmc["rank"], cmc["CMC"], marker="o"); ax.set(xlabel="Rank", ylabel="CMC"); fig.tight_layout(); fig.savefig(plots / "cmc_curve.png", dpi=180); fig.savefig(plots / "cmc_curve.pdf"); plt.close(fig)
    known_scores = merged.loc[known, "top1_similarity"].to_numpy(dtype=float)
    unknown_scores = merged.loc[novel, "top1_similarity"].to_numpy(dtype=float)
    known_correct = (merged.loc[known, "top1_database_id"].astype(str).to_numpy() == merged.loc[known, "known_database_id"].astype(str).to_numpy())
    curve_rows = []
    for threshold_value in _threshold_candidates(np.concatenate([known_scores, unknown_scores]), limit=200):
        curve_rows.append({"threshold": float(threshold_value), "FPR": float((unknown_scores >= threshold_value).mean()), "CCR": float(((known_scores >= threshold_value) & known_correct).mean())})
    oscr = pd.DataFrame(curve_rows).sort_values("FPR"); oscr.to_csv(curves / "oscr_curve.csv", index=False)
    fig, ax = plt.subplots(); ax.plot(oscr["FPR"], oscr["CCR"]); ax.set(xlabel="False positive rate", ylabel="Correct classification rate"); fig.tight_layout(); fig.savefig(plots / "oscr_curve.png", dpi=180); fig.savefig(plots / "oscr_curve.pdf"); plt.close(fig)


def run_evaluation(config: EvaluationConfig) -> dict[str, Any]:
    """Run development tuning, sealed final inference, then final evaluation."""
    if config.evaluation_mode not in {"batch", "online", "both"}:
        raise ValueError("evaluation_mode must be batch, online, or both")
    seed_everything(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    frames = validate_benchmark_directory(config.benchmark_dir)
    model_cfg = load_config(config_path=config.model_config)
    if config.image_size is not None:
        model_cfg = copy.deepcopy(model_cfg); model_cfg.setdefault("data", {})["image_size"] = list(config.image_size)
    device = torch.device(config.device if config.device != "cuda" or torch.cuda.is_available() else "cpu")
    adapter = ApexIDModelAdapter(model_cfg, device, config.amp)
    model_info = adapter.load_weights(config.weights)
    transform = build_eval_transforms(config=model_cfg)
    database = frames["database"].copy(); database.insert(0, "image_id", [f"DBIMG_{index:08d}" for index in range(len(database))])
    database_embeddings, _ = extract_with_cache("database", database, adapter, config, transform)
    gallery = build_gallery(database, database_embeddings)
    development_embeddings, development = extract_with_cache("development", frames["development"], adapter, config, transform)
    development_gt = pd.read_csv(config.benchmark_dir / "development_ground_truth.csv").set_index("image_id").loc[development.image_id].reset_index()
    thresholds, threshold_curve, _ = tune_thresholds(development_embeddings, development_gt, gallery, config)
    write_json(config.output_dir / "thresholds.json", thresholds)
    final_embeddings, final_public = extract_with_cache("final_test", frames["final_test"], adapter, config, transform)
    embedding_by_id = {str(image_id): embedding for image_id, embedding in zip(final_public.image_id, final_embeddings)}
    run_config = asdict(config) | {"model": model_info, "benchmark_config": json.loads((config.benchmark_dir / "config.json").read_text()) if (config.benchmark_dir / "config.json").is_file() else {}, "thresholds": thresholds, "torch": torch.__version__, "cuda": torch.version.cuda}
    write_json(config.output_dir / "run_config.json", run_config)
    results: dict[str, Any] = {"model": model_info, "thresholds": thresholds}
    if config.evaluation_mode in {"batch", "both"}:
        batch_dir = config.output_dir / "batch"; batch_dir.mkdir(exist_ok=True)
        predictions, retrieval, final_scores = infer_batch(final_public, final_embeddings, gallery, thresholds, config)
        assert_prediction_integrity(predictions, final_public, gallery)
        predictions.to_csv(batch_dir / "predictions.csv", index=False); retrieval.to_csv(batch_dir / "retrieval_predictions.csv", index=False)
        # The next call is intentionally after prediction persistence.
        metrics, merged = evaluate_final(predictions, config.benchmark_dir / "final_test_ground_truth.csv", embedding_by_id, gallery)
        dataset_metrics = group_metrics(merged, "dataset")
        species_metrics = group_metrics(merged, "species")
        metrics["overall"]["macro_dataset_ari"] = float(dataset_metrics["ARI"].mean())
        metrics["overall"]["macro_dataset_nmi"] = float(dataset_metrics["NMI"].mean())
        metrics["overall"]["macro_dataset_b3_f1"] = float(dataset_metrics["B3_F1"].mean())
        results["batch"] = metrics
        scenario_metrics = pd.DataFrame([{"scenario": name, **values} for name, values in (("overall", metrics["overall"]), (SCENARIO_NOVEL, metrics["novel_known_species"]), (SCENARIO_DISCOVERY, metrics["discovery_only"]))])
        ci = bootstrap_headlines(merged, config.bootstrap_samples, config.seed)
        for target in (batch_dir, config.output_dir):
            predictions.to_csv(target / "predictions.csv", index=False); retrieval.to_csv(target / "retrieval_predictions.csv", index=False)
            dataset_metrics.to_csv(target / "dataset_metrics.csv", index=False); species_metrics.to_csv(target / "species_metrics.csv", index=False)
            scenario_metrics.to_csv(target / "scenario_metrics.csv", index=False); ci.to_csv(target / "paper_main_results_with_ci.csv", index=False)
        paper = {"Model": config.weights.stem, "Protocol": config.benchmark_dir.name, **metrics["retrieval"], "Known_E2E_Accuracy": metrics["known_identity"]["known_end_to_end_accuracy"], "Novelty_AUROC": metrics["novelty_detection"]["Novelty_AUROC"], "Novelty_AUPR": metrics["novelty_detection"]["Novelty_AUPR"], "Novelty_FPR95": metrics["novelty_detection"]["Novelty_FPR95"], "OSCR_AUC": metrics["novelty_detection"]["OSCR_AUC"], "Novel_KnownSpecies_ARI": metrics["novel_known_species"]["ari"], "DiscoveryOnly_ARI": metrics["discovery_only"]["ari"], "Overall_ARI": metrics["overall"]["ari"], "B3_F1": metrics["overall"]["b3_f1"]}
        paper["Macro_Dataset_ARI"] = metrics["overall"]["macro_dataset_ari"]
        pd.DataFrame([paper]).to_csv(batch_dir / "paper_main_results.csv", index=False)
        pd.DataFrame([paper]).to_csv(config.output_dir / "paper_main_results.csv", index=False)
        known_errors = merged[merged.scenario.eq(SCENARIO_KNOWN) & ~((merged.predicted_status.eq("known")) & merged.predicted_identity.astype(str).eq(merged.known_database_id.astype(str)))].copy()
        known_errors["error_type"] = np.where(known_errors.predicted_status.eq("novel"), "false_novel_rejection", "wrong_known_identity")
        known_errors[["image_id", "true_identity", "predicted_identity", "top1_similarity", "predicted_status", "error_type"]].to_csv(config.output_dir / "known_identity_errors.csv", index=False)
        novel_errors = merged[~merged.scenario.eq(SCENARIO_KNOWN)].copy()
        novel_errors = novel_errors[novel_errors.predicted_status.eq("known") | novel_errors.predicted_identity.duplicated(keep=False)]
        novel_errors["error_type"] = np.where(novel_errors.predicted_status.eq("known"), "false_known_acceptance", "cluster_fragmentation_or_merge")
        novel_errors[["image_id", "true_identity", "predicted_identity", "top1_database_id", "top1_similarity", "discovered_similarity", "error_type"]].to_csv(config.output_dir / "novel_identity_errors.csv", index=False)
        id_to_index = {identity: index for index, identity in enumerate(gallery.identity_ids)}
        hard = merged[merged.scenario.eq(SCENARIO_KNOWN)].copy()
        hard["true_similarity"] = [float(final_scores[final_public.image_id.astype(str).eq(str(row.image_id)).to_numpy().argmax(), id_to_index[str(row.known_database_id)]]) for row in hard.itertuples(index=False)]
        hard["margin"] = hard.true_similarity - hard.top1_similarity
        hard.rename(columns={"known_database_id": "true_database_id", "top1_database_id": "top1_id"})[["image_id", "true_database_id", "top1_id", "top1_similarity", "true_similarity", "margin"]].sort_values("margin").head(200).to_csv(config.output_dir / "hard_known_queries.csv", index=False)
        write_json(config.output_dir / "embedding_diagnostics.json", {"embedding_dim": int(final_embeddings.shape[1]), "database_norm_mean": float(np.linalg.norm(database_embeddings, axis=1).mean()), "development_norm_mean": float(np.linalg.norm(development_embeddings, axis=1).mean()), "final_norm_mean": float(np.linalg.norm(final_embeddings, axis=1).mean())})
        scenario_metrics.to_csv(config.output_dir / "clustering_diagnostics.csv", index=False)
        save_plots(config.output_dir, merged, threshold_curve, retrieval)
    if config.evaluation_mode in {"online", "both"}:
        online_dir = config.output_dir / "online"; online_dir.mkdir(exist_ok=True)
        stream_files = sorted(path for path in (config.benchmark_dir / "streams").glob("stream_seed_*.csv") if not path.name.endswith("_no_species.csv"))
        if not stream_files: stream_files = []
        online_rows = []
        for stream_path in stream_files:
            stream = pd.read_csv(stream_path)
            predictions, retrieval = infer_online(stream, embedding_by_id, gallery, thresholds, config)
            assert_prediction_integrity(predictions, final_public, gallery)
            predictions.to_csv(online_dir / f"{stream_path.stem}_predictions.csv", index=False); retrieval.to_csv(online_dir / f"{stream_path.stem}_retrieval_predictions.csv", index=False)
            metrics, _merged = evaluate_final(predictions, config.benchmark_dir / "final_test_ground_truth.csv", embedding_by_id, gallery, include_online=True)
            online_rows.append({"stream_seed": int(stream_path.stem.removeprefix("stream_seed_")), "overall_ari": metrics["overall"]["ari"], "known_end_to_end_accuracy": metrics["known_identity"]["known_end_to_end_accuracy"], "novelty_f1": metrics["novelty_detection"]["novel_f1"], "novel_known_species_ari": metrics["novel_known_species"]["ari"], "discovery_only_ari": metrics["discovery_only"]["ari"], "novel_reassociation_accuracy": metrics["online"]["novel_reassociation_accuracy"], "predicted_novel_clusters": _merged.loc[_merged.predicted_status.eq("novel"), "predicted_identity"].nunique()})
        per_seed = pd.DataFrame(online_rows); per_seed.to_csv(online_dir / "online_results_per_seed.csv", index=False)
        summary = per_seed.drop(columns="stream_seed").agg(["mean", "std", "min", "max"]).T.reset_index(names="metric") if len(per_seed) else pd.DataFrame(columns=["metric", "mean", "std", "min", "max"])
        summary.to_csv(online_dir / "online_results_summary.csv", index=False); results["online"] = {"per_seed": online_rows, "summary": summary.to_dict(orient="records")}
    write_json(config.output_dir / "evaluation_results.json", results)
    return results


__all__ = [
    "ApexIDModelAdapter", "EvaluationConfig", "Gallery", "assert_prediction_integrity", "build_gallery", "cluster_rejected",
    "evaluate_final", "extract_with_cache", "infer_batch", "infer_online", "l2_normalize", "run_evaluation", "score_gallery",
    "tune_cluster_threshold", "tune_discovery_threshold", "tune_known_threshold", "validate_benchmark_directory",
]
