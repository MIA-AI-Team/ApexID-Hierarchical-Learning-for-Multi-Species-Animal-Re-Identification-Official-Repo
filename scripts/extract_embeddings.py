"""Extract one paper descriptor on the frozen external closed-set catalogues."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apexid.data import build_eval_transforms
from apexid.evaluation.embeddings import ApexIDModelAdapter, BenchmarkImageDataset, read_path_map, remap_path
from apexid.utils.config import load_config
from apexid.utils.provenance import capture_environment, sha256_file, write_json_atomic


DATASETS = ("bristol_gorillas", "elephants", "gcn_id", "jaguar_antonio")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True, help="Stage 00 prepared bundle")
    parser.add_argument("--output", type=Path, required=True, help="Stage 01 embedding directory")
    parser.add_argument("--config", type=Path, required=True, help="Checkpoint sidecar or reference model YAML")
    parser.add_argument("--weights", type=Path, help="Local checkpoint, omitted for an original pretrained reference")
    parser.add_argument("--name", required=True, help="Descriptor name used in subsequent evaluation")
    parser.add_argument("--image-root", type=Path, default=Path.cwd(), help="Root for relative catalogue paths")
    parser.add_argument("--path-map", type=Path, help="JSON object mapping old path prefixes to mounted roots")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true", help="Extract in float32 on CUDA")
    return parser.parse_args()


def save_embeddings(destination, **arrays):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".npz", delete=False) as stream:
        pending = Path(stream.name)
    try:
        np.savez_compressed(pending, **arrays)
        os.replace(pending, destination)
    finally:
        pending.unlink(missing_ok=True)


def main():
    args = parse_args()
    if Path(args.name).name != args.name or args.name in {"", ".", ".."}:
        raise ValueError("--name must be a single directory name")
    config = load_config(config_path=args.config)
    device = torch.device(args.device)
    if args.weights:
        adapter = ApexIDModelAdapter.from_checkpoint(config, args.weights, device, amp=not args.no_amp)
    else:
        adapter = ApexIDModelAdapter(config, device, amp=not args.no_amp)
        if getattr(adapter.model.backbone, "pretrained_provenance", None) is None:
            raise ValueError("Supply --weights for a trained model or select an original pretrained reference config")
    transform = build_eval_transforms(config=config)
    path_map = read_path_map(args.path_map)
    prepared_manifest = args.prepared / "preparation_manifest.json"
    prepared_hash = sha256_file(prepared_manifest)
    prepared = json.loads(prepared_manifest.read_text(encoding="utf-8"))
    if prepared.get("status") != "passed":
        raise ValueError("Stage 00 must be complete")

    artifacts = {}
    for dataset in DATASETS:
        catalogue = args.prepared / "manifests" / f"{dataset}_canonical_manifest.csv"
        catalogue_hash = sha256_file(catalogue)
        declared_file_hash = prepared.get("dataset_manifest_file_hashes", {}).get(dataset)
        if declared_file_hash and catalogue_hash != declared_file_hash:
            raise ValueError(f"Frozen catalogue checksum changed for {dataset}")
        frame = pd.read_csv(catalogue, dtype=str, keep_default_na=False)
        if frame.empty or frame["image_id"].eq("").any() or frame["image_id"].duplicated().any():
            raise ValueError(f"Invalid image IDs in {catalogue}")
        images = frame.assign(path=frame["path"].map(lambda path: remap_path(path, path_map)))
        loader = DataLoader(
            BenchmarkImageDataset(images, args.image_root, transform),
            batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda",
        )
        embeddings, image_ids, paths = adapter.extract(loader)
        if image_ids != frame["image_id"].tolist() or paths != images["path"].tolist():
            raise RuntimeError(f"Extraction order differs from the frozen catalogue for {dataset}")
        metadata = {
            "model": args.name,
            "dataset": dataset,
            "number_images": len(frame),
            "embedding_dimension": embeddings.shape[1],
            "l2_normalized": True,
            "dataset_manifest_sha256": prepared.get("dataset_manifest_hashes", {}).get(dataset, catalogue_hash),
            "dataset_manifest_file_sha256": catalogue_hash,
            "image_order_sha256": hashlib.sha256("\n".join(image_ids).encode("utf-8")).hexdigest(),
            "checkpoint_sha256": adapter.checkpoint_metadata.get("weights_sha256"),
            "pretrained_artifact": getattr(adapter.model.backbone, "pretrained_provenance", None),
            "config_sha256": sha256_file(args.config),
            "preprocessing": {
                "image_size": config["data"]["image_size"],
                "mean": config["data"]["transforms"]["mean"],
                "std": config["data"]["transforms"]["std"],
            },
        }
        destination = args.output / "embeddings" / args.name / dataset / "embeddings.npz"
        save_embeddings(
            destination, embeddings=embeddings, image_ids=np.asarray(image_ids, dtype=str),
            identities=frame["identity"].to_numpy(dtype=str), paths=frame["path"].to_numpy(dtype=str),
            dataset=np.asarray([dataset]), metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        artifacts[dataset] = {"path": destination.relative_to(args.output).as_posix(), "sha256": sha256_file(destination)}
        print(f"{dataset}: {len(embeddings)} embeddings -> {destination}")

    model_manifests = args.output / "manifests"
    write_json_atomic(model_manifests / f"{args.name}.json", {
        "status": "passed",
        "model": args.name,
        "prepared_manifest_sha256": prepared_hash,
        "artifacts": artifacts,
        "environment": capture_environment(),
    })
    completed = sorted(
        metadata["model"] for path in model_manifests.glob("*.json")
        if (metadata := json.loads(path.read_text(encoding="utf-8"))).get("status") == "passed"
        and metadata.get("prepared_manifest_sha256") == prepared_hash
    )
    write_json_atomic(args.output / "stage01_manifest.json", {
        "status": "passed", "models_completed": completed, "prepared_manifest_sha256": prepared_hash,
    })


if __name__ == "__main__":
    main()
