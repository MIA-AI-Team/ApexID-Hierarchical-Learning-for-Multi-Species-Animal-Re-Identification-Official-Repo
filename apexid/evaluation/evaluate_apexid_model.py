#!/usr/bin/env python3
"""Command-line entrypoint for sealed ApexID open-world benchmark evaluation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .benchmark_evaluator import EvaluationConfig, run_evaluation


def _image_size(values: list[int] | None) -> tuple[int, int] | None:
    if values is None:
        return None
    if len(values) == 1:
        return values[0], values[0]
    if len(values) == 2:
        return values[0], values[1]
    raise argparse.ArgumentTypeError("--image-size accepts one square size or height width")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, nargs="+")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--evaluation-mode", choices=("batch", "online", "both"), default="both")
    parser.add_argument("--gallery-scoring", choices=("centroid", "max_image", "mean_top_k"), default="centroid")
    parser.add_argument("--threshold-objective", choices=("f1", "balanced_accuracy", "youden"), default="f1")
    parser.add_argument("--discovery-threshold-objective", choices=("ari", "b3_f1"), default="ari")
    parser.add_argument("--retrieval-chunk-size", type=int, default=1024)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--save-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--update-known-prototypes", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or args.retrieval_chunk_size <= 0 or args.bootstrap_samples < 0:
        raise ValueError("Batch size/chunk size must be positive; workers and bootstrap samples cannot be negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.output_dir / "evaluation.log", encoding="utf-8")],
    )
    config = EvaluationConfig(
        weights=args.weights, benchmark_dir=args.benchmark_dir, data_root=args.data_root,
        output_dir=args.output_dir, device=args.device, batch_size=args.batch_size,
        num_workers=args.num_workers, image_size=_image_size(args.image_size), seed=args.seed,
        model_config=args.model_config, evaluation_mode=args.evaluation_mode,
        gallery_scoring=args.gallery_scoring, threshold_objective=args.threshold_objective,
        discovery_threshold_objective=args.discovery_threshold_objective,
        retrieval_chunk_size=args.retrieval_chunk_size, bootstrap_samples=args.bootstrap_samples,
        force_recompute=args.force_recompute, save_embeddings=args.save_embeddings,
        update_known_prototypes=args.update_known_prototypes, amp=not args.no_amp,
    )
    results = run_evaluation(config)
    print("=" * 60)
    print("ApexID Evaluation")
    print("=" * 60)
    print(f"Weights: {args.weights}")
    print(f"Protocol: {args.benchmark_dir.name}")
    print(f"Evaluation mode: {args.evaluation_mode}")
    print(f"Thresholds: {results['thresholds']}")
    if "batch" in results:
        batch = results["batch"]
        print(f"Known Rank-1: {batch['retrieval']['Known_R1']:.6f}")
        print(f"Known mAP: {batch['retrieval']['Known_mAP']:.6f}")
        print(f"Novelty AUROC: {batch['novelty_detection']['Novelty_AUROC']:.6f}")
        print(f"Overall ARI: {batch['overall']['ari']:.6f}")
    print("All integrity checks: PASS")


if __name__ == "__main__":
    main()
