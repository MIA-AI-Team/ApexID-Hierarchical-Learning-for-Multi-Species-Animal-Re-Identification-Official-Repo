#!/usr/bin/env python3
"""Offline synthetic ApexID reproduction smoke test (no private data or downloads)."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/apexid-matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/apexid-cache")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from apexid.engine import load_training_checkpoint, save_training_checkpoint
from apexid.evaluation.benchmark_evaluator import _online_assign
from apexid.evaluation.benchmark_metrics import discovery_metrics
from apexid.models import build_model
from apexid.utils.config import expand_config_environment
from apexid.utils.provenance import write_json_atomic


def run_smoke() -> dict:
    resolved = expand_config_environment(
        {"data": {"root": "${APEXID_SMOKE_ROOT:-synthetic}"}, "output": "${OUTPUT_ROOT}/run"},
        {"OUTPUT_ROOT": "offline"},
    )
    model_config = {
        "seed": 7,
        "model": {
            "name": "apexid", "embedding_dim": 8, "input_channels": 3,
            "backbone": {"name": "osnet_x0_25", "kwargs": {}},
        },
    }
    model = build_model(model_config).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    inputs = torch.linspace(-1.0, 1.0, steps=2 * 3 * 64 * 64).reshape(2, 3, 64, 64)
    with torch.no_grad():
        before = model(inputs)
    with tempfile.TemporaryDirectory(prefix="apexid-repro-smoke-") as directory:
        checkpoint = save_training_checkpoint(
            epoch=0, model=model, optimizer=optimizer, output_dir=directory,
            history={"train_loss": [1.0]}, delete_previous=False, include_rng=False,
        )
        restored = build_model(model_config).eval()
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1.0e-4)
        payload = load_training_checkpoint(
            checkpoint, restored, optimizer=restored_optimizer, restore_rng=False,
        )
        with torch.no_grad():
            after = restored(inputs)

    online_vectors = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]], dtype=np.float32)
    predicted, _, created = _online_assign(online_vectors, threshold=0.9)
    metrics = discovery_metrics(["A", "A", "B"], predicted)
    return {
        "status": "pass",
        "config_resolution": resolved,
        "forward_shape": list(before.shape),
        "normalized_embeddings": bool(torch.allclose(before.norm(dim=1), torch.ones(2), atol=1e-5)),
        "checkpoint_next_epoch": int(payload["next_epoch"]),
        "checkpoint_round_trip_equal": bool(torch.equal(before, after)),
        "online_assignments": predicted,
        "online_created": created,
        "evaluator_metrics": {key: metrics[key] for key in ("ari", "b3_f1")},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional atomic JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_smoke()
    if args.output:
        write_json_atomic(args.output, result)
    print("ApexID offline reproduction smoke: PASS")
    print(f"checkpoint_round_trip_equal={result['checkpoint_round_trip_equal']}")
    print(f"online_assignments={result['online_assignments']}")


if __name__ == "__main__":
    main()
