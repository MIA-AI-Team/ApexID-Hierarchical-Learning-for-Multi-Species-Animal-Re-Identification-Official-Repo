from __future__ import annotations

import copy
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from apexid.engine.validator import _compute_loss_and_logits, _unpack_batch


@dataclass
class LRRangeTestResult:
    """Serializable LR range test output."""

    lrs: list[float]
    losses: list[float]
    raw_losses: list[float]
    suggested_base_lr: float
    suggested_max_lr: float
    stop_reason: str
    metadata: dict[str, Any]

    def to_dict(self):
        """Return a JSON-serializable dictionary."""
        return asdict(self)


def _known_mask(labels, ignore_index):
    if ignore_index is None:
        return torch.ones_like(labels, dtype=torch.bool)
    return labels != ignore_index


def _set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _suggest_lr_bounds(lrs, losses, start_lr, base_lr_divisor):
    if not lrs or not losses:
        raise RuntimeError("LR range test did not complete any finite optimizer updates.")

    best_idx = min(range(len(losses)), key=lambda idx: losses[idx])
    suggested_idx = min(max(best_idx, 1), len(lrs) - 1)
    suggested_max_lr = float(lrs[suggested_idx])
    suggested_base_lr = max(float(start_lr), suggested_max_lr / float(base_lr_divisor))
    if suggested_base_lr >= suggested_max_lr:
        suggested_base_lr = suggested_max_lr / float(base_lr_divisor)
    return suggested_base_lr, suggested_max_lr


def run_lr_range_test(
    backbone,
    head,
    criterion,
    train_loader,
    optimizer,
    device,
    start_lr=1e-7,
    end_lr=1e-2,
    num_iter=100,
    smooth_beta=0.98,
    stop_factor=4.0,
    base_lr_divisor=4.0,
    classification_ignore_index=-100,
    gradient_clip_norm=None,
):
    """Run an LR range test and restore model/optimizer state afterward."""
    start_lr = float(start_lr)
    end_lr = float(end_lr)
    num_iter = int(num_iter)
    smooth_beta = float(smooth_beta)
    stop_factor = float(stop_factor)
    base_lr_divisor = float(base_lr_divisor)

    if start_lr <= 0.0 or end_lr <= 0.0:
        raise ValueError("start_lr and end_lr must be positive.")
    if end_lr <= start_lr:
        raise ValueError("end_lr must be larger than start_lr.")
    if num_iter <= 0:
        raise ValueError("num_iter must be positive.")
    if not 0.0 <= smooth_beta < 1.0:
        raise ValueError("smooth_beta must be in the range [0, 1).")
    if stop_factor <= 1.0:
        raise ValueError("stop_factor must be > 1.")
    if base_lr_divisor <= 1.0:
        raise ValueError("base_lr_divisor must be > 1.")

    device = torch.device(device)
    backbone_state = copy.deepcopy(backbone.state_dict())
    head_state = copy.deepcopy(head.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    backbone_was_training = backbone.training
    head_was_training = head.training

    _set_optimizer_lr(optimizer, start_lr)
    lr_mult = (end_lr / start_lr) ** (1.0 / max(1, num_iter - 1))

    lrs = []
    losses = []
    raw_losses = []
    avg_loss = 0.0
    best_loss = math.inf
    iteration = 0
    stop_reason = "completed"

    backbone.train()
    head.train()

    try:
        while iteration < num_iter:
            made_progress = False
            for batch in train_loader:
                if iteration >= num_iter:
                    break

                made_progress = True
                images, labels, _ = _unpack_batch(batch)
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).long()

                if not _known_mask(labels, classification_ignore_index).all():
                    raise ValueError(
                        "Unknown/ignored labels appeared in the LR range test loader. "
                        "Training must contain only known IDs."
                    )

                optimizer.zero_grad(set_to_none=True)
                embeddings = backbone(images)
                loss, _ = _compute_loss_and_logits(head, criterion, embeddings, labels)

                if not torch.isfinite(loss):
                    stop_reason = "non_finite_loss"
                    return _build_result(
                        lrs,
                        losses,
                        raw_losses,
                        stop_reason,
                        start_lr,
                        end_lr,
                        num_iter,
                        smooth_beta,
                        stop_factor,
                        base_lr_divisor,
                    )

                loss.backward()
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        list(backbone.parameters()) + list(head.parameters()),
                        float(gradient_clip_norm),
                        foreach=True,
                    )
                optimizer.step()

                loss_value = float(loss.detach().cpu())
                avg_loss = smooth_beta * avg_loss + (1.0 - smooth_beta) * loss_value
                smoothed_loss = avg_loss / (1.0 - smooth_beta ** (iteration + 1))
                current_lr = float(optimizer.param_groups[0]["lr"])

                raw_losses.append(loss_value)
                losses.append(float(smoothed_loss))
                lrs.append(current_lr)

                if smoothed_loss < best_loss:
                    best_loss = smoothed_loss
                elif smoothed_loss > stop_factor * best_loss:
                    stop_reason = "diverged"
                    return _build_result(
                        lrs,
                        losses,
                        raw_losses,
                        stop_reason,
                        start_lr,
                        end_lr,
                        num_iter,
                        smooth_beta,
                        stop_factor,
                        base_lr_divisor,
                    )

                iteration += 1
                _set_optimizer_lr(optimizer, min(end_lr, current_lr * lr_mult))

            if not made_progress:
                stop_reason = "empty_loader"
                break

        return _build_result(
            lrs,
            losses,
            raw_losses,
            stop_reason,
            start_lr,
            end_lr,
            num_iter,
            smooth_beta,
            stop_factor,
            base_lr_divisor,
        )
    finally:
        backbone.load_state_dict(backbone_state)
        head.load_state_dict(head_state)
        optimizer.load_state_dict(optimizer_state)
        backbone.train(backbone_was_training)
        head.train(head_was_training)


def _build_result(
    lrs,
    losses,
    raw_losses,
    stop_reason,
    start_lr,
    end_lr,
    num_iter,
    smooth_beta,
    stop_factor,
    base_lr_divisor,
):
    suggested_base_lr, suggested_max_lr = _suggest_lr_bounds(
        lrs,
        losses,
        start_lr,
        base_lr_divisor,
    )
    best_idx = min(range(len(losses)), key=lambda idx: losses[idx])
    metadata = {
        "start_lr": float(start_lr),
        "end_lr": float(end_lr),
        "num_iter": int(num_iter),
        "completed_iter": int(len(lrs)),
        "smooth_beta": float(smooth_beta),
        "stop_factor": float(stop_factor),
        "base_lr_divisor": float(base_lr_divisor),
        "best_lr": float(lrs[best_idx]),
        "best_loss": float(losses[best_idx]),
    }
    return LRRangeTestResult(
        lrs=[float(lr) for lr in lrs],
        losses=[float(loss) for loss in losses],
        raw_losses=[float(loss) for loss in raw_losses],
        suggested_base_lr=float(suggested_base_lr),
        suggested_max_lr=float(suggested_max_lr),
        stop_reason=str(stop_reason),
        metadata=metadata,
    )


def write_lr_range_test_artifacts(result, output_dir):
    """Write LR range test JSON and CSV artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "lr_range_test.json"
    csv_path = output_dir / "lr_range_test.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, sort_keys=True)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["iteration", "lr", "loss", "raw_loss"])
        writer.writeheader()
        for idx, (lr, loss, raw_loss) in enumerate(zip(result.lrs, result.losses, result.raw_losses)):
            writer.writerow(
                {
                    "iteration": idx,
                    "lr": lr,
                    "loss": loss,
                    "raw_loss": raw_loss,
                }
            )

    return {"json": json_path, "csv": csv_path}


__all__ = [
    "LRRangeTestResult",
    "run_lr_range_test",
    "write_lr_range_test_artifacts",
]
