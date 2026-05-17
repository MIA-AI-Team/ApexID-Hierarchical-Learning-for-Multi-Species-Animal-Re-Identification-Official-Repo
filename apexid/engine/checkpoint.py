import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch


def _unwrap_module(module):
    """Return the wrapped module when using DataParallel or DDP."""
    while hasattr(module, "module"):
        module = module.module
    return module


def _state_dict_to_cpu(state_dict):
    """Move tensor state-dict values to CPU before checkpoint serialization."""
    return OrderedDict(
        (key, value.detach().cpu() if torch.is_tensor(value) else value)
        for key, value in state_dict.items()
    )


def _format_checkpoint_name(filename, epoch):
    """Format checkpoint filename templates using zero-based and one-based epochs."""
    epoch = int(epoch)
    return filename.format(
        epoch=epoch,
        epoch_index=epoch,
        epoch_number=epoch + 1,
    )


def _torch_load(path, map_location):
    """Load a checkpoint with compatibility across PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def collect_rng_state():
    """Collect Python, NumPy, PyTorch CPU, and PyTorch CUDA RNG state.

    Args:
        None

    Returns:
        dict: RNG state dictionary that can be saved inside a checkpoint and later
        passed to ``restore_rng_state``.
    """
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": None,
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(rng_state):
    """Restore RNG state saved by ``collect_rng_state``.

    Args:
        rng_state (dict): RNG dictionary stored in a training checkpoint.

    Returns:
        None
    """
    if not rng_state:
        return
    if rng_state.get("python") is not None:
        random.setstate(rng_state["python"])
    if rng_state.get("numpy") is not None:
        np.random.set_state(rng_state["numpy"])
    if rng_state.get("torch") is not None:
        torch.set_rng_state(rng_state["torch"])
    if torch.cuda.is_available() and rng_state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rng_state["cuda"])


def should_save_checkpoint(epoch, every_n_epochs):
    """Return whether the current epoch should produce a checkpoint.

    Args:
        epoch (int): Zero-based epoch index.
        every_n_epochs (int): Save interval. For example, ``2`` saves after
            epochs 2, 4, 6, ... using one-based epoch numbering.

    Returns:
        bool: ``True`` when ``(epoch + 1)`` is divisible by ``every_n_epochs``.
    """
    every_n_epochs = int(every_n_epochs)
    if every_n_epochs <= 0:
        raise ValueError("every_n_epochs must be > 0.")
    return (int(epoch) + 1) % every_n_epochs == 0


def build_checkpoint_state(
    epoch,
    model,
    optimizer=None,
    scheduler=None,
    history=None,
    metrics=None,
    extra_state=None,
    include_rng=True,
):
    """Build a full training checkpoint state.

    Args:
        epoch (int): Zero-based epoch index that just finished.
        model (torch.nn.Module): Full trainable model. For ApexID this should be
            the combined backbone+head module, usually
            ``ModuleDict({"backbone": backbone, "head": head})``.
        optimizer (torch.optim.Optimizer, optional): Optimizer to restore when
            resuming training.
        scheduler (torch.optim.lr_scheduler.LRScheduler, optional): Scheduler to
            restore when resuming training.
        history (dict, optional): Training history accumulated so far.
        metrics (dict, optional): Latest validation or training metrics.
        extra_state (dict, optional): Additional user metadata such as config,
            scaler state, best metric, or experiment id.
        include_rng (bool): Whether to store Python/NumPy/PyTorch RNG states.

    Returns:
        dict: Serializable checkpoint containing full model weights, optimizer
        state, scheduler state, history, metrics, RNG state, and metadata needed
        to continue training from ``epoch + 1``.
    """
    model = _unwrap_module(model)
    state = {
        "epoch": int(epoch),
        "next_epoch": int(epoch) + 1,
        "model_state_dict": _state_dict_to_cpu(model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "history": history,
        "metrics": metrics,
        "extra_state": dict(extra_state or {}),
        "rng_state": collect_rng_state() if include_rng else None,
    }
    return state


def _delete_previous_checkpoints(output_dir, keep_path, patterns):
    """Delete old checkpoints while keeping ``keep_path``."""
    keep_path = Path(keep_path).resolve()
    for pattern in patterns:
        for candidate in output_dir.glob(pattern):
            if candidate.resolve() == keep_path:
                continue
            if candidate.is_file():
                candidate.unlink()


def save_training_checkpoint(
    epoch,
    model,
    optimizer=None,
    scheduler=None,
    output_dir="checkpoints",
    every_n_epochs=1,
    history=None,
    metrics=None,
    extra_state=None,
    filename="last_checkpoint.pt",
    delete_previous=True,
    delete_patterns=("checkpoint_epoch_*.pt", "last_checkpoint*.pt"),
    include_rng=True,
):
    """Save the latest resumable training checkpoint every ``n`` epochs.

    Args:
        epoch (int): Zero-based epoch index that just finished.
        model (torch.nn.Module): Full trainable model containing backbone and
            head weights.
        optimizer (torch.optim.Optimizer, optional): Optimizer state to save.
        scheduler (torch.optim.lr_scheduler.LRScheduler, optional): Scheduler
            state to save.
        output_dir (str or pathlib.Path): Directory where the checkpoint is
            written.
        every_n_epochs (int): Save interval. ``1`` saves every epoch, ``5`` saves
            after epochs 5, 10, 15, ... using one-based epoch numbers.
        history (dict, optional): Training history to store in the checkpoint.
        metrics (dict, optional): Latest metric dictionary.
        extra_state (dict, optional): Additional metadata needed for resume.
        filename (str): Output filename or format string. It may use
            ``{epoch}`` for zero-based epoch, ``{epoch_index}``, or
            ``{epoch_number}`` for one-based epoch.
        delete_previous (bool): Whether to delete older checkpoint files matching
            ``delete_patterns`` after saving the new checkpoint.
        delete_patterns (tuple[str, ...]): Glob patterns for removable checkpoint
            files inside ``output_dir``.
        include_rng (bool): Whether to save RNG states for reproducibility.

    Returns:
        pathlib.Path or None: Path to the saved checkpoint, or ``None`` when the
        current epoch does not match ``every_n_epochs``.
    """
    if not should_save_checkpoint(epoch, every_n_epochs):
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = _format_checkpoint_name(filename, epoch)
    checkpoint_path = output_dir / checkpoint_name

    state = build_checkpoint_state(
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        history=history,
        metrics=metrics,
        extra_state=extra_state,
        include_rng=include_rng,
    )

    tmp_path = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(checkpoint_path)

    if delete_previous:
        _delete_previous_checkpoints(output_dir, checkpoint_path, delete_patterns)

    return checkpoint_path


def load_training_checkpoint(
    checkpoint_path,
    model,
    optimizer=None,
    scheduler=None,
    map_location="cpu",
    strict=True,
    restore_rng=True,
):
    """Load a resumable training checkpoint.

    Args:
        checkpoint_path (str or pathlib.Path): Checkpoint file produced by
            ``save_training_checkpoint``.
        model (torch.nn.Module): Full trainable model to receive backbone+head
            weights.
        optimizer (torch.optim.Optimizer, optional): Optimizer to restore.
        scheduler (torch.optim.lr_scheduler.LRScheduler, optional): Scheduler to
            restore.
        map_location (str or torch.device): Device mapping passed to
            ``torch.load``.
        strict (bool): Strictness passed to ``model.load_state_dict``.
        restore_rng (bool): Whether to restore saved RNG state.

    Returns:
        dict: Loaded checkpoint dictionary. Use ``checkpoint["next_epoch"]`` as
        the next epoch index when resuming training.
    """
    checkpoint = _torch_load(checkpoint_path, map_location=map_location)
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint is missing 'model_state_dict'.")

    model = _unwrap_module(model)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if restore_rng:
        restore_rng_state(checkpoint.get("rng_state"))

    return checkpoint


def save_checkpoint(
    state,
    is_best=False,
    save_dir="checkpoints",
    filename="last_checkpoint.pt",
    best_name="best.pt",
    delete_previous=False,
):
    """Save an already-built checkpoint state.

    This compatibility helper is useful for code that assembles its own
    checkpoint dictionary. New training code should prefer
    ``save_training_checkpoint`` because it standardizes resume fields.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / filename
    torch.save(state, checkpoint_path)
    if is_best:
        torch.save(state, save_dir / best_name)
    if delete_previous:
        _delete_previous_checkpoints(
            save_dir,
            checkpoint_path,
            ("checkpoint_epoch_*.pt", "last_checkpoint*.pt"),
        )
    return checkpoint_path


def load_checkpoint(
    checkpoint_path,
    model=None,
    optimizer=None,
    scheduler=None,
    map_location="cpu",
    strict=True,
    restore_rng=True,
):
    """Load a checkpoint, optionally restoring model, optimizer, and scheduler.

    Args:
        checkpoint_path (str or pathlib.Path): Path to a checkpoint file.
        model (torch.nn.Module, optional): Full model to restore. Supports
            checkpoints using ``model_state_dict`` or legacy ``state_dict`` keys.
        optimizer (torch.optim.Optimizer, optional): Optimizer to restore.
        scheduler (torch.optim.lr_scheduler.LRScheduler, optional): Scheduler to
            restore.
        map_location (str or torch.device): Device mapping for ``torch.load``.
        strict (bool): Strictness for model state loading.
        restore_rng (bool): Whether to restore RNG state when present.

    Returns:
        dict: Loaded checkpoint dictionary.
    """
    checkpoint = _torch_load(checkpoint_path, map_location=map_location)

    if model is not None:
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
        if state_dict is None:
            raise KeyError("Checkpoint is missing 'model_state_dict' or 'state_dict'.")
        _unwrap_module(model).load_state_dict(state_dict, strict=strict)

    if optimizer is not None:
        optimizer_state = checkpoint.get("optimizer_state_dict", checkpoint.get("optimizer"))
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)

    if scheduler is not None:
        scheduler_state = checkpoint.get("scheduler_state_dict", checkpoint.get("lr_scheduler"))
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)

    if restore_rng:
        restore_rng_state(checkpoint.get("rng_state"))

    return checkpoint


def save_backbone_weights(backbone, output_dir="checkpoints", filename="backbone.pt"):
    """Save only backbone weights.

    Args:
        backbone (torch.nn.Module): Backbone module to save.
        output_dir (str or pathlib.Path): Directory where the file is written.
        filename (str): Backbone weights filename.

    Returns:
        pathlib.Path: Path to the saved backbone state dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    backbone = _unwrap_module(backbone)
    torch.save(_state_dict_to_cpu(backbone.state_dict()), path)
    return path


__all__ = [
    "build_checkpoint_state",
    "collect_rng_state",
    "load_checkpoint",
    "load_training_checkpoint",
    "restore_rng_state",
    "save_backbone_weights",
    "save_checkpoint",
    "save_training_checkpoint",
    "should_save_checkpoint",
]
