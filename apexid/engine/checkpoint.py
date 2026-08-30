import random
import warnings
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch


CHECKPOINT_FORMAT_VERSION = 2
EARLY_STOPPING_STATE_KEY = "early_stopping"
DATA_LOADER_GENERATOR_STATES_KEY = "data_loader_generators"


class LegacyCheckpointWarning(RuntimeWarning):
    """Warn that a historical checkpoint cannot provide exact resume semantics."""


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


def _as_byte_rng_tensor(state):
    """Normalize RNG state payloads to CPU ``torch.ByteTensor`` for PyTorch APIs."""
    if torch.is_tensor(state):
        tensor = state.detach().cpu()
        if tensor.dtype != torch.uint8:
            tensor = tensor.to(dtype=torch.uint8)
        tensor = tensor.contiguous()
    elif isinstance(state, (bytes, bytearray)):
        tensor = torch.frombuffer(state, dtype=torch.uint8)
    else:
        array = np.asarray(state, dtype=np.uint8)
        tensor = torch.from_numpy(array).contiguous()

    # torch.set_rng_state still requires the legacy ByteTensor type on some PyTorch builds.
    if not isinstance(tensor, torch.ByteTensor):
        tensor = torch.ByteTensor(tensor.numpy().astype(np.uint8, copy=False))
    return tensor.clone()


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
        "torch": _as_byte_rng_tensor(torch.get_rng_state()),
        "cuda": None,
    }
    if torch.cuda.is_available():
        state["cuda"] = [_as_byte_rng_tensor(device_state) for device_state in torch.cuda.get_rng_state_all()]
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
    try:
        if rng_state.get("python") is not None:
            random.setstate(rng_state["python"])
        if rng_state.get("numpy") is not None:
            np.random.set_state(rng_state["numpy"])
        if rng_state.get("torch") is not None:
            torch.set_rng_state(_as_byte_rng_tensor(rng_state["torch"]))
        if torch.cuda.is_available() and rng_state.get("cuda") is not None:
            cuda_state = rng_state["cuda"]
            if isinstance(cuda_state, (list, tuple)):
                cuda_state = [_as_byte_rng_tensor(device_state) for device_state in cuda_state]
            torch.cuda.set_rng_state_all(cuda_state)
    except (TypeError, ValueError, RuntimeError) as exc:
        warnings.warn(
            f"Could not restore checkpoint RNG state ({exc}); continuing with current RNG state.",
            RuntimeWarning,
            stacklevel=2,
        )


def collect_data_loader_generator_states(data_loaders):
    """Collect explicit ``torch.Generator`` states from nested DataLoaders."""
    if isinstance(data_loaders, Mapping):
        return {
            key: collect_data_loader_generator_states(loader)
            for key, loader in data_loaders.items()
        }

    generator = getattr(data_loaders, "generator", None)
    if generator is None:
        return None
    if not isinstance(generator, torch.Generator):
        raise TypeError("DataLoader generator must be a torch.Generator or None.")
    return _as_byte_rng_tensor(generator.get_state())


def restore_data_loader_generator_states(data_loaders, generator_states):
    """Restore nested DataLoader generators with strict topology validation."""
    if isinstance(data_loaders, Mapping):
        if not isinstance(generator_states, Mapping):
            raise ValueError("Saved DataLoader generator state does not match loader mapping topology.")
        current_keys = set(data_loaders)
        saved_keys = set(generator_states)
        if current_keys != saved_keys:
            raise ValueError(
                "Saved DataLoader generator state keys do not match current loaders: "
                f"checkpoint={sorted(str(key) for key in saved_keys)}, "
                f"active={sorted(str(key) for key in current_keys)}."
            )
        for key, loader in data_loaders.items():
            restore_data_loader_generator_states(loader, generator_states[key])
        return

    if isinstance(generator_states, Mapping):
        raise ValueError("Saved DataLoader generator mapping does not match a loader leaf.")

    generator = getattr(data_loaders, "generator", None)
    if generator_states is None:
        if generator is not None:
            raise ValueError("Checkpoint has no generator state for a DataLoader that now has one.")
        return
    if not isinstance(generator, torch.Generator):
        raise ValueError("Checkpoint has generator state for a DataLoader without a torch.Generator.")
    generator.set_state(_as_byte_rng_tensor(generator_states))


def _has_explicit_data_loader_generator(data_loaders):
    if isinstance(data_loaders, Mapping):
        return any(_has_explicit_data_loader_generator(loader) for loader in data_loaders.values())
    return isinstance(getattr(data_loaders, "generator", None), torch.Generator)


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
    early_stopping=None,
    data_loaders=None,
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
        early_stopping (optional): Object exposing ``state_dict()``. Its state is
            stored under ``extra_state`` for exact model-selection resume.
        data_loaders (optional): DataLoader or nested mapping of DataLoaders whose
            explicit ``torch.Generator`` states should be preserved.

    Returns:
        dict: Serializable checkpoint containing full model weights, optimizer
        state, scheduler state, history, metrics, RNG state, and metadata needed
        to continue training from ``epoch + 1``.
    """
    model = _unwrap_module(model)
    serialized_extra_state = dict(extra_state or {})
    if early_stopping is not None:
        if EARLY_STOPPING_STATE_KEY in serialized_extra_state:
            raise ValueError(
                f"extra_state already contains reserved key '{EARLY_STOPPING_STATE_KEY}'."
            )
        state_dict = getattr(early_stopping, "state_dict", None)
        if state_dict is None:
            raise TypeError("early_stopping must expose state_dict().")
        serialized_extra_state[EARLY_STOPPING_STATE_KEY] = state_dict()
    if data_loaders is not None:
        if DATA_LOADER_GENERATOR_STATES_KEY in serialized_extra_state:
            raise ValueError(
                f"extra_state already contains reserved key '{DATA_LOADER_GENERATOR_STATES_KEY}'."
            )
        serialized_extra_state[DATA_LOADER_GENERATOR_STATES_KEY] = (
            collect_data_loader_generator_states(data_loaders)
        )

    state = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "epoch": int(epoch),
        "next_epoch": int(epoch) + 1,
        "model_state_dict": _state_dict_to_cpu(model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "history": history,
        "metrics": metrics,
        "extra_state": serialized_extra_state,
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
    early_stopping=None,
    data_loaders=None,
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
        early_stopping (optional): Early-stopping object to serialize.
        data_loaders (optional): DataLoader or nested mapping whose explicit
            generator states should be serialized.

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
        early_stopping=early_stopping,
        data_loaders=data_loaders,
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
    early_stopping=None,
    data_loaders=None,
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
        early_stopping (optional): Active early-stopping object whose state must
            match and receive the checkpoint state.
        data_loaders (optional): Active DataLoader or nested mapping whose
            explicit generator states should be restored.

    Returns:
        dict: Loaded checkpoint dictionary. Use ``checkpoint["next_epoch"]`` as
        the next epoch index when resuming training.
    """
    checkpoint = _torch_load(checkpoint_path, map_location=map_location)
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint is missing 'model_state_dict'.")

    extra_state = checkpoint.get("extra_state")
    if not isinstance(extra_state, Mapping):
        extra_state = {}
    missing_resume_state = []
    if early_stopping is not None:
        early_stopping_state = extra_state.get(EARLY_STOPPING_STATE_KEY)
        if early_stopping_state is None:
            missing_resume_state.append("early-stopping state")
        else:
            load_state_dict = getattr(early_stopping, "load_state_dict", None)
            if load_state_dict is None:
                raise TypeError("early_stopping must expose load_state_dict().")
            load_state_dict(early_stopping_state)

    model = _unwrap_module(model)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if restore_rng:
        restore_rng_state(checkpoint.get("rng_state"))

    if data_loaders is not None:
        if DATA_LOADER_GENERATOR_STATES_KEY not in extra_state:
            if _has_explicit_data_loader_generator(data_loaders):
                missing_resume_state.append("DataLoader generator state")
        else:
            restore_data_loader_generator_states(
                data_loaders,
                extra_state[DATA_LOADER_GENERATOR_STATES_KEY],
            )

    if missing_resume_state:
        warnings.warn(
            "Legacy checkpoint does not contain "
            + " and ".join(missing_resume_state)
            + ". Available model/optimizer/scheduler/history/RNG state was restored, "
            "but resume is not guaranteed equivalent to uninterrupted training; "
            "missing resume state is not reconstructed.",
            LegacyCheckpointWarning,
            stacklevel=2,
        )

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
    "CHECKPOINT_FORMAT_VERSION",
    "LegacyCheckpointWarning",
    "build_checkpoint_state",
    "collect_data_loader_generator_states",
    "collect_rng_state",
    "load_checkpoint",
    "load_training_checkpoint",
    "restore_rng_state",
    "restore_data_loader_generator_states",
    "save_backbone_weights",
    "save_checkpoint",
    "save_training_checkpoint",
    "should_save_checkpoint",
]
