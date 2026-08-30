import io
import random
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from apexid.engine.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    LegacyCheckpointWarning,
    collect_rng_state,
    load_checkpoint,
    load_training_checkpoint,
    restore_rng_state,
    save_training_checkpoint,
    should_save_checkpoint,
)
from apexid.engine.early_stopping import EarlyStopping


def _build_model_optimizer_scheduler():
    model = nn.ModuleDict(
        {
            "backbone": nn.Sequential(nn.Linear(4, 3), nn.ReLU()),
            "head": nn.Linear(3, 2),
        }
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    return model, optimizer, scheduler


def _run_one_update(model, optimizer, scheduler):
    x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    y = torch.tensor([0, 1])
    optimizer.zero_grad()
    embeddings = model["backbone"](x)
    logits = model["head"](embeddings)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    optimizer.step()
    scheduler.step()


def _clone_state_dict(module):
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def _assert_optimizer_states_close(actual, expected):
    assert actual["param_groups"] == expected["param_groups"]
    assert actual["state"].keys() == expected["state"].keys()
    for param_id, expected_state in expected["state"].items():
        actual_state = actual["state"][param_id]
        assert actual_state.keys() == expected_state.keys()
        for key, expected_value in expected_state.items():
            actual_value = actual_state[key]
            if torch.is_tensor(expected_value):
                torch.testing.assert_close(actual_value, expected_value)
            else:
                assert actual_value == expected_value


def test_should_save_checkpoint_uses_one_based_epoch_numbers():
    assert should_save_checkpoint(epoch=0, every_n_epochs=2) is False
    assert should_save_checkpoint(epoch=1, every_n_epochs=2) is True
    assert should_save_checkpoint(epoch=4, every_n_epochs=5) is True


def test_save_training_checkpoint_every_n_epochs_and_delete_previous(tmp_path):
    model, optimizer, scheduler = _build_model_optimizer_scheduler()
    _run_one_update(model, optimizer, scheduler)

    skipped = save_training_checkpoint(
        epoch=0,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=tmp_path,
        every_n_epochs=2,
        filename="checkpoint_epoch_{epoch_number:03d}.pt",
    )
    assert skipped is None

    first_path = save_training_checkpoint(
        epoch=1,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=tmp_path,
        every_n_epochs=2,
        filename="checkpoint_epoch_{epoch_number:03d}.pt",
    )
    assert first_path is not None
    assert first_path.name == "checkpoint_epoch_002.pt"
    assert first_path.exists()

    _run_one_update(model, optimizer, scheduler)
    second_path = save_training_checkpoint(
        epoch=3,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=tmp_path,
        every_n_epochs=2,
        filename="checkpoint_epoch_{epoch_number:03d}.pt",
    )

    assert second_path.name == "checkpoint_epoch_004.pt"
    assert second_path.exists()
    assert not first_path.exists()
    assert sorted(path.name for path in tmp_path.glob("checkpoint_epoch_*.pt")) == [
        "checkpoint_epoch_004.pt"
    ]


def test_load_training_checkpoint_restores_model_optimizer_scheduler_and_metadata(tmp_path):
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    model, optimizer, scheduler = _build_model_optimizer_scheduler()
    _run_one_update(model, optimizer, scheduler)

    expected_model_state = _clone_state_dict(model)
    expected_optimizer_state = deepcopy(optimizer.state_dict())
    expected_scheduler_state = deepcopy(scheduler.state_dict())
    history = {"train_loss": [1.2], "val_loss": [1.1]}
    metrics = {"val_top1": 0.75}

    checkpoint_path = save_training_checkpoint(
        epoch=4,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=tmp_path,
        every_n_epochs=1,
        history=history,
        metrics=metrics,
        extra_state={"experiment": "checkpoint-test"},
    )

    with torch.no_grad():
        for param in model.parameters():
            param.add_(10.0)
    optimizer.param_groups[0]["lr"] = 999.0
    scheduler.gamma = 0.123

    checkpoint = load_training_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    for key, expected_value in expected_model_state.items():
        torch.testing.assert_close(model.state_dict()[key], expected_value)
    _assert_optimizer_states_close(optimizer.state_dict(), expected_optimizer_state)
    assert scheduler.state_dict() == expected_scheduler_state
    assert checkpoint["epoch"] == 4
    assert checkpoint["next_epoch"] == 5
    assert checkpoint["history"] == history
    assert checkpoint["metrics"] == metrics
    assert checkpoint["extra_state"]["experiment"] == "checkpoint-test"


def test_rng_state_round_trip_survives_torch_save_load():
    torch.manual_seed(123)
    random.seed(123)
    np.random.seed(123)

    expected = {
        "python": random.random(),
        "numpy": float(np.random.rand()),
        "torch": torch.rand(4).tolist(),
    }

    torch.manual_seed(123)
    random.seed(123)
    np.random.seed(123)
    rng_state = collect_rng_state()

    buffer = io.BytesIO()
    torch.save(rng_state, buffer)
    buffer.seek(0)
    loaded_rng = torch.load(buffer, weights_only=False)

    random.random()
    np.random.rand()
    torch.rand(4)

    restore_rng_state(loaded_rng)

    actual = {
        "python": random.random(),
        "numpy": float(np.random.rand()),
        "torch": torch.rand(4).tolist(),
    }
    assert actual == expected


def test_restore_rng_state_accepts_uint8_tensor_from_legacy_checkpoint():
    torch.manual_seed(7)
    _ = torch.rand(1)
    expected_next = torch.rand(3).tolist()

    torch.manual_seed(7)
    _ = torch.rand(1)
    legacy_state = {
        "torch": torch.get_rng_state().to(dtype=torch.uint8),
    }

    torch.rand(99)

    restore_rng_state(legacy_state)
    assert torch.rand(3).tolist() == expected_next


def test_load_training_checkpoint_tolerates_invalid_rng_state(tmp_path):
    model, optimizer, scheduler = _build_model_optimizer_scheduler()
    checkpoint_path = save_training_checkpoint(
        epoch=1,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=tmp_path,
        every_n_epochs=1,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["rng_state"] = {"torch": "legacy-invalid-rng-state"}
    torch.save(checkpoint, checkpoint_path)

    with pytest.warns(RuntimeWarning, match="Could not restore checkpoint RNG state"):
        loaded = load_training_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    assert loaded["next_epoch"] == 2


def test_load_checkpoint_supports_legacy_state_dict_key(tmp_path):
    model, _, _ = _build_model_optimizer_scheduler()
    expected_state = _clone_state_dict(model)
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save({"epoch": 2, "state_dict": model.state_dict()}, checkpoint_path)

    with torch.no_grad():
        for param in model.parameters():
            param.zero_()

    checkpoint = load_checkpoint(checkpoint_path, model=model)

    assert checkpoint["epoch"] == 2
    for key, expected_value in expected_state.items():
        torch.testing.assert_close(model.state_dict()[key], expected_value)


def _loader_with_generator(seed):
    dataset = TensorDataset(torch.arange(8, dtype=torch.float32).reshape(4, 2))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=2, shuffle=True, generator=generator)


def test_checkpoint_v2_round_trip_restores_early_stopping_and_loader_generator(tmp_path):
    model, optimizer, scheduler = _build_model_optimizer_scheduler()
    stopper = EarlyStopping(
        monitor="score",
        mode="max",
        patience=3,
        output_dir=tmp_path / "best",
    )
    stopper(0.6, model=model, epoch=0)
    stopper(0.5, model=model, epoch=1)
    loader = _loader_with_generator(123)
    next(iter(loader))
    expected_loader_state = loader.generator.get_state().clone()
    expected_stopper_state = deepcopy(stopper.state_dict())

    checkpoint_path = save_training_checkpoint(
        epoch=1,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=tmp_path,
        early_stopping=stopper,
        data_loaders={"train": loader},
    )

    restored_model, restored_optimizer, restored_scheduler = _build_model_optimizer_scheduler()
    restored_stopper = EarlyStopping(
        monitor="score",
        mode="max",
        patience=3,
        output_dir=tmp_path / "restored_best",
    )
    restored_loader = _loader_with_generator(999)
    checkpoint = load_training_checkpoint(
        checkpoint_path,
        restored_model,
        restored_optimizer,
        restored_scheduler,
        early_stopping=restored_stopper,
        data_loaders={"train": restored_loader},
    )

    assert checkpoint["checkpoint_format_version"] == CHECKPOINT_FORMAT_VERSION
    assert restored_stopper.state_dict() == expected_stopper_state
    assert torch.equal(restored_loader.generator.get_state(), expected_loader_state)


def test_checkpoint_rejects_changed_loader_topology(tmp_path):
    model, optimizer, scheduler = _build_model_optimizer_scheduler()
    checkpoint_path = save_training_checkpoint(
        epoch=0,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=tmp_path,
        data_loaders={"train": _loader_with_generator(1)},
    )

    with pytest.raises(ValueError, match="keys do not match"):
        load_training_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            data_loaders={"training": _loader_with_generator(1)},
        )


def test_legacy_checkpoint_warns_when_early_stopping_state_is_unavailable(tmp_path):
    model, optimizer, scheduler = _build_model_optimizer_scheduler()
    checkpoint_path = tmp_path / "legacy_training.pt"
    torch.save(
        {
            "epoch": 2,
            "next_epoch": 3,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": {"train_loss": [1.0]},
            "rng_state": None,
        },
        checkpoint_path,
    )
    stopper = EarlyStopping(output_dir=tmp_path / "best")

    with pytest.warns(LegacyCheckpointWarning, match="not guaranteed equivalent"):
        loaded = load_training_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            early_stopping=stopper,
        )

    assert loaded["next_epoch"] == 3
    assert stopper.best_metric is None


def test_adamw_and_cyclic_lr_state_survive_checkpoint_resume(tmp_path):
    model = nn.ModuleDict({"backbone": nn.Linear(4, 3), "head": nn.Linear(3, 2)})
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4, foreach=False)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=1e-4,
        max_lr=1e-3,
        step_size_up=2,
        cycle_momentum=False,
    )
    for _ in range(3):
        _run_one_update(model, optimizer, scheduler)
    expected_optimizer = deepcopy(optimizer.state_dict())
    expected_scheduler = deepcopy(scheduler.state_dict())

    checkpoint_path = save_training_checkpoint(
        epoch=0,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=tmp_path,
    )

    restored_model = nn.ModuleDict({"backbone": nn.Linear(4, 3), "head": nn.Linear(3, 2)})
    restored_optimizer = torch.optim.AdamW(
        restored_model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
        foreach=False,
    )
    restored_scheduler = torch.optim.lr_scheduler.CyclicLR(
        restored_optimizer,
        base_lr=1e-4,
        max_lr=1e-3,
        step_size_up=2,
        cycle_momentum=False,
    )
    load_training_checkpoint(
        checkpoint_path,
        restored_model,
        restored_optimizer,
        restored_scheduler,
    )

    _assert_optimizer_states_close(restored_optimizer.state_dict(), expected_optimizer)
    assert restored_scheduler.state_dict() == expected_scheduler
