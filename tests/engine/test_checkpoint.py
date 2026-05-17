import random
from copy import deepcopy

import numpy as np
import torch
from torch import nn

from apexid.engine.checkpoint import (
    load_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
    should_save_checkpoint,
)


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
