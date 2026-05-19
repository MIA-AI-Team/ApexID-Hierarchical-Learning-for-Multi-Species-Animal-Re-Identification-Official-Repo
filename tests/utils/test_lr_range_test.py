import copy
import csv
import json

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from apexid.utils.lr_range_test import run_lr_range_test, write_lr_range_test_artifacts


class TrainHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, embeddings, labels):
        return self.linear(embeddings)


def _loader():
    images = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
    )
    labels = torch.tensor([0, 0, 1, 1])
    return DataLoader(TensorDataset(images, labels, labels), batch_size=2, shuffle=False)


def _assert_state_dict_close(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        assert torch.allclose(actual[key], expected[key])


def test_lr_range_test_restores_state_and_writes_artifacts(tmp_path):
    torch.manual_seed(42)
    backbone = nn.Linear(2, 2)
    head = TrainHead()
    optimizer = torch.optim.SGD(list(backbone.parameters()) + list(head.parameters()), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    backbone_state = copy.deepcopy(backbone.state_dict())
    head_state = copy.deepcopy(head.state_dict())
    optimizer_lr = optimizer.param_groups[0]["lr"]

    result = run_lr_range_test(
        backbone,
        head,
        criterion,
        _loader(),
        optimizer,
        torch.device("cpu"),
        start_lr=1e-5,
        end_lr=1e-2,
        num_iter=4,
    )

    _assert_state_dict_close(backbone.state_dict(), backbone_state)
    _assert_state_dict_close(head.state_dict(), head_state)
    assert optimizer.param_groups[0]["lr"] == optimizer_lr
    assert len(result.lrs) == len(result.losses) == len(result.raw_losses)
    assert len(result.lrs) > 0
    assert torch.isfinite(torch.tensor(result.losses)).all()
    assert result.suggested_base_lr > 0.0
    assert result.suggested_max_lr > result.suggested_base_lr

    paths = write_lr_range_test_artifacts(result, tmp_path)
    assert paths["json"].exists()
    assert paths["csv"].exists()

    with open(paths["json"], "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["suggested_base_lr"] == result.suggested_base_lr
    assert payload["suggested_max_lr"] == result.suggested_max_lr

    with open(paths["csv"], "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(result.lrs)
    assert set(rows[0]) == {"iteration", "lr", "loss", "raw_loss"}


def test_lr_range_test_validates_lr_bounds():
    with pytest.raises(ValueError, match="end_lr must be larger"):
        run_lr_range_test(
            nn.Linear(2, 2),
            TrainHead(),
            nn.CrossEntropyLoss(),
            _loader(),
            torch.optim.SGD(TrainHead().parameters(), lr=0.01),
            torch.device("cpu"),
            start_lr=1e-2,
            end_lr=1e-3,
        )
