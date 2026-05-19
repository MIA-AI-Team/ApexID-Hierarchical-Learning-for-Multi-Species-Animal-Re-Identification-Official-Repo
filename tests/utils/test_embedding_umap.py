import json
import sys
import types

import pytest
import torch

from apexid.utils.embedding_umap import plot_umap_embeddings


class FakeUMAP:
    def __init__(self, n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=None):
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state

    def fit_transform(self, embeddings):
        if embeddings.shape[1] == 1:
            return torch.as_tensor(embeddings).repeat(1, 2).numpy()
        return embeddings[:, :2]


@pytest.fixture(autouse=True)
def fake_umap(monkeypatch):
    monkeypatch.setitem(sys.modules, "umap", types.SimpleNamespace(UMAP=FakeUMAP))


def test_plot_umap_embeddings_writes_png_and_metadata(tmp_path):
    embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.1],
            [0.9, 0.1, 0.2],
            [0.0, 1.0, 0.3],
            [0.1, 0.9, 0.4],
            [0.5, 0.5, 0.5],
            [0.6, 0.4, 0.6],
        ]
    )
    labels = torch.tensor([0, 0, 1, 1, 2, 2])

    result = plot_umap_embeddings(
        embeddings,
        labels,
        tmp_path / "epoch_001_umap.png",
        epoch=0,
        random_state=42,
    )

    assert result["png"].exists()
    assert result["json"].exists()
    assert result["metadata"]["epoch_number"] == 1
    assert result["metadata"]["split"] == "val"
    assert result["metadata"]["num_samples"] == 6
    assert result["metadata"]["label_counts"] == {"0": 2, "1": 2, "2": 2}
    assert set(result["metadata"]["label_to_color"]) == {"0", "1", "2"}

    with open(result["json"], "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["output_path"] == str(result["png"])
    assert payload["metadata_path"] == str(result["json"])


def test_plot_umap_embeddings_color_mapping_is_deterministic(tmp_path):
    embeddings = torch.randn(6, 4)
    labels = torch.tensor([2, 1, 2, 0, 1, 0])

    first = plot_umap_embeddings(embeddings, labels, tmp_path / "first.png", random_state=7)
    second = plot_umap_embeddings(embeddings, labels, tmp_path / "second.png", random_state=7)

    assert first["metadata"]["label_to_color"] == second["metadata"]["label_to_color"]


def test_plot_umap_embeddings_validates_inputs(tmp_path):
    with pytest.raises(ValueError, match="shape"):
        plot_umap_embeddings(torch.randn(4), torch.arange(4), tmp_path / "bad.png")

    with pytest.raises(ValueError, match="same number"):
        plot_umap_embeddings(torch.randn(4, 2), torch.arange(3), tmp_path / "bad.png")

    with pytest.raises(ValueError, match="At least 3 samples"):
        plot_umap_embeddings(torch.randn(2, 2), torch.arange(2), tmp_path / "bad.png")
