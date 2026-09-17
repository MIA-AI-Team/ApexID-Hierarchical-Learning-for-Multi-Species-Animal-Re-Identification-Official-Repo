import math

import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from apexid.data.species_metadata import (
    build_training_hierarchy,
    sqrt_inverse_identity_frequency,
)
from apexid.losses.bhca import (
    BHCAArcFace,
    BHCASubCenterArcFace,
    BHCASubCenterSphereFace2,
    _arcface_logits,
)
from apexid.losses.sphereface2 import SphereFace2, sphereface2_loss_from_cosine
from apexid.models.heads.arcface import ArcFaceHead


def _embeddings():
    return torch.tensor(
        [[1.0, 0.1, 0.0], [0.8, 0.2, 0.1], [0.0, 1.0, 0.1], [0.1, 0.8, 0.2]],
        requires_grad=True,
    )


def test_training_hierarchy_uses_identity_frequency_and_masks_missing():
    frame = pd.DataFrame({
        "identity": ["a", "a", "b", "c", "d", "e"],
        "species": ["common", "common", "common", "common", "rare", None],
    })
    metadata = build_training_hierarchy(
        frame, {name: index for index, name in enumerate(["a", "b", "c", "d", "e"])}
    )
    assert metadata.identities_per_species == [3, 1]
    assert metadata.class_to_species[-1] == -100
    assert metadata.missing_parent_identities == 1
    assert metadata.species_weights[1] > metadata.species_weights[0]
    assert sum(metadata.species_weights) / 2 == pytest.approx(1.0)


def test_species_weights_are_mean_one_and_parameter_free():
    weights = sqrt_inverse_identity_frequency([1, 4, 16])
    assert weights.mean() == pytest.approx(1.0)
    assert weights[0] > weights[1] > weights[2]


def test_training_hierarchy_rejects_conflicting_explicit_parents():
    frame = pd.DataFrame({"identity": ["a", "a"], "species": ["lion", "turtle"]})
    with pytest.raises(ValueError, match="conflicting"):
        build_training_hierarchy(frame, {"a": 0})


def test_arcface_exact_global_margin_and_total_equation():
    head = BHCAArcFace(
        3, 4, s=16.0, class_to_species=[0, 0, 1, 1], hierarchy_weight=0.2
    )
    x = _embeddings()
    labels = torch.tensor([0, 1, 2, 3])
    output = head.compute_output(x, labels)
    expected = F.cross_entropy(
        _arcface_logits(output.identity_cosine, labels, head.identity_margin, head.s), labels
    )
    torch.testing.assert_close(output.loss_id_global, expected)
    torch.testing.assert_close(
        output.loss_hierarchy, 0.5 * (output.loss_species + output.loss_within_species)
    )
    torch.testing.assert_close(
        output.loss_total, output.loss_id_global + 0.2 * output.loss_hierarchy
    )


def test_conditional_candidates_use_global_bank_and_exclude_cross_species():
    head = BHCAArcFace(3, 4, s=8.0, class_to_species=[0, 0, 1, 1])
    x = _embeddings()
    labels = torch.tensor([0, 0, 0, 0])  # only one batch identity from species zero
    first = head.compute_output(x, labels)
    first_within = first.loss_within_species.detach().clone()
    first_global = first.loss_id_global.detach().clone()
    with torch.no_grad():
        head.identity_weight[3].copy_(torch.tensor([10.0, -7.0, 2.0]))
    second = head.compute_output(x, labels)
    torch.testing.assert_close(second.loss_within_species, first_within)
    assert not torch.allclose(second.loss_id_global, first_global)
    assert second.metrics["within_candidate_identities_min"] == 2


def test_missing_parent_keeps_global_identity_training_and_connected_zero():
    head = BHCAArcFace(3, 3, class_to_species=[-100, -100, -100])
    x = _embeddings()[:2].detach().clone().requires_grad_(True)
    labels = torch.tensor([0, 1])
    output = head.compute_output(x, labels)
    assert output.loss_id_global.item() > 0
    assert output.loss_species.item() == 0
    assert output.loss_within_species.item() == 0
    output.loss_total.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert head.identity_weight.grad is not None
    assert head.species_proxies.grad is not None


def test_singleton_species_skips_only_conditional_branch():
    head = BHCAArcFace(3, 3, class_to_species=[0, 1, 1])
    output = head.compute_output(_embeddings()[:2], torch.tensor([0, 1]))
    assert output.metrics["within_singleton_anchors_skipped"] == 1
    assert output.metrics["within_valid_anchors"] == 1
    assert torch.isfinite(output.loss_species)


def test_explicit_missing_species_masks_only_that_row():
    head = BHCAArcFace(3, 3, class_to_species=[0, 0, 1])
    x = _embeddings()[:2]
    output = head.compute_output(
        x,
        torch.tensor([0, 1]),
        species_targets=torch.tensor([-100, 0]),
    )
    assert output.metrics["species_valid_anchors"] == 1
    assert output.metrics["within_valid_anchors"] == 1
    assert output.loss_id_global.item() > 0


@pytest.mark.parametrize(
    "head_cls", [BHCAArcFace, BHCASubCenterArcFace, BHCASubCenterSphereFace2]
)
def test_all_variants_have_finite_deterministic_gradients(head_cls):
    torch.manual_seed(9)
    head = head_cls(3, 4, class_to_species=[0, 0, 1, 1])
    x = _embeddings()
    labels = torch.tensor([0, 1, 2, 3])
    first = head(x, labels)
    second = head(x, labels)
    assert torch.allclose(first, second)
    first.backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(head.identity_weight.grad).all()
    assert torch.isfinite(head.species_proxies.grad).all()


def test_subcenter_max_and_gradient_reach_only_winner_for_one_score():
    head = BHCASubCenterArcFace(
        2, 2, num_subcenters=2, class_to_species=[0, 0], hierarchy_weight=0
    )
    with torch.no_grad():
        head.identity_weight.copy_(torch.tensor([
            [[1.0, 0.0], [0.0, 1.0]],
            [[-1.0, 0.0], [0.0, -1.0]],
        ]))
    x = torch.tensor([[0.8, 0.2]], requires_grad=True)
    scores, winners = head._identity_scores(F.normalize(x, dim=1))
    expected = torch.tensor([[0.8, -0.2]]) / math.sqrt(0.8**2 + 0.2**2)
    torch.testing.assert_close(scores, expected)
    assert winners.tolist() == [[0, 1]]
    scores[0, 0].backward()
    assert head.identity_weight.grad[0, 0].abs().sum() > 0
    assert head.identity_weight.grad[0, 1].abs().sum() == 0


def test_subcenter_arcface_k1_matches_single_center_arcface():
    torch.manual_seed(4)
    ordinary = ArcFaceHead(3, 4, s=16.0, m=0.3)
    bhca = BHCASubCenterArcFace(
        3, 4, s=16.0, identity_margin=0.3, num_subcenters=1,
        class_to_species=[-100] * 4, hierarchy_weight=0,
    )
    with torch.no_grad():
        bhca.identity_weight[:, 0].copy_(ordinary.weight)
    x = _embeddings()
    labels = torch.tensor([0, 1, 2, 3])
    expected = F.cross_entropy(ordinary(x, labels), labels)
    torch.testing.assert_close(bhca(x, labels), expected, atol=1e-6, rtol=1e-6)


def test_subcenter_sphereface2_k1_matches_ordinary_sphereface2():
    torch.manual_seed(5)
    ordinary = SphereFace2(3, 4, magn_type="A", alpha=0.9, r=40, m=0.3, t=3, lw=20)
    bhca = BHCASubCenterSphereFace2(
        3, 4, magn_type="A", alpha=0.9, r=40, identity_margin=0.3, t=3, lw=20,
        num_subcenters=1, class_to_species=[-100] * 4, hierarchy_weight=0,
    )
    with torch.no_grad():
        bhca.identity_weight[:, 0].copy_(ordinary.w.T)
        bhca.b.copy_(ordinary.b)
    x = _embeddings()
    labels = torch.tensor([0, 1, 2, 3])
    torch.testing.assert_close(bhca(x, labels), ordinary(x, labels), atol=1e-6, rtol=1e-6)


def test_external_cosine_sphereface2_path_matches_ordinary_head():
    torch.manual_seed(6)
    ordinary = SphereFace2(3, 4, magn_type="C", alpha=0.7, r=16, m=0.2, t=2, lw=10)
    x = _embeddings()
    labels = torch.tensor([0, 1, 2, 3])
    expected = ordinary(x, labels)
    cosine = F.normalize(x, dim=1).mm(ordinary.w)
    external = sphereface2_loss_from_cosine(
        cosine, labels, magn_type="C", alpha=0.7, r=16, m=0.2, t=2, lw=10,
        bias=ordinary.b,
    )
    torch.testing.assert_close(external, expected, atol=1e-7, rtol=1e-7)


def test_checkpoint_state_preserves_mapping_weights_and_subcenters():
    head = BHCASubCenterArcFace(3, 4, class_to_species=[0, 0, 1, -100])
    state = head.state_dict()
    restored = BHCASubCenterArcFace(3, 4, class_to_species=[0, 0, 1, -100])
    restored.load_state_dict(state)
    assert torch.equal(restored.class_to_species, head.class_to_species)
    assert torch.equal(restored.species_balance_weights, head.species_balance_weights)
    assert torch.equal(restored.identity_weight, head.identity_weight)
    assert torch.equal(restored.species_proxies, head.species_proxies)


def test_only_one_identity_proxy_parameter_bank_exists():
    for head in (
        BHCAArcFace(3, 4, class_to_species=[0, 0, 1, 1]),
        BHCASubCenterArcFace(3, 4, class_to_species=[0, 0, 1, 1]),
        BHCASubCenterSphereFace2(3, 4, class_to_species=[0, 0, 1, 1]),
    ):
        identity_parameters = [name for name, _ in head.named_parameters() if "identity" in name]
        assert identity_parameters == ["identity_weight"]
