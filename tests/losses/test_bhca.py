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


@pytest.mark.parametrize("head_cls", [BHCAArcFace, BHCASubCenterArcFace, BHCASubCenterSphereFace2])
def test_independent_auxiliary_weights_and_logged_contributions(head_cls):
    head = head_cls(
        3, 4, class_to_species=[0, 0, 1, 1],
        hierarchy_weight=7.0, species_weight=0.02, within_species_weight=0.5,
    )
    output = head.compute_output(_embeddings(), torch.arange(4))
    expected = output.loss_id_global + 0.02 * output.loss_species + 0.5 * output.loss_within_species
    torch.testing.assert_close(output.loss_total, expected)
    assert output.metrics["loss_species_weighted"] == pytest.approx(0.02 * output.loss_species.item())
    assert output.metrics["loss_within_species_weighted"] == pytest.approx(0.5 * output.loss_within_species.item())
    assert output.metrics["loss_hierarchy_weighted"] == pytest.approx(
        output.metrics["loss_species_weighted"] + output.metrics["loss_within_species_weighted"]
    )
    output.loss_total.backward()
    assert torch.isfinite(head.identity_weight.grad).all()


def test_zero_auxiliary_weights_disable_their_gradients():
    head = BHCASubCenterSphereFace2(
        3, 4, class_to_species=[0, 0, 1, 1], hierarchy_weight=10,
        species_weight=0, within_species_weight=0, within_negative_reduction="sum",
    )
    output = head.compute_output(_embeddings(), torch.arange(4))
    torch.testing.assert_close(output.loss_total, output.loss_id_global)
    output.loss_total.backward()
    assert head.species_proxies.grad.abs().sum() == 0


@pytest.mark.parametrize("sibling_count", [2, 99, 999])
@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_exact_sibling_gradient_boost_is_count_independent_with_sum(sibling_count, reduction):
    # One eligible anchor: sibling negatives followed by one cross-species negative.
    classes = sibling_count + 2
    head = BHCASubCenterSphereFace2(
        3, classes, class_to_species=[0] * (classes - 1) + [1],
        species_weight=0.05, within_species_weight=0.5,
        within_negative_reduction=reduction, r=8, t=1,
    )
    with torch.no_grad():
        head.b.fill_(-0.5)
    scores = torch.full((1, classes), 0.3, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0])
    global_loss = head._global_identity_loss(scores, labels)
    within_loss, _ = head._within_species_objective(scores, labels, torch.tensor([0]))
    global_gradient = torch.autograd.grad(global_loss, scores, retain_graph=True)[0]
    combined_gradient = torch.autograd.grad(global_loss + 0.5 * within_loss, scores)[0]
    ratio = 1.5 if reduction == "sum" else 1.0 + 0.5 / sibling_count
    torch.testing.assert_close(combined_gradient[0, 1:-1], ratio * global_gradient[0, 1:-1])
    torch.testing.assert_close(combined_gradient[0, 0], 1.5 * global_gradient[0, 0])
    torch.testing.assert_close(combined_gradient[0, -1], global_gradient[0, -1])


def test_legacy_within_formula_is_preserved_explicitly():
    head = BHCASubCenterSphereFace2(3, 4, class_to_species=[0, 0, 0, 1], hierarchy_weight=0.2)
    scores = torch.tensor([[0.4, 0.3, 0.2]])
    labels = torch.tensor([0])
    expected = sphereface2_loss_from_cosine(
        scores, labels, magn_type=head.magn_type, alpha=head.alpha, r=head.r,
        m=head.identity_margin, t=head.t, lw=head.lw, bias=head.b, normalize_negatives=True,
    )
    torch.testing.assert_close(head._within_group_loss(scores, labels), expected)
    assert head.species_weight == pytest.approx(0.1)
    assert head.within_species_weight == pytest.approx(0.1)


@pytest.mark.parametrize("mapping", [[-100, -100, -100], [0, 1, 2], [0, 0, 1]])
def test_corrected_mode_handles_missing_and_singleton_categories(mapping):
    head = BHCASubCenterSphereFace2(
        3, 3, class_to_species=mapping, species_weight=0.05,
        within_species_weight=0.5, within_negative_reduction="sum",
    )
    x = _embeddings()[:3].detach().clone().requires_grad_(True)
    output = head.compute_output(x, torch.arange(3))
    output.loss_total.backward()
    assert torch.isfinite(output.loss_total)
    assert torch.isfinite(x.grad).all()
    if mapping != [0, 0, 1]:
        assert output.loss_within_species.item() == 0


@pytest.mark.parametrize("name", ["species_weight", "within_species_weight", "hierarchy_weight"])
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_auxiliary_weights_fail(name, value):
    with pytest.raises(ValueError, match="finite and non-negative"):
        BHCAArcFace(3, 4, class_to_species=[0, 0, 1, 1], **{name: value})


def test_invalid_negative_reduction_fails():
    with pytest.raises(ValueError, match="within_negative_reduction"):
        BHCASubCenterSphereFace2(3, 4, class_to_species=[0, 0, 1, 1], within_negative_reduction="median")


def test_checkpoint_preserves_and_checks_objective_settings():
    kwargs = dict(
        class_to_species=[0, 0, 1, 1], species_weight=0.05,
        within_species_weight=0.5, within_negative_reduction="sum",
    )
    head = BHCASubCenterSphereFace2(3, 4, **kwargs)
    head.eval()
    x, labels = _embeddings(), torch.arange(4)
    state = head.state_dict()
    restored = BHCASubCenterSphereFace2(3, 4, **kwargs)
    restored.load_state_dict(state)
    torch.testing.assert_close(restored(x, labels), head(x, labels))
    for change in ({"within_species_weight": 0.1}, {"species_weight": 0.1}, {"within_negative_reduction": "mean"}):
        mismatched = BHCASubCenterSphereFace2(3, 4, **(kwargs | change))
        with pytest.raises(ValueError, match="objective settings differ"):
            mismatched.load_state_dict(state)


def test_legacy_checkpoint_requires_legacy_objective():
    head = BHCASubCenterSphereFace2(3, 4, class_to_species=[0, 0, 1, 1])
    state = head.state_dict()
    state["_extra_state"].pop("objective_settings")
    head.load_state_dict(state)
    corrected = BHCASubCenterSphereFace2(
        3, 4, class_to_species=[0, 0, 1, 1], species_weight=0.05,
        within_species_weight=0.5, within_negative_reduction="sum",
    )
    with pytest.raises(ValueError, match="Legacy BHCA checkpoint"):
        corrected.load_state_dict(state)


def test_reweighted_config_builds_head_and_uses_new_output_directory():
    from pathlib import Path
    import yaml
    root = Path(__file__).resolve().parents[2]
    current = yaml.safe_load((root / "configs/cluster_bhca_subcenter_sphereface2.yaml").read_text())
    legacy = yaml.safe_load((root / "configs/cluster_bhca_subcenter_sphereface2_legacy.yaml").read_text())
    head = BHCASubCenterSphereFace2(3, 4, class_to_species=[0, 0, 1, 1], **current["head"]["kwargs"])
    assert head.within_negative_reduction == "sum"
    assert head.species_weight == 0.05
    assert head.within_species_weight == 0.5
    assert current["checkpoint"]["output_dir"] != legacy["checkpoint"]["output_dir"]
    assert current["early_stopping"]["output_dir"] == current["checkpoint"]["output_dir"]
