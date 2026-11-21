"""ApexID Hierarchical Loss (paper equations 4–11)."""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .sphereface2 import (
    sphereface2_initial_bias,
    sphereface2_loss_from_cosine,
)


IGNORE_INDEX = -100


@dataclass
class HierarchicalLossOutput:
    """Differentiable ApexID loss components returned by ``compute_output``."""

    loss_total: torch.Tensor
    loss_id_global: torch.Tensor
    loss_species: torch.Tensor
    identity_cosine: torch.Tensor
    species_cosine: torch.Tensor
    metrics: dict


def _arcface_logits(cosine, labels, margin, scale):
    """Apply the repository's stable additive angular margin transform."""
    margin = float(margin)
    cosine = cosine.clamp(-1.0, 1.0)
    sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-7))
    phi = cosine * math.cos(margin) - sine * math.sin(margin)
    threshold = math.cos(math.pi - margin)
    correction = math.sin(math.pi - margin) * margin
    phi = torch.where(cosine > threshold, phi, cosine - correction)
    one_hot = F.one_hot(labels.long(), num_classes=cosine.shape[1]).to(cosine.dtype)
    return float(scale) * (one_hot * phi + (1.0 - one_hot) * cosine)


def _finite_float(value):
    if torch.is_tensor(value):
        value = value.detach().float().cpu().item()
    return float(value)


class ApexIDHierarchicalHead(nn.Module):
    """Combine global Sub-center SphereFace2 with balanced category supervision."""

    geometry_name = "bhca_subcenter_sphereface2"
    accepts_species_targets = True

    def __init__(
        self,
        in_features,
        out_features,
        *,
        class_to_species,
        species_names=None,
        identity_margin=0.30,
        species_margin=0.10,
        species_weight=0.50,
        species_scale=40.0,
        num_subcenters=2,
        magn_type="A",
        alpha=0.9,
        r=40.0,
        t=3.0,
        lw=20.0,
        within_species_weight=0.0,
        within_negative_reduction=None,  # Accepted in final historical recipes.
        species_balance="sqrt_inverse_identity_frequency",
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.identity_margin = float(identity_margin)
        self.species_margin = float(species_margin)
        self.species_weight = float(species_weight)
        self.num_subcenters = int(num_subcenters)
        self.magn_type = str(magn_type).upper()
        self.alpha = float(alpha)
        self.r = float(r)
        self.t = float(t)
        self.lw = float(lw)
        self.species_scale = float(species_scale)
        self.species_balance = str(species_balance).lower()
        self.species_names = list(species_names or [])

        if self.in_features <= 0:
            raise ValueError("in_features must be a positive integer.")
        if self.out_features <= 1:
            raise ValueError("out_features must be greater than 1.")
        if not 0.0 <= self.identity_margin < math.pi / 2:
            raise ValueError("identity_margin must be in [0, pi/2).")
        if not 0.0 <= self.species_margin < math.pi / 2:
            raise ValueError("species_margin must be in [0, pi/2).")
        if float(within_species_weight) != 0.0:
            raise ValueError("ApexID Hierarchical Loss requires within_species_weight=0.")
        if self.num_subcenters <= 0:
            raise ValueError("num_subcenters must be a positive integer.")
        if self.magn_type != "A":
            raise ValueError("ApexID Hierarchical Loss uses the additive angular margin (A).")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in the open interval (0, 1).")
        if not all(math.isfinite(x) and x > 0 for x in (self.r, self.t, self.lw)):
            raise ValueError("r, t, and lw must be finite and positive.")
        if not math.isfinite(self.species_weight) or self.species_weight < 0:
            raise ValueError("species_weight must be finite and non-negative.")
        if not math.isfinite(self.species_scale) or self.species_scale <= 0:
            raise ValueError("species_scale must be greater than zero.")
        if self.species_balance != "sqrt_inverse_identity_frequency":
            raise ValueError(
                "species_balance must be 'sqrt_inverse_identity_frequency'."
            )

        mapping = torch.as_tensor(class_to_species, dtype=torch.long)
        if mapping.ndim != 1 or mapping.numel() != self.out_features:
            raise ValueError("class_to_species must have length out_features.")
        if torch.any((mapping < 0) & (mapping != IGNORE_INDEX)):
            raise ValueError("class_to_species values must be -100 or non-negative.")
        known = mapping[mapping != IGNORE_INDEX]
        if known.numel():
            unique = torch.unique(known, sorted=True)
            expected = torch.arange(int(unique[-1].item()) + 1)
            if not torch.equal(unique.cpu(), expected):
                raise ValueError("Known species indices must be contiguous from zero.")
            num_species = int(unique[-1].item()) + 1
        else:
            num_species = 0
        self.num_species = num_species
        if self.species_names and len(self.species_names) != self.num_species:
            raise ValueError("species_names must align with the species indices.")

        counts = torch.bincount(known.cpu(), minlength=num_species).long()
        if num_species:
            raw_weights = counts.float().rsqrt()
            balance_weights = raw_weights / raw_weights.mean()
        else:
            balance_weights = torch.empty(0, dtype=torch.float32)

        self.register_buffer("class_to_species", mapping, persistent=True)
        self.register_buffer("identities_per_species", counts, persistent=True)
        self.register_buffer("species_balance_weights", balance_weights, persistent=True)
        self.species_proxies = nn.Parameter(torch.empty(num_species, self.in_features))
        if self.species_proxies.numel():
            nn.init.xavier_uniform_(self.species_proxies)
        self.last_metrics = {}
        self.last_prediction_logits = None

        self.identity_weight = nn.Parameter(
            torch.empty(self.out_features, self.num_subcenters, self.in_features)
        )
        nn.init.xavier_uniform_(self.identity_weight.view(-1, self.in_features))
        self.register_buffer(
            "subcenter_selection_counts",
            torch.zeros(self.out_features, self.num_subcenters, dtype=torch.long),
            persistent=True,
        )
        bias = sphereface2_initial_bias(
            self.out_features,
            magn_type=self.magn_type,
            alpha=self.alpha,
            r=self.r,
            m=self.identity_margin,
            t=self.t,
        )
        self.b = nn.Parameter(torch.tensor([bias], dtype=torch.float32))

    def get_extra_state(self):
        return {
            "geometry_name": self.geometry_name,
            "species_names": list(self.species_names),
            "species_balance": self.species_balance,
            "objective_settings": {"species_weight": self.species_weight},
        }

    def set_extra_state(self, state):
        settings = state.get("objective_settings", {})
        if (
            settings.get("species_weight") != self.species_weight
            or settings.get("within_species_weight", 0.0) != 0.0
        ):
            raise ValueError("Checkpoint must use the same category weight and global-plus-category objective.")
        self.species_names = list(state.get("species_names", self.species_names))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Final E-series checkpoints contain these unused conditional buffers.
        state_dict.pop(prefix + "species_identity_indices", None)
        state_dict.pop(prefix + "species_identity_offsets", None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @property
    def weight(self):
        return self.identity_weight

    def _identity_scores(self, embeddings):
        weights = F.normalize(self.identity_weight, p=2, dim=2)
        all_scores = F.linear(embeddings, weights.reshape(-1, self.in_features))
        all_scores = all_scores.view(-1, self.out_features, self.num_subcenters)
        return all_scores.max(dim=2)

    def _subcenter_metrics(self, winning_centers, labels):
        rows = torch.arange(labels.numel(), device=labels.device)
        target_winners = winning_centers[rows, labels]
        if self.training:
            with torch.no_grad():
                increments = torch.zeros_like(self.subcenter_selection_counts)
                increments.index_put_(
                    (labels, target_winners),
                    torch.ones_like(labels, dtype=increments.dtype),
                    accumulate=True,
                )
                self.subcenter_selection_counts.add_(increments)
        histogram = torch.bincount(target_winners, minlength=self.num_subcenters).float()
        histogram = histogram / histogram.sum().clamp_min(1.0)
        active = self.subcenter_selection_counts > 0
        active_per_identity = active.sum(dim=1)
        totals = self.subcenter_selection_counts.sum(dim=1)
        dominance = self.subcenter_selection_counts.max(dim=1).values.float() / totals.clamp_min(1).float()
        used = totals > 0
        metrics = {
            "subcenter_fraction_ever_selected": _finite_float(active.float().mean()),
            "subcenter_average_active_per_identity": _finite_float(active_per_identity.float().mean()),
            "subcenter_collapse_dominance": _finite_float(dominance[used].mean()) if used.any() else 0.0,
        }
        for index in range(self.num_subcenters):
            metrics[f"winning_subcenter_fraction_{index}"] = _finite_float(histogram[index])
        for count in range(1, self.num_subcenters + 1):
            metrics[f"identities_using_{count}_subcenters"] = int((active_per_identity == count).sum().item())
        return metrics


    def _global_identity_loss(self, scores, labels):
        return sphereface2_loss_from_cosine(
            scores,
            labels,
            magn_type=self.magn_type,
            alpha=self.alpha,
            r=self.r,
            m=self.identity_margin,
            t=self.t,
            lw=self.lw,
            bias=self.b,
        )

    def _prediction_logits(self, scores):
        return self.r * scores + self.b


    def _species_objective(self, embeddings, species_targets):
        valid = species_targets != IGNORE_INDEX
        if self.num_species:
            species_cosine = F.linear(
                embeddings,
                F.normalize(self.species_proxies, p=2, dim=1),
            )
        else:
            species_cosine = embeddings.new_empty((embeddings.shape[0], 0))

        connected_zero = embeddings.sum() * 0.0 + self.species_proxies.sum() * 0.0
        metrics = {
            "species_valid_anchors": int(valid.sum().item()),
            "species_weight_min": _finite_float(self.species_balance_weights.min()) if self.num_species else 0.0,
            "species_weight_mean": _finite_float(self.species_balance_weights.mean()) if self.num_species else 0.0,
            "species_weight_max": _finite_float(self.species_balance_weights.max()) if self.num_species else 0.0,
        }
        if not valid.any():
            metrics.update({
                "species_accuracy": 0.0,
                "species_positive_cosine": 0.0,
                "species_hardest_wrong_cosine": 0.0,
                "species_margin_violation_rate": 0.0,
                "species_effective_contribution_min": 0.0,
                "species_effective_contribution_max": 0.0,
                "species_effective_contribution_entropy": 0.0,
            })
            return connected_zero, species_cosine, metrics

        valid_cosine = species_cosine[valid]
        valid_targets = species_targets[valid]
        logits = _arcface_logits(
            valid_cosine,
            valid_targets,
            self.species_margin,
            self.species_scale,
        )
        per_anchor = F.cross_entropy(logits, valid_targets, reduction="none")
        anchor_weights = self.species_balance_weights[valid_targets]
        loss = (per_anchor * anchor_weights).sum() / anchor_weights.sum().clamp_min(1e-12)
        row_index = torch.arange(valid_targets.numel(), device=valid_targets.device)
        positives = valid_cosine[row_index, valid_targets]
        if self.num_species > 1:
            masked = valid_cosine.clone()
            masked[row_index, valid_targets] = -torch.inf
            hardest = masked.max(dim=1).values
            target_margin_logits = logits[row_index, valid_targets] / self.species_scale
            violation = (target_margin_logits <= hardest).float().mean()
            hard_mean = hardest.mean()
        else:
            violation = positives.new_zeros(())
            hard_mean = positives.new_zeros(())

        contributions = torch.zeros(
            self.num_species, dtype=anchor_weights.dtype, device=anchor_weights.device
        )
        contributions.scatter_add_(0, valid_targets, anchor_weights)
        contributions = contributions / contributions.sum().clamp_min(1e-12)
        nonzero = contributions[contributions > 0]
        entropy = -(nonzero * nonzero.log()).sum() if nonzero.numel() else connected_zero
        metrics.update({
            "species_accuracy": _finite_float((valid_cosine.argmax(dim=1) == valid_targets).float().mean()),
            "species_positive_cosine": _finite_float(positives.mean()),
            "species_hardest_wrong_cosine": _finite_float(hard_mean),
            "species_margin_violation_rate": _finite_float(violation),
            "species_effective_contribution_min": _finite_float(contributions.min()),
            "species_effective_contribution_max": _finite_float(contributions.max()),
            "species_effective_contribution_entropy": _finite_float(entropy),
        })
        return loss, species_cosine, metrics

    def _global_metrics(self, identity_cosine, labels):
        rows = torch.arange(labels.numel(), device=labels.device)
        positives = identity_cosine[rows, labels]
        masked = identity_cosine.clone()
        masked[rows, labels] = -torch.inf
        hardest = masked.max(dim=1).values
        return {
            "identity_top1_proxy_accuracy": _finite_float(
                (identity_cosine.argmax(dim=1) == labels).float().mean()
            ),
            "identity_positive_similarity": _finite_float(positives.mean()),
            "identity_hardest_global_negative_similarity": _finite_float(hardest.mean()),
            "identity_global_similarity_gap": _finite_float((positives - hardest).mean()),
        }

    def compute_output(self, embeddings, labels, species_targets=None):
        labels = labels.long()
        if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
            raise ValueError("labels must have shape [batch_size].")
        if labels.numel() == 0 or int(labels.min()) < 0 or int(labels.max()) >= self.out_features:
            raise ValueError("Identity labels must be in [0, out_features).")
        mapped_species = self.class_to_species[labels]
        if species_targets is None:
            species_targets = mapped_species
        else:
            species_targets = species_targets.to(device=labels.device, dtype=torch.long)
            if species_targets.ndim != 1 or species_targets.shape != labels.shape:
                raise ValueError("species_targets must have shape [batch_size].")
            invalid = (species_targets < 0) & (species_targets != IGNORE_INDEX)
            out_of_range = (
                (species_targets != IGNORE_INDEX) & (species_targets >= self.num_species)
            )
            if invalid.any() or out_of_range.any():
                raise ValueError("species_targets must be -100 or a configured training species index.")
            conflict = (
                (species_targets != IGNORE_INDEX)
                & (species_targets != mapped_species)
            )
            if conflict.any():
                raise ValueError("Explicit species targets conflict with class_to_species.")
        normalized = F.normalize(embeddings, p=2, dim=1)
        identity_cosine, winning_centers = self._identity_scores(normalized)
        loss_id = self._global_identity_loss(identity_cosine, labels)
        loss_species, species_cosine, species_metrics = self._species_objective(
            normalized, species_targets
        )
        loss_species_weighted = self.species_weight * loss_species
        loss_total = loss_id + loss_species_weighted

        metrics = {
            "loss_id_global": _finite_float(loss_id),
            "loss_species": _finite_float(loss_species),
            "balanced_species_loss": _finite_float(loss_species),
            "loss_species_weighted": _finite_float(loss_species_weighted),
            "loss_total": _finite_float(loss_total),
            "species_weight": self.species_weight,
        }
        metrics.update(self._global_metrics(identity_cosine, labels))
        metrics.update(species_metrics)
        metrics.update(self._subcenter_metrics(winning_centers, labels))
        self.last_metrics = metrics
        self.last_prediction_logits = self._prediction_logits(identity_cosine).detach()
        return HierarchicalLossOutput(
            loss_total=loss_total,
            loss_id_global=loss_id,
            loss_species=loss_species,
            identity_cosine=identity_cosine,
            species_cosine=species_cosine,
            metrics=metrics,
        )

    def forward(self, embeddings, labels, species_targets=None):
        return self.compute_output(embeddings, labels, species_targets).loss_total

    def predict_logits(self, embeddings):
        normalized = F.normalize(embeddings, p=2, dim=1)
        scores, _ = self._identity_scores(normalized)
        return self._prediction_logits(scores)


__all__ = ["ApexIDHierarchicalHead", "HierarchicalLossOutput"]
