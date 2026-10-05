# BHCA-SCSF2 auxiliary weighting correction

## Problem and scope

The previous objective used `L_G + (lambda_H / 2) * (L_SP + L_W)`.
Its global SphereFace2 term summed negative penalties, whereas `L_W` averaged
sibling negatives by `n_s - 1`. With `lambda_H = 0.10`, the extra positive
coefficient was 5%, but the extra coefficient on each sibling negative was
only `0.05 / (n_s - 1)`. A category containing 1,000 identities received an
extra sibling-negative coefficient of about 0.005%.

The correction retains the original SphereFace2 logits, angular margin,
similarity transform, straight-through gradient, shared bias, and sub-centers.
It makes the auxiliary weights independent and offers summed sibling negatives.
It does not add conditional softmax, another proxy bank, or a pairwise loss.

## Corrected objective

For each eligible anchor, let `a_ij` be the existing transformed SF2 logits,
`p_i = softplus(-a_i,y_i)`, and `n_ij = softplus(a_ij)`.

`L_W_sum = (eta / (r * |V2|)) * sum_i [alpha * p_i + (1-alpha) * sum_{j in I_s, j != y_i} n_ij]`

`L_total = L_G + species_weight * L_SP + within_species_weight * L_W_sum`

The conditional loss still averages anchors, but it no longer divides each
sibling-negative penalty by the category's identity count. `L_G` is unchanged.
The same positive occurs in both identity components, which is intentional
and must be included when interpreting their effective weighting.

The updated run config uses:

```yaml
head:
  name: BHCA_SCSF2
  kwargs:
    # These replace hierarchy_weight in the corrected config.
    species_weight: 0.05
    within_species_weight: 0.50
    within_negative_reduction: sum
```

These are starting settings to evaluate, not evidence of improved retrieval.
The complete runnable config retains the existing global SF2 parameters.

When all anchors are eligible (`|V2| = B`), the coefficients relative to the
unchanged global identity component are:

| Term | Legacy preset | Corrected preset |
|---|---|---|
| Positive identity | `1.05` | `1.50` |
| Each sibling negative | `1 + 0.05 / (n_s - 1)` | `1.50` |
| Each cross-category negative | `1` | `1` |
| Category CE multiplier | `0.05` | `0.05` |

These are coefficient/score-gradient comparisons at the same logits, not
predictions of the net embedding-gradient norm or final performance. For
partially eligible batches, the conditional mean adds a `B / |V2|` factor for
eligible anchors. Missing categories retain global supervision; singleton
categories skip only the conditional loss. More active sibling negatives can
raise the aggregate conditional loss in sum mode. Monitor weighted losses and
use validation to tune the within-species coefficient.

## Independent weights and compatibility

`species_weight` and `within_species_weight` accept finite non-negative numbers,
including zero. Each omitted weight independently defaults to
`hierarchy_weight / 2`. Explicit values are direct coefficients: no further
factor of `hierarchy_weight` or `1/2` is applied. The constructor's default
`within_negative_reduction="mean"` preserves existing callers; the corrected
run config explicitly requests `"sum"`.

The original config is archived as
`configs/cluster_bhca_subcenter_sphereface2_legacy.yaml`. Reproducing a historical
checkpoint also requires its original metadata; the earlier whale-category
migration changes hierarchy membership independently of this loss correction.

New head state dictionaries record effective auxiliary weights and reduction.
Loading a checkpoint with different recorded settings raises an error. An old
checkpoint with no recorded objective settings is accepted only through legacy
weight/reduction configuration. Start a fresh run to change the objective;
loading a backbone alone remains a separate, deliberate warm-start experiment.

The corrected config has distinct experiment/checkpoint/early-stopping output
directories and W&B identifiers containing `reweighted`. Its `resume_path: auto`
therefore resumes only from that new configured location, if it already exists.

## Logging

Monitor these existing/new head metrics through the trainer:

- `loss_id_global`, `loss_species`, `loss_within_species`.
- `species_weight`, `within_species_weight`, `within_negatives_summed`.
- `loss_species_weighted`, `loss_within_species_weighted`.
- `loss_hierarchy_weighted`, `loss_total`.

`loss_hierarchy` remains the legacy unweighted average for logging compatibility;
`loss_hierarchy_weighted` is the actual auxiliary contribution to the total.

## Run and evaluate

After pulling `pretrained_training_cluster`, use the corrected preset in your
normal training environment/job:

```bash
python scripts/train.py --config configs/cluster_bhca_subcenter_sphereface2.yaml
```

The archived objective can be selected explicitly:

```bash
python scripts/train.py --config configs/cluster_bhca_subcenter_sphereface2_legacy.yaml
```

For controlled ablations, compare mean versus sum with weights held fixed,
then compare independent within-species weights `{0.10, 0.25, 0.50}` at species
weight `0.05`. Investigate species weights `{0, 0.02, 0.05}` separately. Keep
backbone initialization, data, sampling, optimizer, budget, and evaluation
protocol matched. Choose settings on development validation, not external tests.

## Verification

The focused CPU loss suite covers the old and corrected objectives, independent
weight arithmetic and gradients, exact sibling-gradient ratios at 2/99/999
siblings, unchanged cross-category coefficients, zero auxiliary weights,
missing/singleton categories, finite gradients, invalid options, checkpoint
compatibility, and construction from the updated YAML config.

```bash
pytest -q tests/losses/test_bhca.py
```

During this change, 44 tests passed using the actual loss/data modules with
package initialization isolated from unrelated backbone dependencies. This was
a focused CPU verification, not a full application test or a training benchmark.
