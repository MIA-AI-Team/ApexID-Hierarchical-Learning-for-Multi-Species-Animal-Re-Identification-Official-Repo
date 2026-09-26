# Balanced Hierarchical Category–Animal identity losses (BHCA)

## Purpose and scope

BHCA uses the observed training hierarchy `species/category → individual identity`
to organize a normalized ApexID embedding while keeping global identity
discrimination primary. It adds three controlled heads:

- **BHCA-A** (`BHCAArcFace`): one ArcFace proxy per identity.
- **BHCA-SCA** (`BHCASubCenterArcFace`): a fixed number of Sub-center ArcFace
  proxies per identity.
- **BHCA-SCSF2** (`BHCASubCenterSphereFace2`): the same fixed sub-centers,
  collapsed by an exact maximum and consumed by the repository's exact
  SphereFace2 transform.

All variants use exactly the same balanced species and conditional sibling
methodology. BHCA does **not** reserve literal fixed regions or dimensions of the
hypersphere. Its proxy objectives encourage coarse category organization and
fine identity discrimination; the resulting geometry remains an empirical
question.

## Why balancing is based on identities

The controlled sampler draws `P=16` identities uniformly and `K=4` images for
each selected identity. The identity objective is therefore already balanced at
the selected-identity level. Raw image-frequency weighting would partially
correct the same imbalance twice and is not used.

Under identity-uniform sampling, a species is encountered approximately in
proportion to its number of training identities. If

`N_s^{ID} = |{y : g(y) = s}|`

BHCA deterministically uses

`ω_s = (1 / sqrt(N_s^{ID})) / ((1/S) * sum_{r=1}^{S} 1 / sqrt(N_r^{ID}))`

Thus `mean(ω) = 1`, rare categories receive more weight than common categories,
and there is no effective-number beta to tune. Counts come only from the
training identity-to-species mapping. Validation/test rows, validation/test
label frequencies, and raw training image counts never enter these weights.

## Notation and complete objective

For normalized backbone output

`z_i = f(x_i) / ||f(x_i)||_2`

let `y_i` be identity, `s_i` its explicit species/category, `C` the training
identity count, and `S` the training category count. Species proxies and identity
proxies are normalized when scored:

`c^{SP}_{i,s} = z_i^T hat{p}_s`

`c^{ID}_{i,j} = z_i^T hat{w}_j`

The exact implemented objective is

`L_BH = (1/2) * (L_SP + L_within)`

`L_BHCA = L_ID + λ_H * L_BH`

There is no learned or configurable coefficient between `L_SP` and
`L_within`; their fixed average leaves `λ_H` as the sole BHCA hierarchy weight.

## Shared species/category branch

For a valid target, the category target score receives an additive angular
margin:

`φ^{SP}_{i,s_i} = cos(arccos(c^{SP}_{i,s_i}) + m_SP)`

with the repository's stable ArcFace monotonicity correction. Other category
scores remain unmodified. At scale `γ` (the identity scale by default), let
`CE_i^SP` be cross-entropy over all training species. The reduction is

`L_SP = (sum_{i in V} ω_{s_i} * CE_i^{SP}) / (sum_{i in V} ω_{s_i})`

where `V` contains only rows with an explicit known training-category target.
This denominator prevents category composition from changing the global scale
merely through the sum of weights.

## Shared conditional within-species branch

For each valid anchor, BHCA obtains the complete global set

`C(s_i) = {j : g(j) = s_i}`

from persisted training-state buffers. It is not limited to identities present
in the minibatch. Anchors are grouped by species and the corresponding columns
are selected from the same global identity score matrix. No second identity
classifier or hierarchy-specific identity bank exists.

For ArcFace geometries, the implemented term is

`L_within = (1 / |V_2|) * sum_{i in V_2} -log( exp[s * φ(c^{ID}_{i,y_i}, m_ID)] / sum_{j in C(s_i)} exp[s * q_{i,j}] )`

where `q` is the margin-adjusted target score and the unchanged non-target
score, and `V_2` includes categories with at least two identities. A singleton
category has `L_within = 0` for its anchors; category and global identity losses
remain active. Negatives are never fabricated.

## Identity geometries

### BHCA-A

There is one bank `W ∈ R^(C×D)`. Global identity loss is ordinary ArcFace over
all `C` identities, using `m_ID` and scale `s`. Conditional scores select the
true species' columns from this same bank and apply the same ArcFace transform.
Cross-species identities remain in the global objective even though the focused
conditional objective excludes them.

### BHCA-SCA

There are `K_c` fixed sub-centers per identity:

`c_{i,j,k} = z_i^T hat{w}_{j,k}`

`c_{i,j} = max_k c_{i,j,k}`

The hard maximum is exact and gradients flow through the winning center. Global
and conditional ArcFace both use the same collapsed `c_(i,j)` and the same
`m_ID`. The default is `K_c = 2`; center count is not identity-dependent.

### BHCA-SCSF2

The identical fixed-K maximum is passed into ApexID's exact SphereFace2
transformation. It is not a sum of Sub-center ArcFace and SphereFace2, and it has
one identity bank only. For each cosine `c`, the implementation preserves

`g(c) = 2 * ((c + 1) / 2)^t - 1`

the `A`, `C`, or `M` margin transform, detached margin delta, shared bias,
positive/negative `alpha` weighting, scale `r`, and loss factor `lw`. The global
objective is exactly the existing one-vs-all mean BCE, including its
`lw * C / r` element weights.

For a conditional candidate set of size `n > 1`, the exact transformed logits
`a_(i,j)` are reused, but sibling negatives are normalized per anchor:

`L_i^{within} = (lw / r) * [ α * softplus(-a_{i,y_i}) + ((1 - α) / (n - 1)) * sum_{j ≠ y_i} softplus(a_{i,j}) ]`

`L_within` is the mean over valid anchors. This retains the SphereFace2 decision
geometry and balancing semantics without letting a species dominate solely
because it has more sibling proxies. With `K_c = 1`, equivalent weights and bias,
and `λ_H = 0`, tests require numerical equivalence to ordinary SphereFace2.

## Missing labels and mapping validation

`-100` is the independent missing-category contract. A BHCA dataset returns an
extra per-row category target only for BHCA runs; existing heads retain the
original three-item dataset interface. If a row has a valid identity and missing
category, it still contributes to global `L_ID`, while its `L_SP` and
`L_within` contributions are connected zeros. Explicit targets that conflict
with the persisted identity-to-species mapping are rejected.

Before any BHCA dataset is constructed, the training entry point runs one
deterministic metadata-enrichment pass over each configured split. It fills only
missing `species` values using these explicitly approved dataset facts:

| Dataset | Canonical `species` value |
|---|---|
| `CzechLynx` | `lynx` |
| `SalamanderID2025` | `salamander` |
| `humpback-whale-identification` | `humpback-whale` |

`HappyWhale` is multi-species. Missing HappyWhale `species` values are filled from
`data/og happy ids/train.csv` (matched by image basename; identity must equal
`individual_id`) and collapsed to exactly two labels: if the official species
string contains `dolphin` (or the competition typo `dolpin`), the label is
`dolphin`; otherwise it is `whale`. Fine-grained names such as `killer_whale`
are not kept.

Existing non-null species values for every other dataset are preserved exactly,
so mixed datasets such as `GiraffeZebraID` keep their row-level `giraffe` and
`zebra` labels. A covered single-species dataset with a conflicting non-null
value is an error, and any null left in an unmapped dataset stops BHCA training
with a per-dataset count. The pass asserts that row membership, order, and every
non-species column are unchanged. Training normally uses the enriched tables in
memory and does not rewrite the frozen split CSVs.

The identity-to-species mapping, counts, and balancing weights are then built
from the enriched **training split only**. Validation enrichment supports
reporting but its identities/frequencies never enter the training mapping,
weights, or proxies. No identity-string parsing is used. One identity with
conflicting categories is an error. Mapping, counts, weights, grouped identity
indices, proxies, sub-centers, SphereFace2 bias, and sub-center usage counters
are checkpointed in the head state. The generic `-100` masking path remains in
the loss for other callers, although the controlled BHCA training configs now
require complete metadata at preflight.

## Unseen identities/categories and inference

Species proxies are closed-set training-only parameters. The species head cannot
classify an unseen category and is never used as an inference gate. Export and
retrieval remain backbone-only:

`z = f(x) / ||f(x)||_2`

followed by cosine similarity. An unseen species is not forced into a known
category before retrieval. BHCA is intended to transfer organization learned
from training supervision; whether it does so better is experimental.

## Diagnostics

`bhca_species_enrichment.json` records, for every configured split, rows before
and after enrichment, fill counts by approved dataset, unresolved datasets, and
the canonical mapping. `bhca_training_diagnostics.json` records training-only
`C`, `S`, the complete
mapping and species names, identities/category, min/median/max counts,
mean-normalized weights and min/mean/max weight, known/missing parent counts and
percentage, and singleton-category count.

The trainer logs the following scalar head metrics (also under `train/…`):

- global: `loss_id_global`, `identity_top1_proxy_accuracy`,
  `identity_positive_similarity`, `identity_hardest_global_negative_similarity`,
  `identity_global_similarity_gap`;
- category: `loss_species`, `balanced_species_loss`, `species_accuracy`, positive
  and hardest-wrong cosine, margin violation rate, static weight range, and
  effective per-batch contribution min/max/entropy;
- conditional: `loss_within_species`, candidate mean/min/median/max, true and
  hardest sibling cosine, cosine gap, violation rate, valid anchors, and skipped
  singleton anchors;
- total: `loss_hierarchy`, `loss_total`, `hierarchy_weight`;
- sub-centers: batch winning-center fractions, fraction ever selected, mean
  active centers/identity, identities using each center count, and mean dominant
  center fraction (`subcenter_collapse_dominance`).

Validation diagnostics, never training targets, report sampled distributions
for `same_identity`, `different_identity_same_species`, and `different_species`
with count/mean/std/5th percentile/median/95th percentile. They report

`Δ_ID = E[S_sameID] - E[S_diffID,sameSP]`

`Δ_SP = E[S_diffID,sameSP] - E[S_diffSP]`

plus per-query hardest same-species negatives, hardest available positives,
their gap distribution, and tail violation rate. Computation is chunked and runs
on the configured diagnostics schedule.

## Complexity, memory, and parameter counts

Let batch size be `B`. BHCA-A uses `C*D + S*D` proxy parameters, the same proxy
parameter count as an independent ArcFace identity+species HA head. Its score
storage is `O(B(C+S))`; grouped conditional CE selects views/slices from the
already computed global identity matrix and adds no second full proxy bank.

BHCA-SCA uses `K_c*C*D + S*D` proxy parameters. BHCA-SCSF2 adds one scalar bias.
Their exact global sub-center score tensor costs `O(B*C*K_c)` transient memory,
then collapses to `O(B*C)`. Species scores cost `O(B*S)`. Conditional work for a
group is `O(B_s * N_s^{ID})` and reuses collapsed scores. There is no `B×C×K_c`
hierarchy duplicate. Peak memory should still be benchmarked with the real `C`,
`D=2048`, mixed precision setting, and device allocator because implementation
correctness does not substitute for a hardware measurement.

Relative to HA, BHCA-A adds conditional computation but no proxy parameters.
Sub-center variants add `(K_c-1)*C*D` identity parameters and the corresponding
global score memory. Checkpoints are therefore larger by the same formulas.

For the repository's current local enriched Split V2 training table
(`C=11,677`, `S=17`, `D=2,048`, `B=64`, float32 proxy parameters), the formulas
give 23,949,312 proxy parameters / 91.36 MiB for BHCA-A, 47,863,808 / 182.59 MiB
for BHCA-SCA at `K_c=2`, and one additional scalar for BHCA-SCSF2. Raw forward score tensors
are 2.85 MiB for `B×C`, 0.0042 MiB for `B×S`, and 5.70 MiB for `B×C×2`.
The sub-center maximum also returns a 2.85 MiB value tensor and a 5.70 MiB int64
winner-index tensor. These are tensor-size accounting results, not a fabricated
GPU allocator benchmark: this checkout has no CUDA device. Actual peak allocated
and reserved GPU memory (including gradients, optimizer state, autocast caches,
and the backbone) must be recorded on the training H100.

## Defaults and controlled configs

| Variant | `m_ID` | `m_SP` | `λ_H` | scale | `K_c` | Other geometry |
|---|---:|---:|---:|---:|---:|---|
| BHCA-A | 0.30 | 0.10 | 0.10 | 40 | 1 | ArcFace |
| BHCA-SCA | 0.30 | 0.10 | 0.10 | 40 | 2 | ArcFace |
| BHCA-SCSF2 | 0.30 | 0.10 | 0.10 | `r=40` | 2 | A, `alpha=.9`, `t=3`, `lw=20` |

The three `configs/cluster_bhca_*.yaml` files share Split V2 paths, ConvNeXt
Small pretrained backbone, 2048-dimensional embedding, preprocessing, PK
sampling, optimizer/scheduler, finetuning, validation, seed, epochs, and early
stopping. Only identity geometry and run/output identifiers differ. The PK
sampler itself is unchanged.

Suggested small validation-only sensitivity study: `λ_H ∈ {0.025, .05, .10,
.20}`, optionally `m_SP ∈ {.05, .10, .15}`, and `K_c ∈ {1,2,3}`. Keep `m_ID`
fixed initially and do not tune on final-test labels.

## Controlled ablation plan

| Model | Identity proxy | Identity objective | Hierarchy |
|---|---:|---|---|
| ArcFace baseline | 1 | ArcFace | none |
| HA | 1 | ArcFace | independent species ArcFace |
| BHCA-A | 1 | ArcFace | balanced + conditional |
| BHCA-SCA | K | Sub-center ArcFace | balanced + conditional |
| SphereFace2 | 1 | SphereFace2 | none |
| BHCA-SCSF2 | K | Sub-center SphereFace2 | balanced + conditional |

Primary comparisons are HA vs BHCA-A, BHCA-A vs BHCA-SCA, SphereFace2 vs
BHCA-SCSF2, and BHCA-SCA vs BHCA-SCSF2. Do not change unrelated variables.
Selection should prioritize Rank-1/5, mAP and identity-balanced retrieval;
AUROC/AUPR/FPR95/OSCR/DIR@FAR; and B³/ARI/AMI/NMI, fragmentation, merged-cluster
fraction, and cluster-count error. Cross-identity merging is the key failure mode,
but a reduction must not be purchased through excessive fragmentation.

## Commands

BHCA training performs the species preflight automatically. To audit a split
without writing it:

```bash
python scripts/prepare_bhca_species_metadata.py \
  --input open_set_splits_v2/train.csv \
  --check-only \
  --report outputs/bhca_species_preflight.json
```

To create a separate enriched CSV, use `--output`; use `--in-place` only when an
intentional atomic replacement of the source CSV is desired. Neither mode adds,
removes, reorders, or changes the split membership of rows.

```bash
python scripts/train.py --config configs/cluster_bhca_arcface.yaml
python scripts/train.py --config configs/cluster_bhca_subcenter_arcface.yaml
python scripts/train.py --config configs/cluster_bhca_subcenter_sphereface2.yaml
```

Backbone-only evaluation example (replace paths for each run):

```bash
python -m apexid.evaluation.evaluate_apexid_model \
  --weights outputs/apexid_convnext_small_bhca_a_v2_gray3/checkpoints/best_backbone.pt \
  --model-config configs/cluster_bhca_arcface.yaml \
  --benchmark-dir open_set_splits_v2 \
  --data-root /path/to/apexid-dataset \
  --output-dir outputs/bhca_a_evaluation \
  --evaluation-mode both
```

Use the corresponding weights/config for BHCA-SCA and BHCA-SCSF2.

## Expected failure cases and scientific status

Potential failures include noisy or overly broad category metadata, species with
one identity (no conditional signal), extreme numbers of identities in one
category, unused/collapsed sub-centers, excessive hierarchy weight that harms
global identity geometry, and memory pressure from `B*C*K_c`. Closed-set species
accuracy can improve without open-set representation quality improving.

Supported implementation claims are: training-only identity-frequency weights;
mean-1 square-root balancing; one shared identity bank; global plus conditional
identity discrimination; exact maximum sub-center collapse; exact reused
SphereFace2 transformation; independent `-100` masking; persisted hierarchy
state; and backbone-only inference.

Still-unproven hypotheses are: better retrieval, open-set recognition, unseen
category transfer, cluster discovery, lower merged-cluster fraction, useful
coarse-to-fine geometry, and superiority of any of the three identity decision
geometries. Improved performance must be demonstrated experimentally and must
not be claimed from implementation correctness alone.
