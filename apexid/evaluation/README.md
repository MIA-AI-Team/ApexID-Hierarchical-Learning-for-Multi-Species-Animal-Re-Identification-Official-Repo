# apexid/evaluation

This package contains ApexID validation metrics for closed-set retrieval,
unknown identity discovery, known-vs-unknown rejection, mixed open-set behavior,
and embedding diagnostics.

## Three Validation CSVs

ApexID uses three validation CSVs because open-set animal re-identification has
different failure modes:

- `val_known.csv`: held-out images whose identities also exist in `train.csv`.
  This measures closed-set known identity retrieval against the train gallery.
- `val_unknown.csv`: images whose identities do not exist in `train.csv`. True
  labels are used only for evaluation, never as training classifier classes.
- `val.csv`: the concatenation of `val_known.csv` and `val_unknown.csv`. This
  measures the full mixed open-set decision: identify known animals, reject
  unknown animals, and cluster rejected unknowns.

| Group | CSV / Source | Metrics |
|---|---|---|
| Known | `val_known.csv` | mAP, Top-1, Top-5, Top-10, BAKS, CMC |
| Unknown | `val_unknown.csv` | ARI, AMI, V-measure, BAUS |
| Rejection | `val_known.csv + val_unknown.csv` | AUROC, FAR, FPR95 |
| Mixed | `val.csv` | Mixed ARI, F1, NormAcc, OSCR AUC, DIR@1%FAR, DIR@0.1%FAR |
| Diagnostics | known + unknown / gallery | Intra/inter gap, norms, alignment, uniformity, k-NN accuracy |

## Metric Groups

Known metrics show whether the model retrieves known animals correctly. mAP
measures full ranking quality, Top-k metrics show whether the correct identity is
near the top, BAKS protects against long-tail identity imbalance, and CMC reports
additional retrieval quality at larger ranks such as 20 and 50. `known/top1`,
`known/top5`, and `known/top10` already cover CMC ranks 1, 5, and 10, so those
duplicate CMC keys are not emitted.

Unknown metrics show whether unseen individuals form meaningful clusters. ARI,
AMI, and V-measure compare predicted unknown clusters with true unknown labels.
BAUS measures whether unknown individuals are rejected instead of forced into
known train identities.

Rejection metrics measure known-vs-unknown separation using each validation
image's best similarity to the train gallery as the knownness score. AUROC is
threshold-free, FAR measures how often unknown animals are wrongly accepted as
known, and FPR95 measures false accepts when known recall is forced to stay high.

Mixed metrics evaluate the final open-set system. Mixed ARI measures final
identity grouping across known and unknown samples. F1 measures known-vs-unknown
decision balance. NormAcc is `sqrt(BAKS * BAUS)`, which discourages trivial
always-known or always-unknown behavior. OSCR AUC jointly evaluates known
identification and unknown rejection. DIR@FAR reports identification quality
under strict false-accept limits.

Diagnostics run every 5 epochs by default to reduce validation cost. The
intra/inter gap checks whether same identities are closer than different
identities. Embedding norms detect collapse or instability. Alignment measures
whether positive pairs are close. Uniformity detects whether embeddings are well
distributed on the hypersphere. k-NN accuracy tests whether frozen embeddings are
directly useful for retrieval.

Large train galleries make exact open-set metrics expensive because gallery
matching, mAP, and clustering scale with the number of validation images. The
default config keeps `evaluation.sample_limits` set to `null`, so epoch-time
validation uses the full splits. When `val.csv` is exactly
`val_known.csv + val_unknown.csv`, the validator reuses those embeddings for the
mixed split and reuses gallery-match scores for mixed metrics. You can set
`evaluation.sample_limits` to positive integers for faster approximate
identity-balanced validation.

The heaviest exact metrics are `known/mAP` and CMC/top-k retrieval against the
train gallery, rejection and mixed knownness scores, mixed threshold selection,
and threshold-connected clustering for unknown/rejected samples. The evaluator
keeps those metrics exact by default while avoiding repeated gallery scoring
where the known, unknown, and mixed splits overlap.

## W&B Metric Names

Known:

```text
known/mAP
known/top1
known/top5
known/top10
known/BAKS
known/cmc_rank20
known/cmc_rank50
```

Unknown:

```text
unknown/ARI
unknown/AMI
unknown/v_measure
unknown/BAUS
```

Rejection:

```text
rejection/AUROC
rejection/FAR
rejection/FPR95
rejection/threshold_at_95_tpr
rejection/known_tpr_at_threshold
rejection/unknown_recall_at_threshold
```

Mixed:

```text
mixed/ARI
mixed/F1
mixed/NormAcc
mixed/BAKS
mixed/BAUS
mixed/OSCR_AUC
mixed/DIR_at_1FAR
mixed/DIR_at_0_1FAR
mixed/known_unknown_threshold
```

Diagnostics:

```text
validation/diagnostics_ran
validation/mixed_embeddings_reused
validation/mixed_scores_reused
validation/metrics_sampled
validation/gallery_samples_used
validation/gallery_samples_total
validation/known_samples_used
validation/known_samples_total
validation/unknown_samples_used
validation/unknown_samples_total
validation/mixed_samples_used
validation/mixed_samples_total
diagnostics/intra_similarity
diagnostics/inter_similarity
diagnostics/intra_inter_gap
diagnostics/embedding_norm_mean
diagnostics/embedding_norm_std
diagnostics/embedding_norm_min
diagnostics/embedding_norm_max
diagnostics/alignment
diagnostics/uniformity
diagnostics/knn_top1
diagnostics/knn_top5
```

Legacy aliases are kept for history and older callbacks: `val_mAP` maps to
`known/mAP`, `val_top1` maps to `known/top1`, `val_top5` maps to `known/top5`,
and `val_micro_f1` maps to `mixed/F1`.

## Common Commands

Run evaluation tests:

```bash
pytest tests/evaluation
```

Use the open-set evaluator directly:

```python
from apexid.evaluation import evaluate_all_validation_splits

metrics = evaluate_all_validation_splits(
    gallery_embeddings,
    gallery_labels,
    known_embeddings,
    known_labels,
    unknown_embeddings,
    unknown_labels,
    mixed_embeddings,
    mixed_labels,
    mixed_cls_labels=mixed_cls_labels,
    config=evaluation_cfg,
    epoch_index=4,
)
```
