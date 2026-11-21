# Evaluation

Use the same frozen manifests, foreground images, and checkpoint-specific preprocessing for every model. External images, the private jaguar source, and trained checkpoints are separate inputs.

## Closed-set retrieval

1. Use the original prepared bundle, or run [00_prepare_benchmark.ipynb](../notebooks/00_prepare_benchmark.ipynb) with the curated inputs. Set its configuration for the ApexID corpus metadata, merged ELPephants/GCN-ID catalogue, Bristol manifest, and private jaguar manifest. This writes the canonical catalogues and observation-disjoint gallery/query manifests for seeds 42, 123, 3407, 2026, and 1337.
2. From the repository root, extract each of the twelve trained descriptors with a distinct name and the same stage output root.

   ~~~bash
   python scripts/extract_embeddings.py \
     --prepared outputs/apexid_cvpr_prepared \
     --output outputs/stage01 \
     --config /path/to/config_resolved.yaml \
     --weights /path/to/best_backbone.pt \
     --name convnext_small_apexid_hierarchical
   ~~~

   If frozen paths use another mount, add --path-map with a prefix mapping in JSON, for example {"/original/images": "/mounted/images"}. Use --image-root for relative paths. For the two original pretrained references use configs/pretrained/miewid_v3.yaml or configs/pretrained/megadescriptor_l.yaml and omit --weights.
3. Set APEXID_PREPARED_INPUT and APEXID_STAGE01_INPUT to the two artifact roots and run [02_closed_set_retrieval.ipynb](../notebooks/02_closed_set_retrieval.ipynb). It scores Rank-1, CMC through rank 50, mAP, mINP, and identity-balanced Rank-1 and mAP. It also includes within-dataset verification, embedding geometry, limited-gallery, and same-category distractor analyses. Comparison mode aligns per-query records and uses 10,000 dataset-stratified identity-block draws shared across the five seeds for paired 95% intervals. Use runs from the same model family for loss comparisons.

The original supplementary ScoreCAM generation code and per-example settings were unavailable.

## Dataset-adapted open-set analysis

Install [the optional evaluation dependencies](../requirements-evaluation.txt) alongside the core environment. WildFusion requires CUDA. The archived runs did not record the WildlifeDatasets and WildlifeTools revisions. The optional dependency file records revisions for this checkout.

~~~bash
python -m pip install -r requirements-evaluation.txt -c requirements-lock.txt
~~~

1. Run [03_open_set_roles.ipynb](../notebooks/03_open_set_roles.ipynb) to write seed-42 roles under outputs/stage03/03_splits. Set APEXID_PREPARED_INPUT to the bundle root. Set APEXID_PATH_MAP and APEXID_IMAGE_ROOT if the image mount changed.
2. Run [03a_wildfusion.ipynb](../notebooks/03a_wildfusion.ipynb) once per dataset and descriptor. Set APEXID_DATASET_NAME, APEXID_GLOBAL_DESCRIPTOR, APEXID_MODEL_CONFIG and, for a trained model, APEXID_WEIGHTS. Set APEXID_CALIBRATION_COHORT to use a recorded 512-image dataset/image-ID CSV. Otherwise the notebook derives the cohort from the frozen catalogues with seed 42. The matrix bundle records the cohort hash and role overlaps.
3. Run [03b_select_open_set_parameters.ipynb](../notebooks/03b_select_open_set_parameters.ipynb) for the descriptor. Optuna maximizes 0.6 × ARI + 0.4 × V-measure on the labelled query pool scored in the next step. Gallery-free private jaguar uses fixed defaults.
4. Run [03c_open_set_metrics.ipynb](../notebooks/03c_open_set_metrics.ipynb) for the same descriptor. It applies the selected parameters and reports clustering and known-attachment metrics. Clear APEXID_DATASET_NAME to evaluate all four datasets. Use matrix bundles from this workflow or archived matrices from a trusted source.

The archived 512-image calibration cohort includes 18 known and 15 novel query images from Bristol, 17 known and 19 novel from ELPephants, 7 known and 18 novel from GCN-ID, and 102 novel from the gallery-free jaguar set. This workflow preserves the archived dataset-adapted procedure. The supplement's statement that calibration samples are part of the gallery is incomplete for these runs.

[04_dataset_split_analysis.ipynb](../notebooks/04_dataset_split_analysis.ipynb) summarizes the training and validation corpus and the frozen external closed-set splits.
