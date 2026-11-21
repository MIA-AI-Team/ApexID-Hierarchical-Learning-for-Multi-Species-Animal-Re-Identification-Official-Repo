# Paper configurations

Run commands from the repository root. The default preset trains ConvNeXt-Small with ApexID Hierarchical Loss.

```bash
export APEXID_DATA_ROOT=/path/to/segmented-images
python scripts/train.py --config configs/default.yaml
```

Set `APEXID_SPLIT_ROOT` to use another directory of training and validation CSVs. The default is `open_set_splits_v2/`. Set `APEXID_OUTPUT_ROOT` to change the default `outputs/` directory. Each experiment has its own run directory, resolved config, and checkpoint files. The training loader resolves `extends` relative to that config file.

| Encoder | ArcFace | Dynamic ArcFace | SphereFace2 | ApexID Hierarchical Loss |
| --- | --- | --- | --- | --- |
| ConvNeXt-Small | [AF](convnext_small/arcface.yaml) | [DAF](convnext_small/dynamic_arcface.yaml) | [SF2](convnext_small/sphereface2.yaml) | [ApexID](default.yaml) |
| MIEW-ID v3 | [AF](miewid_v3/arcface.yaml) | [DAF](miewid_v3/dynamic_arcface.yaml) | [SF2](miewid_v3/sphereface2.yaml) | [ApexID](miewid_v3/apexid_hierarchical.yaml) |
| MegaDescriptor-L | [AF](megadescriptor_l/arcface.yaml) | [DAF](megadescriptor_l/dynamic_arcface.yaml) | [SF2](megadescriptor_l/sphereface2.yaml) | [ApexID](megadescriptor_l/apexid_hierarchical.yaml) |

The inference-only configs under [`pretrained/`](pretrained) select the pinned original MIEW-ID v3 and MegaDescriptor-L provider weights. Fine-tuned evaluation uses each released configuration with its corresponding saved encoder checkpoint.

The [category-weight sweep](ablations) changes only the category coefficient and output directory relative to the ConvNeXt-Small ApexID preset. Its 0.50 endpoint is the default recipe. The ConvNeXt-Small SphereFace2 preset is the paper's reference without category supervision.

## Historical run mapping

The matrix follows the archived `cluster_{convnext,miewid,megadescriptor}_{arcface,dynamic_arcface,sphereface2}_v2_rgb.yaml` presets for the baselines. The ApexID Hierarchical runs correspond to the E6 configs listed below. They use the global identity term, the balanced category term with weight 0.50, and two sub-centers per identity.

| Encoder | Historical ApexID run directory | PK batch |
| --- | --- | --- |
| ConvNeXt-Small | `apexid_convnext_small_bhca_scsf2_ablation_e6_v2_rgb` | 16 × 4 |
| MIEW-ID v3 | `apexid_miewid_msv3_bhca_scsf2_ablation_e6_v2_rgb` | 8 × 4 |
| MegaDescriptor-L | `apexid_megadescriptor_l_384_bhca_scsf2_ablation_e6_v2_rgb` | 4 × 4 |

The six smaller ConvNeXt-Small category weights correspond to historical E1 (0.025), A1 (0.05), E2 (0.075), E3 (0.10), E4 (0.15), and E5 (0.25). The released sweep uses the same final-epoch validation cadence across weights. A1's archived config scheduled intermediate validation.

The public training presets follow Appendix B with a 150-epoch schedule, gradient-norm clipping at 5.0, and a five-epoch linear warmup from 1e-6 for both parameter groups. The archived ConvNeXt-Small YAMLs left clipping unset, and the historical scheduler applied the same relative warmup factor to both groups. These are paper-facing training presets. To resume a historical run, use a copy of its `config_resolved.yaml`, restore the original training inputs, and set `checkpoint.resume_path` to its `last_checkpoint.pt`. The checkpoint only needs matching encoder architecture and preprocessing for frozen inference.
