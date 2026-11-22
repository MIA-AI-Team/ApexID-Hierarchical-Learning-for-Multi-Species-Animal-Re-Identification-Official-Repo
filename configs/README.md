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

To resume a run, point `checkpoint.resume_path` at its `last_checkpoint.pt` and keep its `config_resolved.yaml` and training inputs. Frozen inference only needs the matching encoder architecture and preprocessing.
