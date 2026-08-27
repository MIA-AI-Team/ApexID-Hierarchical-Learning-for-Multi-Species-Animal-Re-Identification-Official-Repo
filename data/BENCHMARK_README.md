# ApexID open-world benchmark builder

Build the benchmark from the repository root:

```bash
python data/build_apexid_benchmark.py \
  --input merged.csv \
  --output-dir apexid_benchmark \
  --seed 42 \
  --dev-novel-fraction 0.30 \
  --gallery-encounter-fraction 0.60 \
  --num-stream-orders 10
```

The builder writes `open20`, `open40`, and `open60`.  They target 20%, 40%,
and 60% novel identities respectively for the three reference datasets.
Targets are intentionally relaxed whenever exact rounding would violate an
identity or encounter leakage rule.

## Files and access boundary

Public inference inputs are `database.csv`, `development.csv`,
`final_test.csv`, `final_test_no_species.csv`, `prediction_template.csv`, and
the files under `streams/`.  They never expose original or true identity labels.

Private evaluator-only files are `development_ground_truth.csv`,
`final_test_ground_truth.csv`, `private_identity_mapping.csv`, and
`split_manifest.csv`.  Do not provide them to an inference process.

`known_identity_registry.csv` describes the canonical database IDs.  The JSON
and CSV summaries, `config.json`, and `validation_report.json` record exactly
how the benchmark was produced and which leakage checks passed.

## Encounter and split policy

SPAC-Wild encounters use the camera token plus calendar date.  Bristol groups
numbered sessions and trial sequences.  Elephant encounter dates are parsed
from filename date tokens; undated images of an identity are deliberately
treated as one `unknown_date` encounter.  Parser failures use identity-scoped
fallback groups and are reported as warnings.

Known identities require at least two encounter groups.  Their gallery,
development, and final-query images are assigned only as complete encounter
groups.  Novel identities are selected at identity level and are disjoint
between development and final test.  `gcn_id` is always discovery-only final
test data.

## Batch and online evaluation

For batch evaluation all final-test images may be processed together.  For
online evaluation process a stream in `arrival_index` order; a system may use
the static database and knowledge from preceding queries only.  The ten stream
orders change arrival order but never test membership.
