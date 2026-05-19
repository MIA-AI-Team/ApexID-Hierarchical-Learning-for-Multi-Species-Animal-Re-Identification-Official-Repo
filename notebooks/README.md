# notebooks

This folder contains exploratory notebooks for running or demonstrating ApexID
workflows.

## Main Files

- `Train_APEXID_sample.ipynb`: sample notebook for training ApexID from an
  interactive environment.

## How It Connects

Notebooks should use the same package modules, configs, and scripts as the rest
of the repository. They are useful for exploration, but reusable logic should be
moved into `apexid/` or `scripts/` so it can be tested and documented.

## Common Commands

Install dependencies before opening notebooks:

```bash
pip install -r requirements.txt
```

Run the core training flow from the terminal when you want a reproducible script
equivalent:

```bash
python scripts/train.py --config configs/default.yaml
```

Run tests after moving notebook logic into source modules:

```bash
pytest
```

## Contributor Notes

- Keep notebooks lightweight and reproducible. Avoid storing large generated
  outputs in notebook cells.
- Prefer relative paths from the repository root.
- When notebook code becomes part of the project workflow, move it into a Python
  module or script and add tests.
