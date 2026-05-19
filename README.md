# ApexID

Run all commands from the repository root:

```bash
cd /home/seifelkerdany/PC/MIA/ApexID
```

## Train

Edit `configs/default.yaml` first:

```yaml
data:
  train_path: animals/fine_tune_train.csv
  val_path: animals/val_all.csv
  root_dir: animals
```

The train and validation CSV files must contain at least these columns:

```text
identity,path
```

Start training:

```bash
python scripts/train.py --config configs/default.yaml
```

## Track Experiments With Weights & Biases

Install the project requirements, then log in to W&B once on your machine:

```bash
wandb login
```

Enable tracking in `configs/default.yaml`:

```yaml
tracking:
  enabled: true
  provider: wandb
  project: ApexID
  entity: MIA-Robotics
  run_name: null
  group: null
  tags: []
  mode: online
  log_artifacts: best
```

Tips:

- Set `run_name` when you want a human-readable run name; otherwise the experiment name is used.
- Add strings to `tags` to group runs, for example `["osnet", "sphereface2"]`.
- Use `mode: offline` to test logging locally without uploading. Local W&B files are written under `wandb/`, which is ignored by Git.
- `log_artifacts: best` uploads the best model files saved by early stopping, not every epoch checkpoint.

The trainer logs epoch-level metrics such as `train_loss`, `train_acc`, `val_loss`, `val_acc`, `val_top1`, `val_top5`, `val_mAP`, `val_micro_f1`, `lr`, validation sample counts, and early-stopping state.

## Calculate Mean And Std

Calculate mean/std for the configured training CSV:

```bash
python -m apexid.utils.calculate_mean_std --config configs/default.yaml --split train
```

Calculate mean/std for the configured validation CSV:

```bash
python -m apexid.utils.calculate_mean_std --config configs/default.yaml --split val
```

Calculate mean/std and write the values back into `configs/default.yaml`:

```bash
python -m apexid.utils.calculate_mean_std --config configs/default.yaml --split train --write-config
```

Use an explicit CSV instead of the config path:

```bash
python -m apexid.utils.calculate_mean_std \
  --config configs/default.yaml \
  --csv animals/fine_tune_train.csv \
  --root-dir animals
```
