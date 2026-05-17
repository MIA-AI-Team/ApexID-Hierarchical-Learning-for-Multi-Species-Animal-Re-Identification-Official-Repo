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
