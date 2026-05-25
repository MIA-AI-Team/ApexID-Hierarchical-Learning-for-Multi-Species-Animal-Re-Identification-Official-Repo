# ApexID — HPI / AISC cluster runbook

End-to-end commands to regenerate [open-set splits](data/README.md), compute dataset mean/std, and train on **1× H100** using SLURM on the HPI Scientific Compute / AISC cluster.

Official docs: [SCI](https://docs.sc.hpi.de/), [AISC](https://aisc.hpi.de/doc/), [VPN](https://docs.sc.hpi.de/VPN/), [SLURM basics](https://docs.sc.hpi.de/cluster/SLURM/Basics/).

---

## Paths used on the cluster

| What | Path |
|------|------|
| Login node | `lx01` (hostname after SSH) |
| Your home | `/sc/home/<username>/` |
| ApexID repo (example) | `~/ApexID` |
| Shared dataset root (`DATA_ROOT`) | `/sc/projects/sci-aisc/apex-id/extracted/apexid-dataset` |
| Shared split CSVs (read-only for most users) | `…/apexid-dataset/open_set_splits/` |
| Your writable split CSVs | `~/ApexID/open_set_splits/` |
| Training config | `~/ApexID/configs/cluster_home.yaml` |
| SLURM logs | `~/logs/` |
| Checkpoints / outputs | `~/ApexID/outputs/apexid_osnet_arcface_64_0.5/` |

**Images** always load from shared `DATA_ROOT`. **Split CSVs** go in your home if you cannot write to project storage.

---

## SLURM scripts in this repo

| Script | Resource | Purpose |
|--------|----------|---------|
| [`slurm_make_splits.sh`](slurm_make_splits.sh) | CPU (4 cores, 16G) | Regenerate open-set split CSVs |
| [`slurm_mean_std.sh`](slurm_mean_std.sh) | CPU (8 cores, 32G) | Compute mean/std, write to config |
| [`slurm_train_h100.sh`](slurm_train_h100.sh) | **1× H100** (64G) | Install deps + run `scripts/train.py` |

One-time `uv` install (not in repo; copy from `hpi-aisc-gpu-skill/assets/install_uv_slurm.sh` to `~/install_uv_slurm.sh`).

---

## Non-negotiable cluster rules

- **`lx01` is for light admin only**: `ssh`, `ls`, `cd`, `sbatch`, `squeue`, `sacct`, `scancel`, `tail`.
- **Never** run training, `pip install`, or long Python jobs directly on the login node.
- Use **`uv`**, not bare `python`/`pip`, inside SLURM jobs.
- Request **`#SBATCH --constraint=ARCH:X86`** for jobs that install wheels.
- **`scp` from inside an SSH session on lx01 is blocked.** Upload files from your **local laptop** (see step 8) or use base64 paste (step 9).

---

## Step-by-step commands

### 1. Connect VPN (local laptop)

Connect your HPI OpenVPN profile **before** SSH.

```bash
pgrep -af openvpn    # Linux/macOS — optional check
```

### 2. SSH to the login node (local laptop)

Merge a `Host hpi-cluster` block into `~/.ssh/config`, or pass a config file explicitly.

**Windows PowerShell example:**

```powershell
ssh -F "C:\Users\asus\Desktop\WILDLIFE\GPU_Cluster_Run\hpi-cluster.ssh-config" hpi-cluster
```

**Verify you landed on the login node:**

```bash
hostname    # expect lx01
pwd         # expect /sc/home/<username>
```

### 3. Inspect home and enter the ApexID repo

```bash
ls
cd ~/ApexID && pwd && ls -la
```

You should see `scripts/`, `configs/`, `data/`, `requirements.txt`.

### 4. Confirm training config points at the shared dataset

```bash
grep -A 8 "^data:" configs/default.yaml
```

Expected: paths under `/sc/projects/sci-aisc/apex-id/extracted/apexid-dataset/`.

### 5. Verify the shared dataset exists (read-only check)

```bash
ls /sc/projects/sci-aisc/apex-id/extracted/apexid-dataset/metadata.csv
ls /sc/projects/sci-aisc/apex-id/extracted/apexid-dataset/open_set_splits/
```

### 6. Check whether you can overwrite shared split CSVs

```bash
test -w /sc/projects/sci-aisc/apex-id/extracted/apexid-dataset/open_set_splits \
  && echo "WRITABLE" || echo "NOT WRITABLE - use \$HOME instead"
```

If **NOT WRITABLE** (typical), splits and config must use `~/ApexID/open_set_splits/` (steps 7 and 12).

### 7. Create your personal splits output directory

```bash
mkdir -p ~/ApexID/open_set_splits && ls -la ~/ApexID/open_set_splits
mkdir -p ~/logs
```

### 8. One-time: install `uv` on a compute node

Copy `install_uv_slurm.sh` to `~/install_uv_slurm.sh` (from `hpi-aisc-gpu-skill/assets/`), then:

```bash
test -x ~/.local/bin/uv && ~/.local/bin/uv --version || echo "uv NOT installed"
```

If not installed:

```bash
sbatch --parsable ~/install_uv_slurm.sh
```

Wait for completion (replace `JOB_ID`):

```bash
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode -n --parsable2
tail -20 ~/logs/uv_install_JOB_ID.out
```

Expect `COMPLETED` and `uv 0.x.x`.

### 9. Upload SLURM scripts to the cluster

The three scripts live in this repo: `slurm_make_splits.sh`, `slurm_mean_std.sh`, `slurm_train_h100.sh`.

#### Option A — `scp` from your **local laptop** (recommended)

Open a **new** PowerShell window (not the SSH session):

```powershell
$CFG = "C:\Users\asus\Desktop\WILDLIFE\GPU_Cluster_Run\hpi-cluster.ssh-config"
$REPO = "C:\Users\asus\Desktop\WILDLIFE\GPU_Cluster_Run\ApexID"

scp -F $CFG "$REPO\slurm_make_splits.sh"  hpi-cluster:~/ApexID/slurm_make_splits.sh
scp -F $CFG "$REPO\slurm_mean_std.sh"     hpi-cluster:~/ApexID/slurm_mean_std.sh
scp -F $CFG "$REPO\slurm_train_h100.sh"    hpi-cluster:~/ApexID/slurm_train_h100.sh
```

Then on the cluster:

```bash
chmod +x ~/ApexID/slurm_*.sh
ls -la ~/ApexID/slurm_*.sh
```

#### Option B — base64 paste inside SSH

If `scp` from the laptop is unavailable, encode locally and paste on the cluster:

```bash
# On local machine (PowerShell):
python -c "import base64; p=r'PATH\\TO\\slurm_train_h100.sh'; print(base64.b64encode(open(p,'rb').read()).decode())"
```

```bash
# On cluster (paste full base64 string):
printf '%s' 'BASE64_STRING_HERE' | base64 -d > ~/ApexID/slurm_train_h100.sh
chmod +x ~/ApexID/slurm_train_h100.sh
```

Repeat for each script. Verify line count:

```bash
wc -l ~/ApexID/slurm_*.sh
```

### 10. Regenerate open-set splits (SLURM, CPU)

```bash
SPLIT_JOB=$(sbatch --parsable ~/ApexID/slurm_make_splits.sh)
echo "split job: $SPLIT_JOB"
```

Monitor:

```bash
squeue -j SPLIT_JOB -h -o '%T %M %R'
sacct -j SPLIT_JOB --format=JobID,State,Elapsed,ExitCode -n --parsable2
tail -30 ~/logs/apexid_splits_SPLIT_JOB.out
```

Verify output CSVs:

```bash
ls ~/ApexID/open_set_splits/
```

Expected: `train.csv`, `val_known.csv`, `val_unknown.csv`, `val.csv`, etc.

### 11. Create cluster config (home splits + shared images)

`train.py` only accepts `--config`; there are no CLI path overrides. Copy `default.yaml` and repoint split paths to your home directory:

```bash
cd ~/ApexID
cp configs/default.yaml configs/cluster_home.yaml
sed -i "s|/sc/projects/sci-aisc/apex-id/extracted/apexid-dataset/open_set_splits|/sc/home/$(whoami)/ApexID/open_set_splits|g" configs/cluster_home.yaml
grep -A 10 "^data:" configs/cluster_home.yaml
```

- **`data.root_dir`** stays on shared storage (images).
- **`data.train_path` / `val_*_path`** point to `~/ApexID/open_set_splits/`.
- **`experiment.output_dir`** stays under the repo (writable).

### 12. Submit mean/std (CPU) then training (H100)

Mean/std scans every training image on CPU and can take a long time. Keep it off the GPU job so the H100 is not idle.

**Chain both jobs** (training starts only after mean/std succeeds):

```bash
MEAN_JOB=$(sbatch --parsable ~/ApexID/slurm_mean_std.sh)
echo "mean/std job: $MEAN_JOB"

TRAIN_JOB=$(sbatch --parsable --dependency=afterok:$MEAN_JOB ~/ApexID/slurm_train_h100.sh)
echo "train job: $TRAIN_JOB"
```

**Or submit separately:**

```bash
MEAN_JOB=$(sbatch --parsable ~/ApexID/slurm_mean_std.sh)
# wait until COMPLETED, then:
sbatch --parsable ~/ApexID/slurm_train_h100.sh
```

### 13. Monitor jobs

**Queue status:**

```bash
squeue -u $(whoami) -o '%.10i %.12j %.8T %.10M %.20R'
```

**Single job:**

```bash
squeue -j JOB_ID -h -o '%T %M %R'
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode,MaxRSS,NodeList -n --parsable2
```

**Logs** (replace `JOB_ID`):

```bash
tail -30 ~/logs/apexid_mean_std_JOB_ID.out
tail -30 ~/logs/apexid_mean_std_JOB_ID.err

tail -30 ~/logs/apexid_train_JOB_ID.out
tail -30 ~/logs/apexid_train_JOB_ID.err
```

**Live follow** (Ctrl+C stops tail only; job keeps running):

```bash
tail -f ~/logs/apexid_train_JOB_ID.out
```

**Cancel a job:**

```bash
scancel JOB_ID
```

### 14. What to expect in the training log

1. Job header (`HOSTNAME=gx…`, `CUDA_VISIBLE_DEVICES=0`)
2. `nvidia-smi` showing **NVIDIA H100 80GB**
3. `uv pip install` (quiet for several minutes)
4. Epoch / loss output from `scripts/train.py`

Mean/std log should end with updated `transforms:` mean/std values written into `configs/cluster_home.yaml`.

### 15. Where outputs land

| Artifact | Location |
|----------|----------|
| Split CSVs | `~/ApexID/open_set_splits/` |
| Updated mean/std | `~/ApexID/configs/cluster_home.yaml` |
| Checkpoints | `~/ApexID/outputs/apexid_osnet_arcface_64_0.5/checkpoints/` |
| Training history | `~/ApexID/outputs/apexid_osnet_arcface_64_0.5/history.pt` |
| SLURM stdout/stderr | `~/logs/apexid_*_<jobid>.out` / `.err` |

---

## Full pipeline (copy-paste summary)

Run on **lx01** after steps 1–9 are done:

```bash
# 1. Splits (CPU)
SPLIT_JOB=$(sbatch --parsable ~/ApexID/slurm_make_splits.sh)

# 2. Config (once per split regen)
cp configs/default.yaml configs/cluster_home.yaml
sed -i "s|/sc/projects/sci-aisc/apex-id/extracted/apexid-dataset/open_set_splits|/sc/home/$(whoami)/ApexID/open_set_splits|g" configs/cluster_home.yaml

# 3. Mean/std (CPU) → Train (1× H100), chained
MEAN_JOB=$(sbatch --parsable ~/ApexID/slurm_mean_std.sh)
TRAIN_JOB=$(sbatch --parsable --dependency=afterok:$MEAN_JOB ~/ApexID/slurm_train_h100.sh)
echo "split=$SPLIT_JOB mean=$MEAN_JOB train=$TRAIN_JOB"
```

Wait for `SPLIT_JOB` to finish **before** step 3 if you need fresh CSVs for mean/std. Or run splits first, wait, then chain mean→train.

---

## Optional: W&B tracking

Set in `configs/cluster_home.yaml`:

```yaml
tracking:
  enabled: true
  provider: wandb
```

### Secrets file (`~/ApexID/.env`)

**Do not run `scp` inside your SSH session on lx01** — it is blocked on the login node.

#### Option A — upload from Windows PowerShell (new window, VPN on)

```powershell
cd C:\Users\asus\Desktop\WILDLIFE\GPU_Cluster_Run\ApexID
.\upload_to_cluster.ps1
```

Or manually (one line each; type password when prompted):

```powershell
$CFG = "C:\Users\asus\Desktop\WILDLIFE\GPU_Cluster_Run\hpi-cluster.ssh-config"
scp -F $CFG "C:\Users\asus\Desktop\WILDLIFE\GPU_Cluster_Run\ApexID\slurm_train_h100.sh" hpi-cluster:~/ApexID/slurm_train_h100.sh
scp -F $CFG "C:\Users\asus\Desktop\WILDLIFE\GPU_Cluster_Run\.env" hpi-cluster:~/ApexID/.env
```

#### Option B — create `.env` on the cluster (no scp)

In your **SSH session**:

```bash
nano ~/ApexID/.env
```

Paste one line:

```bash
WANDB_API_KEY=your_key_here
```

Save, then:

```bash
chmod 600 ~/ApexID/.env
```

#### Verify SLURM script loads the env file

On the cluster:

```bash
grep -n "ENV_FILE" ~/ApexID/slurm_train_h100.sh
```

You should see lines like `ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"` and `source "$ENV_FILE"`.

If missing, either re-upload `slurm_train_h100.sh` (Option A) or paste via base64 (step 9 Option B in this doc), or add this block **before** `cd "$REPO_ROOT"` in `nano ~/ApexID/slurm_train_h100.sh`:

```bash
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"
if [ ! -f "$ENV_FILE" ] && [ -f "$HOME/.env.apexid" ]; then
  ENV_FILE="$HOME/.env.apexid"
fi
if [ -f "$ENV_FILE" ]; then
  set -a
  source "$ENV_FILE"
  set +a
  echo "Loaded env from $ENV_FILE"
else
  echo "No env file at $ENV_FILE (optional; scp .env to ~/ApexID/.env)"
fi
```

Submit:

```bash
chmod +x ~/ApexID/slurm_train_h100.sh
sbatch ~/ApexID/slurm_train_h100.sh
```

Job log should show: `Loaded env from /sc/home/.../ApexID/.env`

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| SSH hangs behind VPN | Add `KexAlgorithms curve25519-sha256` to SSH config |
| `scp` blocked on lx01 | Run `scp` from **local laptop**, not inside SSH |
| `scp` hangs / no prompt | Connect VPN first; run in a **new** PowerShell window (not SSH) |
| `.env` / W&B not loaded | `grep ENV_FILE ~/ApexID/slurm_train_h100.sh` — re-upload script or paste env block via `nano` |
| Heredoc paste corrupts scripts | Use Option A (`scp`) or base64 (Option B) |
| `squeue` empty but job done | Use `sacct -j JOB_ID …` (squeue only shows active jobs) |
| Cannot write shared splits | Use `~/ApexID/open_set_splits/` + `cluster_home.yaml` |
| GPU idle during mean/std | Use separate `slurm_mean_std.sh` (CPU) and `slurm_train_h100.sh` (GPU) |
| Training config wrong paths | Train with `--config configs/cluster_home.yaml` only |

---

## Navigation cheatsheet

| Goal | Command |
|------|---------|
| Login node | `ssh hpi-cluster` (or `-F …/hpi-cluster.ssh-config`) |
| Home | `cd ~` |
| Repo | `cd ~/ApexID` |
| Shared dataset | `cd /sc/projects/sci-aisc/apex-id/extracted/apexid-dataset` |
| Your splits | `cd ~/ApexID/open_set_splits` |
| Logs | `ls ~/logs/` |
