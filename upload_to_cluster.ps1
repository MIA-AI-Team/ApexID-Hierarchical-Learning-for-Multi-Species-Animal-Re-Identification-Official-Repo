# Upload ApexID files to HPI cluster from Windows (run in PowerShell, NOT inside SSH).
# Requires: VPN connected, password ready when scp prompts.

$ErrorActionPreference = "Stop"
$CFG = "C:\Users\asus\Desktop\WILDLIFE\GPU_Cluster_Run\hpi-cluster.ssh-config"
$REPO = "C:\Users\asus\Desktop\WILDLIFE\GPU_Cluster_Run\ApexID"
$ENV_FILE = "C:\Users\asus\Desktop\WILDLIFE\GPU_Cluster_Run\.env"

Write-Host "Uploading slurm_train_h100.sh ..."
scp -F $CFG "$REPO\slurm_train_h100.sh" hpi-cluster:~/ApexID/slurm_train_h100.sh

if (Test-Path $ENV_FILE) {
    Write-Host "Uploading .env -> ~/ApexID/.env ..."
    scp -F $CFG $ENV_FILE hpi-cluster:~/ApexID/.env
} else {
    Write-Warning ".env not found at $ENV_FILE — skip or create it first."
}

Write-Host "Done. On cluster run:"
Write-Host "  chmod 600 ~/ApexID/.env; chmod +x ~/ApexID/slurm_train_h100.sh"
Write-Host "  grep ENV_FILE ~/ApexID/slurm_train_h100.sh"
Write-Host "  sbatch ~/ApexID/slurm_train_h100.sh"
