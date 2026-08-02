# Sub-center ArcFace A/B Gates

This document defines the pass/fail gates for switching from the ArcFace baseline to the Sub-center ArcFace dynamic head.

## Baseline (A)

Baseline config: `configs/cluster_home.yaml`  
Baseline run family: `apexid_osnet_pt_arcface_full_finetune_1024`

Latest best baseline checkpoint metrics (user-reported):

- `known/mAP`: `0.1846`
- `known/top1`: `0.4613`
- `known/BAKS`: `0.3579`
- `unknown/BAUS`: `0.7765`
- `unknown/v_measure`: `0.8217`
- `mixed/NormAcc`: `0.4534`
- `mixed/OSCR_AUC`: `0.3883`
- `mixed/DIR_at_1FAR`: `0.1359`

## Candidate (B)

Candidate config: `configs/cluster_subcenter_dynamic.yaml`  
Candidate run family: `apexid_osnet_pt_subcenter_arcface_full_finetune_1024`

## Decision gates

Promote B over A only if all required gates hold:

Required gates:

1. `known/mAP` improves by at least `+0.010` absolute.
2. `known/top1` does not drop.
3. `unknown/BAUS` drop is at most `0.010` absolute.
4. `mixed/OSCR_AUC` does not drop.

Tie-breaker gates (if required gates pass):

1. Prefer higher `mixed/NormAcc`.
2. Prefer higher `mixed/DIR_at_1FAR`.
3. Prefer higher `known/BAKS`.

## Run commands

Run A (baseline):

```bash
cd ~/ApexID
CONFIG=configs/cluster_home.yaml sbatch --exclude=gx13v1 slurm_train_h100.sh
```

Run B (candidate):

```bash
cd ~/ApexID
CONFIG=configs/cluster_subcenter_dynamic.yaml sbatch --exclude=gx13v1 slurm_train_h100.sh
```

## First comparison template

Fill this checklist after both runs finish:

- `known/mAP`: A=`...`, B=`...`, delta=`...`, gate=`pass/fail`
- `known/top1`: A=`...`, B=`...`, delta=`...`, gate=`pass/fail`
- `unknown/BAUS`: A=`...`, B=`...`, delta=`...`, gate=`pass/fail`
- `mixed/OSCR_AUC`: A=`...`, B=`...`, delta=`...`, gate=`pass/fail`
- `mixed/NormAcc`: A=`...`, B=`...`, delta=`...`, gate=`tie-breaker`
- `mixed/DIR_at_1FAR`: A=`...`, B=`...`, delta=`...`, gate=`tie-breaker`

Promotion rule:

- Promote B only if all required gates pass.
- Otherwise keep A and tune `k` and/or margin schedule.
