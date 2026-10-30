#!/bin/bash
# One training arm per SLURM job; extra arguments go to train.py. Submit from the repository root, in the Python
# environment to use (or set PRH_PYTHON), after `mkdir -p runs/logs`; add your cluster's partition and account:
#   sbatch -J prh-image [--partition <p> --account <a>] submit.sh --modality image
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=runs/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# Download tokenizers and pretrained weights beforehand (login node); compute nodes never hit the hub.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Ranks load their data shards before stable-pretraining's Manager, so they reach it minutes apart; non-zero ranks
# wait this long (instead of the default 60 s) for rank 0's run-directory handoff.
export SPT_RANK_HANDOFF_TIMEOUT_S=1800
srun "${PRH_PYTHON:-python}" train.py --workers 20 "$@"
