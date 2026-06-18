#!/bin/bash
# =====================================================================
# Job-array variant: one array task per backbone (parallel, separate GPUs).
# Each task runs run_all.sh for ONE backbone with RUN_COMPARE=0; then submit a
# single dependent compare job to merge the results.
#
#     sbatch submit_array.sbatch
#     # then (after the array finishes), the script auto-submits the compare job,
#     # OR do it manually:
#     #   python compare.py --glob "./cards/*/summary.csv" --out comparison.csv
# =====================================================================

#SBATCH --job-name=cifar-fd-arr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-1                 # 2 tasks -> 2 backbones
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

# ---------- environment ----------
# module load cuda/11.8
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate cifar-fd

# ---------- data paths ----------
export CIFAR_DATA_ROOT=${CIFAR_DATA_ROOT:-/path/to/cifar10}
export CIFAR10C_ROOT=${CIFAR10C_ROOT:-/path/to/CIFAR-10-C}
export CIFAR_DOWNLOAD=${CIFAR_DOWNLOAD:-0}

# ---------- pick this task's backbone ----------
BBS=(vit_s swin_t)
export BACKBONES=${BBS[$SLURM_ARRAY_TASK_ID]}
export GPU=0
export METHODS=${METHODS:-"lrr lth imp ep bp"}
export SPARSITIES=${SPARSITIES:-"0.5 0.9"}
export FD_EPOCHS=${FD_EPOCHS:-100}
export DOMAIN_SOURCE=${DOMAIN_SOURCE:-synthetic}
export N_TRAIN_DOMAINS=${N_TRAIN_DOMAINS:-10}
export RUN_COMPARE=0               # each task skips the (partial) compare

echo "array task $SLURM_ARRAY_TASK_ID -> backbone $BACKBONES"
srun bash run_all.sh

# The last finishing task launches the merged comparison over ALL cards.
# (Simpler/robust alternative: submit the compare job yourself with a dependency:
#   sbatch --dependency=afterok:<thisArrayJobId> --wrap \
#     "python compare.py --glob './cards/*/summary.csv' --out comparison.csv")
