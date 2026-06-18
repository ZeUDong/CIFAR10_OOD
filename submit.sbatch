#!/bin/bash
# =====================================================================
# SLURM submission script for the CIFAR-10-FD pipeline.
# run_all.sh itself needs NO changes -- this wrapper just sets resources,
# the environment, and the data-path env vars, then calls run_all.sh.
#
# Submit from inside the repo directory:
#     sbatch submit.sbatch
# Override the sweep without editing this file:
#     sbatch --export=ALL,BACKBONES=vit_s,METHODS=lrr,SPARSITIES=0.9 submit.sbatch
# =====================================================================

#SBATCH --job-name=cifar-fd
#SBATCH --partition=gpu            # <-- your GPU partition / queue name
#SBATCH --gres=gpu:1              # request 1 GPU
#SBATCH --cpus-per-task=8         # data-loader workers
#SBATCH --mem=32G
#SBATCH --time=24:00:00           # walltime (the full sweep is long; raise or split)
#SBATCH --output=logs/%x_%j.out   # stdout -> logs/cifar-fd_<jobid>.out
#SBATCH --error=logs/%x_%j.err    # stderr

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

# ---------- environment (uncomment / edit for your cluster) ----------
# module load cuda/11.8
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate cifar-fd          # env with torch / torchvision / scikit-learn / tqdm / numpy
echo "python: $(which python)"; python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# ---------- data paths (compute nodes are usually offline) ----------
export CIFAR_DATA_ROOT=${CIFAR_DATA_ROOT:-/path/to/cifar10}        # folder that CONTAINS cifar-10-batches-py/
export CIFAR10C_ROOT=${CIFAR10C_ROOT:-/path/to/CIFAR-10-C}         # folder with *.npy
export CIFAR_DOWNLOAD=${CIFAR_DOWNLOAD:-0}                         # 0 = don't download on an offline node
# sanity check the data is reachable before burning GPU hours
ls "$CIFAR_DATA_ROOT/cifar-10-batches-py/test_batch" >/dev/null || { echo "CIFAR-10 not found at CIFAR_DATA_ROOT"; exit 1; }
ls "$CIFAR10C_ROOT/labels.npy" >/dev/null || { echo "CIFAR-10-C not found at CIFAR10C_ROOT"; exit 1; }

# ---------- run config (override via --export or here) ----------
export BACKBONES=${BACKBONES:-"vit_s swin_t"}
export METHODS=${METHODS:-"lrr lth imp ep bp"}
export SPARSITIES=${SPARSITIES:-"0.5 0.9"}
export FD_EPOCHS=${FD_EPOCHS:-100}
export COMPRESS_EPOCHS=${COMPRESS_EPOCHS:-50}
export DOMAIN_SOURCE=${DOMAIN_SOURCE:-synthetic}   # or: cifar10c
export N_TRAIN_DOMAINS=${N_TRAIN_DOMAINS:-10}
export GPU=0     # SLURM exposes the allocated GPU as index 0 (via CUDA_VISIBLE_DEVICES)

srun bash run_all.sh

# =====================================================================
# Variations
# ---------------------------------------------------------------------
# (a) Two GPUs in one job (vit_s on gpu0, swin_t on gpu1, in parallel):
#       #SBATCH --gres=gpu:2
#     then replace the `srun bash run_all.sh` line with:
#       BACKBONES=vit_s  GPU=0 RUN_BASE=1 bash run_all.sh &
#       BACKBONES=swin_t GPU=1 RUN_BASE=1 bash run_all.sh &
#       wait
#       python compare.py --glob "./cards/*/summary.csv" --out comparison.csv
#
# (b) Job array, one array task per backbone (separate GPUs/nodes), then a
#     dependent compare job:
#       sbatch --array=0-1 submit_array.sbatch        # see header note below
#     In an array script:  BBS=(vit_s swin_t); export BACKBONES=${BBS[$SLURM_ARRAY_TASK_ID]}
#     and run `RUN_COMPARE=0 bash run_all.sh` per task, then:
#       sbatch --dependency=afterok:<arrayJobId> --wrap \
#         "python compare.py --glob './cards/*/summary.csv' --out comparison.csv"
# =====================================================================
