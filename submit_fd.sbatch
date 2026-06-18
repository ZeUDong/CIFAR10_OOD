#!/bin/bash
# =====================================================================
# Use BOTH GPUs in one job. Train a compressed FD for every mask in ./masks/
# (plus dense FD + base eval), split across 2 GPUs, then compare.
#   sbatch submit_fd.sbatch
# Everything is set directly below -- only the two >>> EDIT <<< lines are
# cluster-specific (partition name + how you activate your python env).
# =====================================================================
#SBATCH --job-name=fd-ood
#SBATCH --partition=gpu              # >>> EDIT <<< your GPU partition / queue name
#SBATCH --gres=gpu:2                 # two GPUs on one node
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs cards masks

# >>> EDIT <<< activate your python env (pick the one that applies):
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate myenv
# module load cuda/11.8

echo "python: $(which python)"; python -c "import torch;print('torch',torch.__version__,'GPUs',torch.cuda.device_count())"

# ---------- data paths (set directly) ----------
export CIFAR_DATA_ROOT=/scratch/zdong112/cifar10
export CIFAR10C_ROOT=/scratch/zdong112/cifar10c
export CIFAR_DOWNLOAD=0
ls "$CIFAR_DATA_ROOT/cifar-10-batches-py/test_batch" >/dev/null || { echo "CIFAR-10 missing";   exit 1; }
ls "$CIFAR10C_ROOT/labels.npy"                          >/dev/null || { echo "CIFAR-10-C missing"; exit 1; }

# ---------- config (set directly) ----------
BACKBONE=vit_s
FD_EPOCHS=100
AUG=multi

# ---------- 0. base OOD (reuse checkpoint, quick, GPU 0) ----------
BASE_CKPT=./cards/base_${BACKBONE}/best_cifar_base_${BACKBONE}.pt
if [ -f "$BASE_CKPT" ]; then
  echo "==== [base] eval on CIFAR-10-C (reuse) ===="
  python eval_ood.py --ckpt "$BASE_CKPT" --gpu 0
fi

# ---------- build task list: "backbone|mask|out_dir" ----------
shopt -s nullglob
TASKS=()
TASKS+=("$BACKBONE|none|./cards/dense_${BACKBONE}")     # dense FD (no pruning)
for MASK in ./masks/*.pt; do
  name=$(basename "$MASK" .pt); name=${name#mask_}      # e.g. lrr_vit_s_s50
  rest=${name#*_}; bb=${rest%_*}                        # strip method -> vit_s
  TASKS+=("$bb|$MASK|./cards/${name}")
done
echo "Total FD trainings: ${#TASKS[@]} (dense + masks), split over 2 GPUs"

# ---------- one GPU lane: run its tasks serially on a fixed GPU ----------
run_lane() {
  local gpu=$1; shift
  for t in "$@"; do
    IFS='|' read -r bb mask out <<< "${TASKS[$t]}"
    if [ -f "$out/summary.csv" ]; then echo "[gpu$gpu] skip $out (done)"; continue; fi
    echo "[gpu$gpu] train $out  (mask=$mask)"
    python train_cifar_fd_sparse.py --backbone "$bb" --mask "$mask" --aug "$AUG" \
      --epochs "$FD_EPOCHS" --gpu "$gpu" --save_dir "$out"
  done
}

# split task indices: even -> GPU 0, odd -> GPU 1
even=(); odd=()
for i in "${!TASKS[@]}"; do
  if (( i % 2 == 0 )); then even+=("$i"); else odd+=("$i"); fi
done

run_lane 0 "${even[@]}" &
run_lane 1 "${odd[@]}" &
wait

# ---------- comparison table ----------
echo "==== [compare] ===="
python compare.py --glob "./cards/*/summary.csv" --out comparison.csv
echo "############ DONE -> comparison.csv ############"
