#!/bin/bash
# =====================================================================
# FD with REAL CIFAR-10-C corruptions as training domains (leave-domains-out),
# warm-started from the trained base backbone, vs base, on the SAME held-out
# corruptions. Results -> ./cards_c10c/. Auto-uses all available GPUs.
#   tmux new -d -s c10c 'bash run_cifar10c.sh 2>&1 | tee run_c10c.log'
#
# NOTE: the python commands are single-line on purpose (no '\' continuations),
# so please do NOT add inline comments or break the lines.
# =====================================================================
set -uo pipefail
cd "$(dirname "$0")"

# >>> EDIT <<< activate your python env:
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate myenv

export CIFAR_DATA_ROOT=/scratch/zdong112/cifar10
export CIFAR10C_ROOT=/scratch/zdong112/cifar10c
export CIFAR_DOWNLOAD=0
ls "$CIFAR_DATA_ROOT/cifar-10-batches-py/test_batch" >/dev/null || { echo "CIFAR-10 missing";   exit 1; }
ls "$CIFAR10C_ROOT/labels.npy"                          >/dev/null || { echo "CIFAR-10-C missing"; exit 1; }

BACKBONE=vit_s
FD_EPOCHS=100
LATENT=384
HELDOUT=2000                 # hold out the LAST N images for val/OOD (no content leak)
CARD_DIR=./cards_c10c
mkdir -p "$CARD_DIR"

# 3-way corruption split (disjoint): 10 train / 4 val / 5 test
TRAIN_CORR="gaussian_noise,shot_noise,impulse_noise,defocus_blur,glass_blur,motion_blur,zoom_blur,snow,frost,fog"
VAL_CORR="speckle_noise,gaussian_blur,brightness,spatter"
TEST_CORR="contrast,elastic_transform,pixelate,jpeg_compression,saturate"
VAL_SEV=3
BASE_CKPT=./cards/base_${BACKBONE}/best_cifar_base_${BACKBONE}.pt

NG=$(python -c "import torch;print(torch.cuda.device_count())" 2>/dev/null || echo 0)
[ "$NG" -ge 1 ] || NG=1
echo "Using $NG GPU(s)."

# ---------- base on the held-out corruptions (reuse checkpoint) ----------
if [ -f "$BASE_CKPT" ]; then
  echo "[base] eval on held-out corruptions"
  python eval_ood.py --ckpt "$BASE_CKPT" --corruptions "$TEST_CORR" --heldout_images "$HELDOUT" --out_dir "$CARD_DIR/base_${BACKBONE}" --gpu 0
else
  echo "(no base checkpoint -- skipping base)"
fi

# ---------- task list: dense FD + one per mask ----------
shopt -s nullglob
TASKS=("$BACKBONE|none|$CARD_DIR/dense_${BACKBONE}")
for M in ./masks/*.pt; do
  n=$(basename "$M" .pt); n=${n#mask_}
  r=${n#*_}; bb=${r%_*}
  TASKS+=("$bb|$M|$CARD_DIR/$n")
done
echo "Total FD trainings: ${#TASKS[@]} (dense + masks), split over $NG GPU(s)"

WARM=""
[ -f "$BASE_CKPT" ] && WARM="--init_backbone $BASE_CKPT"

run_lane() {
  local gpu=$1
  for i in "${!TASKS[@]}"; do
    (( i % NG == gpu )) || continue
    IFS='|' read -r bb mask out <<< "${TASKS[$i]}"
    if [ -f "$out/summary.csv" ]; then echo "[gpu$gpu] skip $out (done)"; continue; fi
    echo "[gpu$gpu] train $out (mask=$mask)"
    python train_cifar_fd_sparse.py --backbone "$bb" --mask "$mask" --domain_source cifar10c --train_corruptions "$TRAIN_CORR" --val_corruptions "$VAL_CORR" --val_severities "$VAL_SEV" --test_corruptions "$TEST_CORR" --latent_dim "$LATENT" --heldout_images "$HELDOUT" $WARM --epochs "$FD_EPOCHS" --gpu "$gpu" --save_dir "$out"
  done
  echo "[gpu$gpu] lane finished."
}

for g in $(seq 0 $((NG - 1))); do run_lane "$g" & done
wait

echo "==== compare ===="
python compare.py --glob "$CARD_DIR/*/summary.csv" --out comparison_c10c.csv
echo "############ ALL DONE -> comparison_c10c.csv ############"
