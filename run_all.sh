#!/usr/bin/env bash
#
# FD-native compression pipeline (no CARD-Deck), over one or more backbones:
#   for each backbone:  no-FD base  ->  dense FD baseline
#                       -> Stage 1 compress (per method x sparsity)
#                       -> Stage 2 sparse FD-native training
#   then compare ALL models (both backbones) in one table.
#
# Every FD model is a single multi-domain model; what varies is backbone /
# compression method / sparsity. Final comparison.csv lists the `backbone`
# column so vit_s and swin_t are compared side by side.
#
# Configurable via env, e.g.:
#   BACKBONES="vit_s swin_t" SPARSITIES="0.5 0.9" METHODS="lrr lth" GPU=1 bash run_all.sh
#   DOMAIN_SOURCE=cifar10c N_TRAIN_DOMAINS=10 bash run_all.sh
#
# Data paths come from data_paths.py / env vars (CIFAR_DATA_ROOT, CIFAR10C_ROOT,
# CIFAR_DOWNLOAD). Set those first.
#
# No `set -e`: a single failed run won't abort the sweep.

set -uo pipefail
cd "$(dirname "$0")"

# ----------------------- config (override via env) -----------------------
BACKBONES=${BACKBONES:-"vit_s swin_t"}      # train both, then compare together
METHODS=${METHODS:-"lrr lth imp ep bp"}     # compression methods to sweep
SPARSITIES=${SPARSITIES:-"0.5 0.9"}         # target sparsities to sweep
COMPRESS_EPOCHS=${COMPRESS_EPOCHS:-50}      # epochs per compression train/round
COMPRESS_ROUNDS=${COMPRESS_ROUNDS:-3}       # iterative rounds (lth/lrr)
FD_EPOCHS=${FD_EPOCHS:-100}                 # epochs for FD training
GPU=${GPU:-0}
MASK_DIR=${MASK_DIR:-./masks}
CARD_DIR=${CARD_DIR:-./cards}
RUN_DENSE=${RUN_DENSE:-1}                    # 1 = train the dense FD baseline (no pruning)
RUN_BASE=${RUN_BASE:-1}                      # 1 = train the no-FD base classifier
RUN_COMPARE=${RUN_COMPARE:-1}                # 0 = skip final compare (e.g. in job-array tasks)
# --- FD domain source ---
DOMAIN_SOURCE=${DOMAIN_SOURCE:-synthetic}    # synthetic (AugMix/Gaussian) | cifar10c (real corruptions)
N_TRAIN_DOMAINS=${N_TRAIN_DOMAINS:-10}       # cifar10c: how many CIFAR-10-C corruptions used as TRAINING domains
TRAIN_CORRUPTIONS=${TRAIN_CORRUPTIONS:-}     # cifar10c: optional explicit train corruption list (comma)
TEST_CORRUPTIONS=${TEST_CORRUPTIONS:-}       # cifar10c: optional explicit held-out test list (comma)
# ------------------------------------------------------------------------

mkdir -p "$MASK_DIR" "$CARD_DIR"

C10C_ARG=""
if [ -n "${CIFAR10C_ROOT:-}" ]; then C10C_ARG="--cifar10c_root ${CIFAR10C_ROOT}"; fi

# FD domain-source args (threaded into the FD training commands)
DOM_ARGS="--domain_source ${DOMAIN_SOURCE}"
if [ "$DOMAIN_SOURCE" = "cifar10c" ]; then
  if [ -z "${CIFAR10C_ROOT:-}" ]; then
    echo "ERROR: DOMAIN_SOURCE=cifar10c needs CIFAR10C_ROOT set (corruptions are the training domains)."; exit 1
  fi
  DOM_ARGS="$DOM_ARGS --num_train_domains ${N_TRAIN_DOMAINS}"
  [ -n "$TRAIN_CORRUPTIONS" ] && DOM_ARGS="$DOM_ARGS --train_corruptions ${TRAIN_CORRUPTIONS}"
  [ -n "$TEST_CORRUPTIONS" ]  && DOM_ARGS="$DOM_ARGS --test_corruptions ${TEST_CORRUPTIONS}"
fi

pct() { awk "BEGIN{printf \"%d\", $1*100}"; }

echo "############ config ############"
echo "backbones=[$BACKBONES] methods=[$METHODS] sparsities=[$SPARSITIES] gpu=$GPU"
echo "domain_source=$DOMAIN_SOURCE  N_TRAIN_DOMAINS=$N_TRAIN_DOMAINS"
echo "data: CIFAR_DATA_ROOT=${CIFAR_DATA_ROOT:-(data_paths default)} CIFAR10C_ROOT=${CIFAR10C_ROOT:-(none)}"
echo "################################"

for BB in $BACKBONES; do
  echo "######## backbone = $BB ########"

  # ----- no-FD base classifier (reference) -----
  if [ "$RUN_BASE" = "1" ]; then
    echo "==== [$BB][baseline] no-FD base classifier ===="
    python train_cifar_base.py --backbone "$BB" \
      --gpu "$GPU" --epochs "$FD_EPOCHS" --save_dir "$CARD_DIR/base_${BB}" $C10C_ARG
  fi

  # ----- dense FD-native baseline (no pruning) -----
  if [ "$RUN_DENSE" = "1" ]; then
    echo "==== [$BB][baseline] dense FD-native (--mask none) ===="
    python train_cifar_fd_sparse.py --backbone "$BB" --mask none --aug multi $DOM_ARGS \
      --gpu "$GPU" --epochs "$FD_EPOCHS" --save_dir "$CARD_DIR/dense_${BB}" $C10C_ARG
  fi

  # ----- Stage 1: compress (clean full CIFAR-10 train) -----
  for m in $METHODS; do
    for s in $SPARSITIES; do
      echo "==== [$BB][Stage 1] compress method=$m sparsity=$s ===="
      python compress.py --method "$m" --sparsity "$s" --backbone "$BB" --gpu "$GPU" \
        --epochs "$COMPRESS_EPOCHS" --rounds "$COMPRESS_ROUNDS" --save_dir "$MASK_DIR"
    done
  done

  # ----- Stage 2: FD-native training under each mask -----
  for m in $METHODS; do
    for s in $SPARSITIES; do
      P=$(pct "$s")
      MASK="$MASK_DIR/mask_${m}_${BB}_s${P}.pt"
      if [ ! -f "$MASK" ]; then echo "  (skip) missing mask $MASK"; continue; fi
      OUT="$CARD_DIR/${m}_${BB}_s${P}"
      echo "==== [$BB][Stage 2] sparse FD-native method=$m sparsity=$s ===="
      python train_cifar_fd_sparse.py --backbone "$BB" --mask "$MASK" --aug multi $DOM_ARGS \
        --gpu "$GPU" --epochs "$FD_EPOCHS" --save_dir "$OUT" $C10C_ARG
    done
  done
done

# ----------------------- compare ALL models (both backbones) -----------------------
if [ "$RUN_COMPARE" = "1" ]; then
  echo "==== [compare] all backbones ===="
  python compare.py --glob "$CARD_DIR/*/summary.csv" --out comparison.csv
  echo "############ DONE -> comparison.csv (base / FD / FD+best per backbone) ############"
else
  echo "############ DONE (compare skipped; run compare.py separately) ############"
fi
