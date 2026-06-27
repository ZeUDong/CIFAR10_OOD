"""
Compression (pruning / NAS) entry point -- SCAFFOLD.

Self-contained, like the other train_cifar_*.py scripts. Mirrors the two-stage
structure of the original NAS training:

    Stage 1: architecture search on an FD-trained supernet
             (loss = CE + lambda_flops * expected_cost_ratio)
    Stage 2: derive the sub-architecture and fine-tune (reuse the FD epoch loop
             from train_cifar_fd.py).

Target compression ratios to sweep: 25% / 50% / 75%.

Two intended back-ends:
  1. Structured pruning via Torch-Pruning over the two image encoders, then
     fine-tune.
  2. Differentiable FFN-width search (Gumbel-Softmax) over the transformer
     blocks in layers/Transformer_EncDec.py.

The core search/prune step raises NotImplementedError so the interface is clear
without pretending the method is implemented.
"""

import os
import sys
import argparse

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import build_backbone  # noqa: F401  (used when you rebuild the FD model)


def expected_cost_ratio(model):
    """Differentiable compute-cost ratio in [0,1] for the current supernet."""
    raise NotImplementedError("Define the cost model for your search space.")


def prune_fd_model(fd_model, example_image, target_ratio=0.5, importance="magnitude"):
    """
    Structurally prune a trained FeatureDisentangled model to `target_ratio`
    (fraction to KEEP), then return it for fine-tuning with the FD epoch loop.

    TODO: wire up torch_pruning.MetaPruner over fd_model.shared_encoder and
    fd_model.private_encoder; keep the FD heads intact.
    """
    raise NotImplementedError(
        "Implement with torch_pruning.MetaPruner; prune encoders then finetune."
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", type=str, default="vit_s", choices=["vit_s", "swin_t"])
    p.add_argument("--fd_ckpt", type=str, required=True,
                   help="Trained FD checkpoint from train_cifar_fd.py.")
    p.add_argument("--method", type=str, default="prune", choices=["prune", "nas"])
    p.add_argument("--target_ratio", type=float, default=0.5, help="Fraction to KEEP (0.25/0.5/0.75).")
    p.add_argument("--lambda_flops", type=float, default=0.1)
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--cifar10c_root", type=str, default="")
    p.add_argument("--save_dir", type=str, default="./checkpoints_cifar_compress")
    p.add_argument("--img_size", type=int, default=32)
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"[compression] method={args.method} target_ratio={args.target_ratio}")
    print(f"[compression] loading FD checkpoint: {args.fd_ckpt}")

    # NOTE: rebuild FeatureDisentangled exactly as in train_cifar_fd.py, load
    # weights, then prune / search, fine-tune with the FD epoch loop, and
    # evaluate OOD on CIFAR-10-C.
    raise NotImplementedError(
        "Compression back-end not implemented yet. Fill in prune_fd_model "
        "(Torch-Pruning) or the Gumbel-Softmax FFN-width search."
    )


if __name__ == "__main__":
    main()
