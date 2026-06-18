"""
Legacy Gumbel-NAS compression scaffold -- SUPERSEDED by the CARD-style pipeline
(compress.py + train_cifar_fd_sparse.py + compare.py). Kept for reference only.

Two-stage idea (not implemented here):
    Stage 1: architecture search on an FD-trained supernet
             (loss = CE + lambda_flops * expected_cost_ratio)
    Stage 2: derive the sub-architecture and fine-tune.

For OOD-oriented compression, use the lottery-ticket pipeline instead:
    compress.py  ->  train_cifar_fd_sparse.py  ->  compare.py
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", type=str, default="vit_s", choices=["vit_s", "swin_t"])
    p.add_argument("--fd_ckpt", type=str, default="")
    p.add_argument("--method", type=str, default="prune", choices=["prune", "nas"])
    p.add_argument("--target_ratio", type=float, default=0.5)
    args = p.parse_args()
    raise NotImplementedError(
        "Superseded. Use compress.py + train_cifar_fd_sparse.py + compare.py for "
        "the CARD-style lottery-ticket compression pipeline."
    )


if __name__ == "__main__":
    main()
