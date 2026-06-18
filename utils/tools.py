"""
Training utilities (kept in utils/, like the original project).

Optional helpers you can wire into the train scripts:
    - EarlyStopping        : patience-based early stopping on a monitored loss
    - adjust_learning_rate : cosine / step / exponential LR schedules
    - count_parameters     : total / trainable parameter counts
    - model_size_mb        : parameter memory footprint
    - dotdict              : attribute access for config dicts

The FD training scripts already do inline early stopping; these are here for
convenience and for the compression experiments (reporting parameter counts).
"""

import math

import numpy as np
import torch


class EarlyStopping:
    """Monitor a value; save the best model and stop after `patience` epochs without improvement."""

    def __init__(self, patience=50, verbose=True, delta=0.0, path="checkpoint.pt"):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self._save(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save(val_loss, model)
            self.counter = 0

    def _save(self, val_loss, model):
        if self.verbose:
            print(f"Val loss decreased ({self.val_loss_min:.6f} -> {val_loss:.6f}). Saving model.")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def adjust_learning_rate(optimizer, epoch, base_lr, schedule="cosine",
                         total_epochs=100, warmup_epochs=0, min_lr=1e-6, step_size=30, gamma=0.1):
    """Set the LR for `epoch` (1-indexed) by a schedule and apply it. Returns the new LR."""
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        lr = base_lr * epoch / max(1, warmup_epochs)
    elif schedule == "cosine":
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
    elif schedule == "step":
        lr = base_lr * (gamma ** ((epoch - 1) // step_size))
    elif schedule == "exp":
        lr = base_lr * (gamma ** (epoch - 1))
    else:
        lr = base_lr
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_size_mb(model):
    total, _ = count_parameters(model)
    return total * 4 / (1024 ** 2)


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
