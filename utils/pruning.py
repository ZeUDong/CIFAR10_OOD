"""
Pruning utilities: global magnitude masks + mask enforcement.

A "mask" is a dict { module_path -> bool tensor } over the prunable weights
(nn.Linear and nn.Conv2d ``weight`` tensors). LayerNorm, biases, positional
embeddings and the class token are never pruned (standard practice).
"""

import torch
import torch.nn as nn


PRUNABLE_TYPES = (nn.Linear, nn.Conv2d)


def prunable_modules(model):
    for path, module in model.named_modules():
        if isinstance(module, PRUNABLE_TYPES):
            yield path, module


def make_full_mask(model, device=None):
    mask = {}
    for path, m in prunable_modules(model):
        t = torch.ones_like(m.weight, dtype=torch.bool)
        mask[path] = t.to(device) if device is not None else t
    return mask


@torch.no_grad()
def global_magnitude_mask(model, sparsity, prev_mask=None):
    """
    Global magnitude pruning: keep the (1 - sparsity) fraction of weights with
    the largest |w| across ALL prunable layers; the rest are pruned.

    sparsity:  cumulative target fraction of prunable weights set to 0.
    prev_mask: optional existing mask (already-pruned weights stay pruned).
    """
    scores = []
    layers = list(prunable_modules(model))
    for path, m in layers:
        w = m.weight.detach().abs().flatten()
        if prev_mask is not None:
            w = w[prev_mask[path].flatten()]   # only currently-unpruned weights
        scores.append(w)
    all_scores = torch.cat(scores)             # remaining (unpruned) weights only

    total = sum(m.weight.numel() for _, m in layers)
    already_pruned = total - all_scores.numel()
    n_prune_cum = int(round(sparsity * total))           # cumulative target count
    additional = n_prune_cum - already_pruned            # prune this many MORE now

    if additional <= 0:
        return {path: (prev_mask[path].clone() if prev_mask is not None
                       else torch.ones_like(m.weight, dtype=torch.bool))
                for path, m in layers}

    additional = min(additional, all_scores.numel())
    threshold = torch.kthvalue(all_scores, additional).values  # additional-th smallest of remaining

    mask = {}
    for path, m in layers:
        keep = m.weight.detach().abs() > threshold
        if prev_mask is not None:
            keep = keep & prev_mask[path]
        mask[path] = keep
    return mask


class MaskManager:
    """
    Applies a mask to a model's prunable weights and keeps pruned weights at
    zero during training.

        mm.mask_grads()     # after loss.backward(), before optimizer.step()
        optimizer.step()
        mm.apply()          # re-zero pruned weights
    """

    def __init__(self, model, mask):
        self.model = model
        self.modules = dict(prunable_modules(model))
        dev = next(model.parameters()).device
        self.mask = {k: v.to(dev) for k, v in mask.items()}
        self.apply()

    @torch.no_grad()
    def apply(self):
        for path, m in self.modules.items():
            if path in self.mask:
                m.weight.mul_(self.mask[path])

    @torch.no_grad()
    def mask_grads(self):
        for path, m in self.modules.items():
            if path in self.mask and m.weight.grad is not None:
                m.weight.grad.mul_(self.mask[path])

    def sparsity(self):
        zeros = sum((~self.mask[p]).sum().item() for p in self.mask)
        total = sum(self.mask[p].numel() for p in self.mask)
        return zeros / max(1, total), total


def remap_mask(mask, prefix):
    """Re-key a backbone mask onto a sub-module path (e.g. 'shared_encoder')."""
    return {f"{prefix}.{k}": v for k, v in mask.items()}


def save_mask(mask, path, meta=None):
    torch.save({"mask": {k: v.cpu() for k, v in mask.items()}, "meta": meta or {}}, path)


def load_mask(path, map_location="cpu"):
    obj = torch.load(path, map_location=map_location, weights_only=False)
    return obj["mask"], obj.get("meta", {})
