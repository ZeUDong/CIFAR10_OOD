"""
Strong-lottery-ticket layers: edge-popup (EP) and biprop (BP).

Instead of training weights, we freeze a signed random initialization and learn
a per-weight *score*; a straight-through top-k selects the subnetwork. BP also
binarizes the kept weights. After training the scores we export a boolean mask
in the same format as the magnitude methods, so downstream FD training treats
all compression methods identically.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GetSubnet(torch.autograd.Function):
    """Top-k (by score) binary mask with a straight-through gradient."""

    @staticmethod
    def forward(ctx, scores, keep_fraction):
        out = torch.zeros_like(scores)
        flat = scores.flatten()
        k = max(1, int(round(keep_fraction * flat.numel())))
        _, idx = flat.sort()
        keep_idx = idx[-k:]
        o = out.flatten()
        o[keep_idx] = 1.0
        return out

    @staticmethod
    def backward(ctx, g):
        return g, None  # straight-through


class SubnetLinear(nn.Linear):
    def __init__(self, in_f, out_f, bias=True, keep_fraction=0.5, binary=False):
        super().__init__(in_f, out_f, bias=bias)
        self.keep_fraction = keep_fraction
        self.binary = binary
        self.scores = nn.Parameter(torch.empty_like(self.weight))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def _masked_weight(self):
        mask = GetSubnet.apply(self.scores.abs(), self.keep_fraction)
        if self.binary:
            alpha = self.weight.abs().mean()
            return self.weight.sign() * alpha * mask
        return self.weight * mask

    def forward(self, x):
        return F.linear(x, self._masked_weight(), self.bias)


class SubnetConv2d(nn.Conv2d):
    def __init__(self, *args, keep_fraction=0.5, binary=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.keep_fraction = keep_fraction
        self.binary = binary
        self.scores = nn.Parameter(torch.empty_like(self.weight))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def _masked_weight(self):
        mask = GetSubnet.apply(self.scores.abs(), self.keep_fraction)
        if self.binary:
            alpha = self.weight.abs().mean()
            return self.weight.sign() * alpha * mask
        return self.weight * mask

    def forward(self, x):
        return self._conv_forward(x, self._masked_weight(), self.bias)


def _replace(parent, name, child):
    setattr(parent, name, child)


def convert_to_subnet(model, keep_fraction=0.5, binary=False):
    """
    Replace every nn.Linear / nn.Conv2d in `model` (in place) with its Subnet
    counterpart, copying the existing (signed) weights as the frozen init.
    Module paths are preserved, so the exported mask keys match a fresh backbone.
    """
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear) and not isinstance(child, SubnetLinear):
            new = SubnetLinear(child.in_features, child.out_features,
                               bias=child.bias is not None,
                               keep_fraction=keep_fraction, binary=binary)
            new.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                new.bias.data.copy_(child.bias.data)
            _replace(model, name, new)
        elif isinstance(child, nn.Conv2d) and not isinstance(child, SubnetConv2d):
            new = SubnetConv2d(child.in_channels, child.out_channels, child.kernel_size,
                               stride=child.stride, padding=child.padding,
                               dilation=child.dilation, groups=child.groups,
                               bias=child.bias is not None,
                               keep_fraction=keep_fraction, binary=binary)
            new.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                new.bias.data.copy_(child.bias.data)
            _replace(model, name, new)
        else:
            convert_to_subnet(child, keep_fraction, binary)
    return model


@torch.no_grad()
def extract_mask_from_subnet(model):
    """Export {path -> bool mask} from learned scores (top-k by |score|)."""
    mask = {}
    for path, m in model.named_modules():
        if isinstance(m, (SubnetLinear, SubnetConv2d)):
            sel = GetSubnet.apply(m.scores.abs(), m.keep_fraction).bool()
            mask[path] = sel.cpu()
    return mask
