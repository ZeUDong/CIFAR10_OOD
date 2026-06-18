"""Image patch embedding and patch merging."""

import torch
import torch.nn as nn

from .helpers import to_2tuple


class PatchEmbed(nn.Module):
    """
    Conv-based image-to-patch embedding.

    [B, C, H, W] -> [B, num_patches, embed_dim], where num_patches =
    (H/patch) * (W/patch). Used by both ViT (patch=4) and Swin (patch=2).
    """

    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=384, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.proj(x)                  # [B, embed_dim, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]
        x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    """
    Swin downsampling: concatenate 2x2 neighbours and project 4C -> 2C.

    [B, H*W, C] -> [B, (H/2)*(W/2), 2C].
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4C]
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)
        return x
