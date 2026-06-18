"""Swin-Tiny backbone, hand-written from layers/ (no timm)."""

import torch
import torch.nn as nn

from layers import PatchEmbed, PatchMerging, BasicLayer


class SwinTransformer(nn.Module):
    """
    Hierarchical Swin transformer. ``forward`` returns the global-average-pooled
    feature ``[B, num_features]`` (no classification head), with
    num_features = embed_dim * 2**(num_stages-1). ``feature_dim`` exposes that
    dimension for downstream heads.

    Swin-T config: embed_dim=96, depths=(2,2,6,2), num_heads=(3,6,12,24).
    """

    def __init__(self, img_size=32, patch_size=2, in_chans=3, embed_dim=96,
                 depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), window_size=4,
                 mlp_ratio=4.0, qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.feature_dim = self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim,
                                      norm_layer=norm_layer)
        grid = self.patch_embed.grid_size  # e.g. (16, 16) for img=32, patch=2
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i),
                input_resolution=(grid[0] // (2 ** i), grid[1] // (2 ** i)),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i < self.num_layers - 1) else None,
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)        # [B, L, embed_dim]
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)               # [B, L', num_features]
        x = self.norm(x)
        x = x.mean(dim=1)              # global average pool -> [B, num_features]
        return x


def build_swin_t(img_size=32, patch_size=2, window_size=4, in_chans=3,
                 drop_rate=0.0, pretrained=False):
    """Swin-T, native 32x32 from scratch (patch_size=2, window_size=4)."""
    if pretrained:
        raise NotImplementedError(
            "From-scratch implementation; no pretrained weights. Use pretrained=False."
        )
    return SwinTransformer(
        img_size=img_size, patch_size=patch_size, in_chans=in_chans,
        embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
        window_size=window_size, mlp_ratio=4.0, drop_rate=drop_rate,
    )
