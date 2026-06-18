"""ViT-Small backbone, hand-written from layers/ (no timm)."""

import torch
import torch.nn as nn

from layers import PatchEmbed, EncoderLayer, Encoder


class VisionTransformer(nn.Module):
    """
    ViT with a class token. ``forward`` returns the final class-token feature
    ``[B, embed_dim]`` (no classification head). ``feature_dim`` exposes that
    dimension so downstream heads (FD projection, linear classifier) can size
    themselves.

    ViT-S config: embed_dim=384, depth=12, num_heads=6.
    """

    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=384,
                 depth=12, num_heads=6, mlp_ratio=4.0, qkv_bias=True,
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.feature_dim = self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = Encoder([
            EncoderLayer(embed_dim, num_heads, mlp_ratio, qkv_bias,
                         drop_rate, attn_drop_rate, dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
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
        x = self.patch_embed(x)                              # [B, N, C]
        cls = self.cls_token.expand(x.shape[0], -1, -1)      # [B, 1, C]
        x = torch.cat([cls, x], dim=1)                       # [B, N+1, C]
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0]                                       # class token -> [B, C]


def build_vit_s(img_size=32, patch_size=4, in_chans=3, drop_rate=0.0, pretrained=False):
    """ViT-S, native 32x32 from scratch (patch_size=4 -> 64 tokens)."""
    if pretrained:
        raise NotImplementedError(
            "From-scratch implementation; no pretrained weights. Use pretrained=False."
        )
    return VisionTransformer(
        img_size=img_size, patch_size=patch_size, in_chans=in_chans,
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0, drop_rate=drop_rate,
    )
