"""
Backbone models for the CIFAR-10 feature-disentanglement framework.

Each backbone is an ``nn.Module`` whose ``forward(x)`` returns a 2D feature
tensor ``[B, D]`` with the dimension exposed as ``.feature_dim``.
"""

from .vit import VisionTransformer, build_vit_s
from .swin import SwinTransformer, build_swin_t

VALID_BACKBONES = ("vit_s", "swin_t")


def build_backbone(
    name,
    img_size=32,
    in_chans=3,
    drop_rate=0.0,
    pretrained=False,
    vit_patch_size=4,
    swin_patch_size=2,
    swin_window_size=4,
):
    """
    Build a single encoder (returns an nn.Module mapping image -> [B, D]).

    name:
        'vit_s'  -> ViT-Small  (native 32x32 from scratch)
        'swin_t' -> Swin-Tiny  (native 32x32 from scratch)
    """
    key = name.lower().replace("-", "_")
    if key in ("vit_s", "vit_small", "vits"):
        return build_vit_s(img_size, vit_patch_size, in_chans, drop_rate, pretrained)
    if key in ("swin_t", "swin_tiny", "swint"):
        return build_swin_t(img_size, swin_patch_size, swin_window_size,
                            in_chans, drop_rate, pretrained)
    raise ValueError(f"Unknown backbone '{name}'. Valid options: {VALID_BACKBONES}")


__all__ = ["build_backbone", "VALID_BACKBONES",
           "VisionTransformer", "build_vit_s",
           "SwinTransformer", "build_swin_t"]
