"""
Self-contained smoke test (no dataset download required).

For each backbone (vit_s, swin_t):
    - backbone feature-dim
    - FeatureDisentangled: forward, all 5 losses, backward, optimizer step
    - forward_shared_only (eval / inference path)
Also checks the augmentation-as-domains dataset + ImageBatchSampler, and the
compression mask + sparse FD step.

Run:  python smoke_test.py
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import build_backbone
from train_cifar_fd import (
    FeatureDisentangled, run_epoch_fd,
    CIFAR10AugDomainDataset, ImageBatchSampler, ParameterizedAugment,
)


def fake_batch(B=8, H=32, sim_dim=8, n_groups=4, device="cpu"):
    return {
        "image": torch.randn(B, 3, H, H, device=device),
        "label": torch.randint(0, 10, (B,), device=device),
        "sim_param": torch.randn(B, sim_dim, device=device),
        "content_id": torch.arange(B, device=device) % n_groups,  # 2 views/content
    }


def test_backbone(name, device):
    print(f"\n=== {name} ===")
    H = 32
    ex = torch.zeros(2, 3, H, H, device=device)

    def builder():
        return build_backbone(name, img_size=H).to(device)

    print(f"  backbone feature dim: {builder()(ex).shape[1]}")

    model = FeatureDisentangled(
        backbone_builder=builder, input_image=ex, recon_dim=3 * H * H, sim_dim=8,
        num_classes=10, latent_dim=64, sim_embed_dim=32, hidden_dim=128, recon_hidden_dim=256,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loader = [fake_batch(device=device) for _ in range(2)]
    m = run_epoch_fd(model, loader, nn.CrossEntropyLoss(), device, optimizer=opt, desc="fd-train")
    print(f"  FD train: loss={m['loss']:.4f} acc={m['acc']:.3f} "
          f"(rec={m['rec_loss']:.3f} inv={m['inv_loss']:.3f} sim={m['sim_loss']:.3f} sep={m['sep_loss']:.3f})")

    eval_loader = [{"image": b["image"], "label": b["label"]} for b in loader]
    me = run_epoch_fd(model, eval_loader, nn.CrossEntropyLoss(), device, optimizer=None, desc="fd-eval")
    print(f"  FD eval (shared-only): acc={me['acc']:.3f}")
    print(f"  OK: {name}")


def test_data_pipeline():
    print("\n=== data pipeline ===")
    imgs = (np.random.rand(20, 32, 32, 3) * 255).astype(np.uint8)
    labels = np.random.randint(0, 10, 20)
    ds = CIFAR10AugDomainDataset(imgs, labels, sims_per_image=5, augment=ParameterizedAugment())
    item = ds[0]
    print(f"  item keys: {sorted(item.keys())}")
    print(f"  image {tuple(item['image'].shape)} sim_param {tuple(item['sim_param'].shape)} "
          f"sim_dim={ds.sim_dim} len={len(ds)}")
    sizes = [len(b) for b in ImageBatchSampler(ds.content_ids, batch_size=8, shuffle=True)]
    print(f"  sampler: {len(sizes)} batches, sizes min/max={min(sizes)}/{max(sizes)}")
    print("  OK: data pipeline")


def test_compression(device):
    print("\n=== compression (mask + sparse FD) ===")
    from utils.pruning import global_magnitude_mask, MaskManager, remap_mask
    from utils.augmentations import AugMixAugment, GaussianAugment, MultiDomainAugment
    H = 32
    bb = build_backbone("vit_s", img_size=H).to(device)
    mask = global_magnitude_mask(bb, 0.9)
    sp = 1 - sum(m.sum().item() for m in mask.values()) / sum(m.numel() for m in mask.values())
    print(f"  magnitude mask sparsity (target 0.9): {sp:.3f}")

    def builder():
        return build_backbone("vit_s", img_size=H).to(device)
    ex = torch.zeros(2, 3, H, H, device=device)
    model = FeatureDisentangled(builder, ex, 3 * H * H, sim_dim=8, num_classes=10,
                                latent_dim=64, sim_embed_dim=32, hidden_dim=128, recon_hidden_dim=256).to(device)
    combined = {**remap_mask(mask, "shared_encoder"), **remap_mask(mask, "private_encoder")}
    mm = MaskManager(model, combined)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    run_epoch_fd(model, [fake_batch(device=device)], nn.CrossEntropyLoss(), device,
                 optimizer=opt, desc="sparse-fd", mask_manager=mm)
    print(f"  encoder sparsity kept after a step: {mm.sparsity()[0]:.3f}")

    x = torch.rand(3, H, H)
    md = MultiDomainAugment([("augmix", AugMixAugment()), ("gaussian", GaussianAugment())])
    mi, mp = md(x)
    print(f"  MultiDomainAugment sim_dim={mp.shape[0]} (expect {md.sim_dim})")
    print("  OK: compression")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    test_data_pipeline()
    for name in ["vit_s", "swin_t"]:
        test_backbone(name, device)
    test_compression(device)
    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
