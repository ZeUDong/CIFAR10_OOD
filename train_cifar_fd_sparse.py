"""
Stage 2 -- FD training under a precomputed compression mask.

Loads a mask produced by compress.py, applies it to BOTH FD encoders, and
trains the FD-native multi-domain model under that fixed sparsity pattern (the
pruned weights stay zero throughout). Uses the combined AugMix + Gaussian
domains (--aug multi) and evaluates OOD on CIFAR-10-C. Produces one sparse
multi-domain model whose summary.csv the comparison stage (compare.py) consumes.
With --mask none it trains the dense FD-native baseline via the same code path.

The FD model / losses / epoch loop are reused from train_cifar_fd.py so the
method stays single-sourced; only the mask + augmentation differ.

Examples
--------
python train_cifar_fd_sparse.py --backbone vit_s --mask ./masks/mask_lrr_vit_s_s90.pt \
    --aug augmix --data_root ./data --cifar10c_root ./data/CIFAR-10-C \
    --save_dir ./cards/lrr_vit_s_augmix_s90
"""

import os
import sys
import csv
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import build_backbone
import data_paths
from train_cifar_fd import (
    FeatureDisentangled, run_epoch_fd, print_metrics, set_seed,
    CIFAR10AugDomainDataset, CIFAR10EvalDataset, ImageBatchSampler,
    load_cifar10_raw, load_cifar10c, evaluate_cifar10c, CIFAR10C_CORRUPTIONS,
)
from utils.augmentations import AugMixAugment, GaussianAugment, MultiDomainAugment
from utils.pruning import MaskManager, load_mask, remap_mask
from utils.domain_data import CIFAR10CDomainDataset, split_corruptions

METRIC_KEYS = ["epoch", "loss", "cls_loss", "rec_loss", "inv_loss", "sim_loss",
               "sep_loss", "acc", "macro_f1", "macro_precision", "macro_recall"]
LOSS_KEYS = {"loss", "cls_loss", "rec_loss", "inv_loss", "sim_loss", "sep_loss"}


def make_augment(name):
    if name == "augmix":
        return AugMixAugment()
    if name == "gaussian":
        return GaussianAugment()
    if name == "multi":
        # FD-native multi-domain: one model trained across AugMix + Gaussian.
        return MultiDomainAugment([("augmix", AugMixAugment()), ("gaussian", GaussianAugment())])
    raise ValueError("--aug must be augmix | gaussian | multi.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="vit_s", choices=["vit_s", "swin_t"])
    p.add_argument("--mask", default="none",
                   help="mask file from compress.py; 'none' = dense FD-native baseline (no pruning)")
    p.add_argument("--aug", default="multi", choices=["augmix", "gaussian", "multi"],
                   help="multi = FD-native multi-domain (recommended); augmix/gaussian = single domain")
    p.add_argument("--data_root", default=data_paths.DATA_ROOT)
    p.add_argument("--cifar10c_root", default=data_paths.CIFAR10C_ROOT)
    p.add_argument("--save_dir", default="./checkpoints_cifar_fd_sparse")
    p.add_argument("--sims_per_image", type=int, default=5)
    p.add_argument("--img_size", type=int, default=32)
    p.add_argument("--corruptions", default="")
    p.add_argument("--severities", default="1,2,3,4,5")
    # --- domain source: synthetic augmentation vs real CIFAR-10-C corruptions ---
    p.add_argument("--domain_source", default="synthetic", choices=["synthetic", "cifar10c"],
                   help="synthetic = AugMix/Gaussian as domains; "
                        "cifar10c = real CIFAR-10-C corruptions as domains (leave-domains-out)")
    p.add_argument("--num_train_domains", type=int, default=10,
                   help="cifar10c: how many CIFAR-10-C corruptions to use as TRAINING domains")
    p.add_argument("--train_corruptions", default="",
                   help="cifar10c: explicit comma list of training corruptions (overrides num_train_domains)")
    p.add_argument("--test_corruptions", default="",
                   help="cifar10c: explicit comma list of held-out test corruptions")
    p.add_argument("--val_corruptions", default="",
                   help="cifar10c: VALIDATION corruptions (disjoint from train/test) for early stopping")
    p.add_argument("--val_severities", default="3",
                   help="cifar10c: severities used for the OOD validation set")
    p.add_argument("--no_clean_domain", action="store_true", help="cifar10c: drop the clean domain")
    p.add_argument("--no_aug_domains", action="store_true", help="cifar10c: drop AugMix+Gaussian domains")
    p.add_argument("--init_backbone", default="", help="base checkpoint to warm-start BOTH FD encoders (faster)")
    p.add_argument("--init_fd", default="", help="dense FD checkpoint to warm-start the FULL FD model (train-then-prune fine-tuning)")
    p.add_argument("--heldout_images", type=int, default=2000, help="cifar10c: hold out the LAST N images for val/OOD (no content leak)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument("--no_grouped_batches", action="store_true")
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--sim_embed_dim", type=int, default=64)
    p.add_argument("--fd_hidden_dim", type=int, default=256)
    p.add_argument("--recon_hidden_dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lambda_rec", type=float, default=0.05)
    p.add_argument("--lambda_inv", type=float, default=0.1)
    p.add_argument("--lambda_sim", type=float, default=0.1)
    p.add_argument("--lambda_sep", type=float, default=0.01)
    p.add_argument("--vit_patch_size", type=int, default=4)
    p.add_argument("--swin_patch_size", type=int, default=2)
    p.add_argument("--swin_window_size", type=int, default=4)
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dense = args.mask in (None, "", "none", "None")
    if dense:
        mask, meta = None, {"method": "dense"}
        print("No mask -> training DENSE FD-native baseline (no pruning).")
    else:
        mask, meta = load_mask(args.mask)
        print(f"Loaded mask: {args.mask} | meta={meta}")

    # ---- Data: FD 'domains' ----
    # Validation is always clean classification-only (no sim_param), so it works
    # for any sim_param dim and tracks in-distribution accuracy.
    ood_idx = None
    if args.domain_source == "cifar10c":
        # SimXRD-style: real CIFAR-10-C corruptions as domains, leave-domains-out.
        if not args.cifar10c_root or not os.path.isdir(args.cifar10c_root):
            raise SystemExit("--domain_source cifar10c requires a valid --cifar10c_root")
        test_x, test_y = load_cifar10_raw(args.data_root, train=False)  # 10000 content images
        train_corr, test_corr = split_corruptions(
            args.num_train_domains,
            args.train_corruptions or None,
            args.test_corruptions or None,
        )
        sev = [int(s) for s in args.severities.split(",")]
        # image-level split: hold out the LAST N images so the model never trains
        # on them (removes content leak; val/OOD use unseen images).
        n_all = len(test_y)
        ho = max(0, min(args.heldout_images, n_all - 1))
        train_idx = list(range(0, n_all - ho))
        test_idx = list(range(n_all - ho, n_all))
        ood_idx = test_idx if ho > 0 else None
        train_ds = CIFAR10CDomainDataset(
            test_x, test_y, train_corr, args.cifar10c_root, severities=sev,
            use_clean=not args.no_clean_domain,
            use_augmix=not args.no_aug_domains, use_gaussian=not args.no_aug_domains,
            content_indices=train_idx,
        )
        # OOD validation set: held-out images under the VAL corruptions (disjoint
        # from train/test corruptions) -> early stopping tracks OOD, no leak.
        val_corr = [c.strip() for c in args.val_corruptions.split(",") if c.strip()]
        val_sev = [int(s) for s in args.val_severities.split(",")]
        if val_corr and ho > 0:
            ti = np.array(test_idx)
            vx_list, vy_list = [], []
            for c in val_corr:
                arr = np.load(os.path.join(args.cifar10c_root, f"{c}.npy"), mmap_mode="r")
                for s in val_sev:
                    blk = arr[(s - 1) * 10000:s * 10000]
                    vx_list.append(np.asarray(blk[ti]))
                    vy_list.append(test_y[ti])
            val_ds = CIFAR10EvalDataset(np.concatenate(vx_list), np.concatenate(vy_list))
            print(f"[cifar10c] VAL = {val_corr} x sev{val_sev} on {len(ti)} held-out imgs "
                  f"-> {len(np.concatenate(vy_list))} val images")
        elif ho > 0:
            val_ds = CIFAR10EvalDataset(test_x[test_idx], test_y[test_idx])  # clean fallback
        else:
            val_ds = CIFAR10EvalDataset(test_x, test_y)
        sim_dim = train_ds.sim_dim
        sim_mean, sim_std = train_ds.sim_mean, train_ds.sim_std
        args.corruptions = ",".join(test_corr)   # OOD eval = held-out corruptions
        print(f"[cifar10c] images: train {len(train_idx)} / held-out {len(test_idx)}")
        print(f"[cifar10c] train domains ({train_ds.K}): {train_ds.domain_names}")
        print(f"[cifar10c] held-out OOD corruptions ({len(test_corr)}): {test_corr}")
    else:
        # synthetic augmentation-as-domains (AugMix/Gaussian/multi): FD trains on
        # the FULL CIFAR-10 train set (50000 images); CIFAR-10-C is OOD eval only.
        train_x, train_y = load_cifar10_raw(args.data_root, train=True)
        test_x, test_y = load_cifar10_raw(args.data_root, train=False)
        train_aug = make_augment(args.aug)
        train_ds = CIFAR10AugDomainDataset(train_x, train_y, args.sims_per_image, train_aug)
        val_ds = CIFAR10EvalDataset(test_x, test_y)
        sim_dim = train_ds.sim_dim
        sim_mean, sim_std = train_aug.sim_mean, train_aug.sim_std

    if args.no_grouped_batches:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    else:
        sampler = ImageBatchSampler(train_ds.content_ids, args.batch_size, shuffle=True)
        train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=args.num_workers,
                                  pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    # ---- FD model ----
    def backbone_builder():
        return build_backbone(args.backbone, img_size=args.img_size, in_chans=3,
                              drop_rate=args.dropout, vit_patch_size=args.vit_patch_size,
                              swin_patch_size=args.swin_patch_size,
                              swin_window_size=args.swin_window_size).to(device)

    input_image = torch.zeros(2, 3, args.img_size, args.img_size, device=device)
    model = FeatureDisentangled(
        backbone_builder=backbone_builder, input_image=input_image,
        recon_dim=3 * args.img_size * args.img_size, sim_dim=sim_dim, num_classes=10,
        latent_dim=args.latent_dim, sim_embed_dim=args.sim_embed_dim,
        hidden_dim=args.fd_hidden_dim, recon_hidden_dim=args.recon_hidden_dim,
        dropout=args.dropout,
    ).to(device)

    # Optional warm-start: load a trained base classifier's backbone into BOTH
    # FD encoders (shared + private). Much faster than training ViT from scratch;
    # the FD heads (shared_proj/classifier/...) still init randomly.
    if args.init_backbone and os.path.isfile(args.init_backbone):
        _bsd = torch.load(args.init_backbone, map_location=device, weights_only=False)["model_state_dict"]
        _enc = {k[len("backbone."):]: v for k, v in _bsd.items() if k.startswith("backbone.")}
        model.shared_encoder.load_state_dict(_enc)
        model.private_encoder.load_state_dict(_enc)
        print(f"Warm-started both FD encoders from {args.init_backbone} ({len(_enc)} tensors)")

    # Train-then-prune: warm-start the FULL FD model from a trained dense FD
    # checkpoint, so this run fine-tunes the (now pruned) trained model.
    if args.init_fd and os.path.isfile(args.init_fd):
        _fsd = torch.load(args.init_fd, map_location=device, weights_only=False)["model_state_dict"]
        model.load_state_dict(_fsd)
        print(f"Warm-started FULL FD model from {args.init_fd} (train-then-prune fine-tune)")

    # Apply the backbone mask to BOTH encoders (heads stay dense). Dense => no mask.
    if dense:
        mm, sp = None, 0.0
    else:
        combined = {**remap_mask(mask, "shared_encoder"), **remap_mask(mask, "private_encoder")}
        mm = MaskManager(model, combined)
        sp, total = mm.sparsity()
        print(f"Applied mask to encoders | encoder sparsity={sp:.3f} over {total} weights")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    kw = dict(lambda_rec=args.lambda_rec, lambda_inv=args.lambda_inv,
              lambda_sim=args.lambda_sim, lambda_sep=args.lambda_sep)

    best_val, best_epoch, patience = float("inf"), -1, 0
    ckpt_path = os.path.join(args.save_dir, f"best_sparse_fd_{args.backbone}.pt")
    train_log, val_log = [], []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tm = run_epoch_fd(model, train_loader, criterion, device, optimizer=optimizer,
                          desc="train", grad_clip=args.grad_clip, mask_manager=mm, **kw)
        vm = run_epoch_fd(model, val_loader, criterion, device, optimizer=None, desc="val", **kw)
        print_metrics("Train", tm)
        print_metrics("Val  ", vm)
        train_log.append({"epoch": epoch, **tm})
        val_log.append({"epoch": epoch, **vm})

        if vm["cls_loss"] < best_val:
            best_val, best_epoch, patience = vm["cls_loss"], epoch, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "script_args": vars(args), "val_metrics": vm,
                        "mask_meta": meta, "encoder_sparsity": sp,
                        "sim_mean": sim_mean, "sim_std": sim_std,
                        "aug": args.aug}, ckpt_path)
            print(f"Saved best checkpoint to: {ckpt_path}")
        else:
            patience += 1
            print(f"No val cls improvement. Patience: {patience}/{args.patience}")
            if patience >= args.patience:
                print("Early stopping.")
                break

    print(f"\nDone. Best epoch {best_epoch} | best val cls {best_val:.5f} | sparsity {sp:.3f}")

    def write_csv(log, path):
        if not log:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=METRIC_KEYS, extrasaction="ignore")
            w.writeheader(); w.writerows(log)
    write_csv(train_log, os.path.join(args.save_dir, "train_metrics.csv"))
    write_csv(val_log, os.path.join(args.save_dir, "val_metrics.csv"))

    # ---- Final eval: clean + CIFAR-10-C OOD ----
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    clean = run_epoch_fd(model, val_loader, criterion, device, optimizer=None, desc="clean test", **kw)
    print_metrics("Clean Test", clean)
    ood_rows = evaluate_cifar10c(model, args, criterion, device, idx=ood_idx)
    if ood_rows:
        ood_path = os.path.join(args.save_dir, "cifar10c_ood_metrics.csv")
        with open(ood_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["corruption", "severity", "acc", "macro_f1", "cls_loss"])
            w.writeheader(); w.writerows(ood_rows)
            w.writerow({"corruption": "MEAN", "severity": "",
                        "acc": float(np.mean([r["acc"] for r in ood_rows])),
                        "macro_f1": float(np.mean([r["macro_f1"] for r in ood_rows])),
                        "cls_loss": float(np.mean([r["cls_loss"] for r in ood_rows]))})
        # one-line summary row for the cross-method comparison aggregator
        summary = {"method": meta.get("method", "?"), "backbone": args.backbone,
                   "aug": (f"cifar10c_{args.num_train_domains}d" if args.domain_source == "cifar10c"
                           else args.aug),
                   "target_sparsity": meta.get("target_sparsity", ""),
                   "encoder_sparsity": round(sp, 4),
                   "clean_acc": round(clean["acc"], 4),
                   "ood_mean_acc": round(float(np.mean([r["acc"] for r in ood_rows])), 4)}
        with open(os.path.join(args.save_dir, "summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary.keys()))
            w.writeheader(); w.writerow(summary)
        print("SUMMARY:", summary)


if __name__ == "__main__":
    main()
