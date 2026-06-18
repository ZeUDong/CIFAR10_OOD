"""
Evaluate an already-trained checkpoint on clean CIFAR-10 + CIFAR-10-C, WITHOUT
retraining. Works for both base (no-FD) and FD checkpoints, and writes the
summary.csv that compare.py consumes.

Use this when a training run finished but skipped OOD eval (e.g. CIFAR10C_ROOT
was not set). It reuses the saved best_*.pt.

Examples
--------
python eval_ood.py --ckpt ./cards/dense_vit_s/best_sparse_fd_vit_s.pt \
    --cifar10c_root /path/CIFAR-10-C
python eval_ood.py --ckpt ./cards/base_vit_s/best_cifar_base_vit_s.pt \
    --cifar10c_root /path/CIFAR-10-C
# or batch every checkpoint:
#   for c in ./cards/*/best_*.pt; do python eval_ood.py --ckpt "$c" --cifar10c_root /path/CIFAR-10-C; done
"""

import os
import sys
import csv
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import build_backbone
import data_paths
from train_cifar_fd import (
    FeatureDisentangled, CIFAR10EvalDataset, load_cifar10_raw, load_cifar10c,
    CIFAR10C_CORRUPTIONS,
)


def build_model_from_ckpt(ckpt, device):
    a = ckpt["script_args"]
    sd = ckpt["model_state_dict"]
    img = a.get("img_size", 32)
    is_fd = any(k.startswith("shared_encoder.") for k in sd)

    def bb():
        return build_backbone(a["backbone"], img_size=img, in_chans=3,
                              drop_rate=a.get("dropout", 0.1),
                              vit_patch_size=a.get("vit_patch_size", 4),
                              swin_patch_size=a.get("swin_patch_size", 2),
                              swin_window_size=a.get("swin_window_size", 4)).to(device)

    if is_fd:
        sim_dim = sd["sim_encoder.0.weight"].shape[1]
        ex = torch.zeros(2, 3, img, img, device=device)
        model = FeatureDisentangled(
            bb, ex, 3 * img * img, sim_dim, num_classes=10,
            latent_dim=a.get("latent_dim", 128), sim_embed_dim=a.get("sim_embed_dim", 64),
            hidden_dim=a.get("fd_hidden_dim", 256), recon_hidden_dim=a.get("recon_hidden_dim", 512),
            dropout=a.get("dropout", 0.1),
        ).to(device)
        model.load_state_dict(sd)
        model.eval()
        predict = model.forward_shared_only
        method = ckpt.get("mask_meta", {}).get("method", "dense")
        sparsity = float(ckpt.get("encoder_sparsity", 0.0))
        aug = ckpt.get("aug", a.get("aug", "multi"))
    else:
        backbone = bb()
        head = nn.Linear(backbone.feature_dim, 10).to(device)

        class BaseClf(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = backbone
                self.head = head

            def forward(self, x):
                return self.head(self.backbone(x))

        model = BaseClf().to(device)
        model.load_state_dict(sd)
        model.eval()
        predict = model.forward
        method, sparsity, aug = "base", 0.0, a.get("aug", "standard")

    return model, predict, a["backbone"], method, sparsity, aug


@torch.no_grad()
def acc_on(predict, loader, device):
    ys, ps = [], []
    for b in loader:
        x = b["image"].to(device)
        ys.extend(b["label"].numpy())
        ps.extend(torch.argmax(predict(x), 1).cpu().numpy())
    return accuracy_score(ys, ps), f1_score(ys, ps, average="macro", zero_division=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="best_*.pt checkpoint (base or FD)")
    p.add_argument("--data_root", default=data_paths.DATA_ROOT)
    p.add_argument("--cifar10c_root", default=data_paths.CIFAR10C_ROOT)
    p.add_argument("--corruptions", default="", help="comma list (default: standard 15)")
    p.add_argument("--severities", default="1,2,3,4,5")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out_dir", default="", help="where to write summary.csv (default: ckpt dir)")
    p.add_argument("--heldout_images", type=int, default=0, help="if >0, evaluate only the LAST N images (match cifar10c image split)")
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.ckpt))
    os.makedirs(out_dir, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model, predict, backbone, method, sparsity, aug = build_model_from_ckpt(ckpt, device)
    print(f"Loaded {args.ckpt} | backbone={backbone} method={method} sparsity={sparsity:.3f}")

    # ---- clean CIFAR-10 test ----
    test_x, test_y = load_cifar10_raw(args.data_root, train=False)
    if args.heldout_images > 0:
        test_x, test_y = test_x[-args.heldout_images:], test_y[-args.heldout_images:]
    clean_loader = DataLoader(CIFAR10EvalDataset(test_x, test_y), batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)
    clean_acc, clean_f1 = acc_on(predict, clean_loader, device)
    print(f"Clean test acc = {clean_acc:.4f}")

    # ---- CIFAR-10-C ----
    if not args.cifar10c_root or not os.path.isdir(args.cifar10c_root):
        print("[CIFAR-10-C] root not found; cannot run OOD eval. Set --cifar10c_root.")
        return
    corruptions = args.corruptions.split(",") if args.corruptions else list(CIFAR10C_CORRUPTIONS)
    severities = [int(s) for s in args.severities.split(",")]

    rows, accs = [], []
    print("[CIFAR-10-C] OOD evaluation:")
    for corr in corruptions:
        for sev in severities:
            try:
                imgs, labels = load_cifar10c(args.cifar10c_root, corr, sev)
                if args.heldout_images > 0:
                    imgs, labels = imgs[-args.heldout_images:], labels[-args.heldout_images:]
            except FileNotFoundError:
                print(f"  missing: {corr} (skipped)")
                continue
            loader = DataLoader(CIFAR10EvalDataset(imgs, labels), batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers)
            acc, f1 = acc_on(predict, loader, device)
            rows.append({"corruption": corr, "severity": sev, "acc": acc, "macro_f1": f1})
            accs.append(acc)
            print(f"  {corr:>18s} sev{sev}: acc={acc:.4f}")
    ood_mean = float(np.mean(accs)) if accs else 0.0
    print(f"[CIFAR-10-C] mean OOD acc over {len(accs)} settings: {ood_mean:.4f}")

    # ---- write csvs ----
    with open(os.path.join(out_dir, "cifar10c_ood_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["corruption", "severity", "acc", "macro_f1"])
        w.writeheader(); w.writerows(rows)
        w.writerow({"corruption": "MEAN", "severity": "", "acc": ood_mean,
                    "macro_f1": float(np.mean([r["macro_f1"] for r in rows])) if rows else 0.0})
    summary = {"method": method, "backbone": backbone, "aug": aug,
               "target_sparsity": ckpt.get("mask_meta", {}).get("target_sparsity", ""),
               "encoder_sparsity": round(sparsity, 4),
               "clean_acc": round(clean_acc, 4), "ood_mean_acc": round(ood_mean, 4)}
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader(); w.writerow(summary)
    print("SUMMARY:", summary)


if __name__ == "__main__":
    main()
