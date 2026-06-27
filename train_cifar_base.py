
import os
import sys
import csv
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms.functional as TF
from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import build_backbone
import data_paths


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR10C_CORRUPTIONS = (
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
)


# ============================================================
# Dataset
# ============================================================

class CIFAR10Dataset(Dataset):
    """Clean CIFAR-10 with optional light standard augmentation (train)."""

    def __init__(self, images, labels, train=True):
        self.images = images
        self.labels = np.asarray(labels).astype(np.int64)
        self.train = train

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = torch.from_numpy(np.ascontiguousarray(self.images[idx])).permute(2, 0, 1).float() / 255.0
        if self.train:
            if random.random() < 0.5:
                img = TF.hflip(img)
            img = TF.pad(img, [4, 4, 4, 4], padding_mode="reflect")
            i, j = random.randint(0, 8), random.randint(0, 8)
            img = img[:, i:i + 32, j:j + 32]
        img = TF.normalize(img, CIFAR10_MEAN, CIFAR10_STD)
        return {"image": img, "label": torch.tensor(self.labels[idx], dtype=torch.long)}


def load_cifar10_raw(root, train):
    ds = torchvision.datasets.CIFAR10(root=root, train=train, download=data_paths.DOWNLOAD)
    return ds.data, np.array(ds.targets, dtype=np.int64)


def load_cifar10c(root, corruption, severity):
    arr = np.load(os.path.join(root, f"{corruption}.npy"))
    labels = np.load(os.path.join(root, "labels.npy"))
    lo, hi = (severity - 1) * 10000, severity * 10000
    return arr[lo:hi], labels[lo:hi]


# ============================================================
# Model
# ============================================================

class BackboneClassifier(nn.Module):
    """Backbone (-> [B, D]) + linear classification head."""

    def __init__(self, backbone, feature_dim, num_classes=10):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))


# ============================================================
# Utilities + training loop
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None, desc="train", grad_clip=0.0):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    loss_sum, count = 0.0, 0
    all_labels, all_preds = [], []

    pbar = tqdm(loader, desc=desc)
    for batch in pbar:
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        bs = y.size(0)
        loss_sum += loss.item() * bs
        count += bs
        all_labels.extend(y.detach().cpu().numpy())
        all_preds.extend(torch.argmax(logits, 1).detach().cpu().numpy())
        pbar.set_postfix(loss=loss_sum / count)

    return {
        "loss": loss_sum / count,
        "cls_loss": loss_sum / count,
        "acc": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "macro_precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "macro_recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="vit_s", choices=["vit_s", "swin_t"])
    parser.add_argument("--data_root", type=str, default=data_paths.DATA_ROOT)
    parser.add_argument("--cifar10c_root", type=str, default=data_paths.CIFAR10C_ROOT)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_cifar_base")
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--vit_patch_size", type=int, default=4)
    parser.add_argument("--swin_patch_size", type=int, default=2)
    parser.add_argument("--swin_window_size", type=int, default=4)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_x, train_y = load_cifar10_raw(args.data_root, train=True)
    test_x, test_y = load_cifar10_raw(args.data_root, train=False)

    train_loader = DataLoader(CIFAR10Dataset(train_x, train_y, train=True),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(CIFAR10Dataset(test_x, test_y, train=False),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    backbone = build_backbone(args.backbone, img_size=args.img_size, drop_rate=args.dropout,
                              vit_patch_size=args.vit_patch_size,
                              swin_patch_size=args.swin_patch_size,
                              swin_window_size=args.swin_window_size).to(device)
    model = BackboneClassifier(backbone, backbone.feature_dim, num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val, best_epoch, patience_counter = float("inf"), -1, 0
    ckpt_path = os.path.join(args.save_dir, f"best_cifar_base_{args.backbone}.pt")
    metric_keys = ["epoch", "loss", "cls_loss", "acc", "macro_f1", "macro_precision", "macro_recall"]
    train_log, val_log = [], []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tm = run_epoch(model, train_loader, criterion, device, optimizer, "train", args.grad_clip)
        vm = run_epoch(model, val_loader, criterion, device, None, "val")
        print(f"Train | loss={tm['loss']:.5f} acc={tm['acc']:.5f}")
        print(f"Val   | loss={vm['loss']:.5f} acc={vm['acc']:.5f}")
        train_log.append({"epoch": epoch, **tm})
        val_log.append({"epoch": epoch, **vm})

        if vm["cls_loss"] < best_val:
            best_val, best_epoch, patience_counter = vm["cls_loss"], epoch, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "script_args": vars(args), "val_metrics": vm}, ckpt_path)
            print(f"Saved best checkpoint to: {ckpt_path}")
        else:
            patience_counter += 1
            print(f"No val cls improvement. Patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print("Early stopping.")
                break

    print(f"\nTraining finished. Best epoch: {best_epoch} | best val cls: {best_val:.5f}")
    for log, name in [(train_log, "train_metrics.csv"), (val_log, "val_metrics.csv")]:
        with open(os.path.join(args.save_dir, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=metric_keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(log)

    # CIFAR-10-C OOD evaluation.
    if args.cifar10c_root and os.path.isdir(args.cifar10c_root):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        accs = []
        print("\n[CIFAR-10-C] OOD evaluation:")
        for corr in CIFAR10C_CORRUPTIONS:
            for sev in range(1, 6):
                try:
                    imgs, labels = load_cifar10c(args.cifar10c_root, corr, sev)
                except FileNotFoundError:
                    continue
                loader = DataLoader(CIFAR10Dataset(imgs, labels, train=False),
                                    batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers)
                m = run_epoch(model, loader, criterion, device, None, f"{corr}-s{sev}")
                accs.append(m["acc"])
        if accs:
            ood_mean = float(np.mean(accs))
            print(f"[CIFAR-10-C] mean OOD acc: {ood_mean:.4f}")
            # summary row for compare.py (method='base' = no FD)
            summary = {"method": "base", "backbone": args.backbone, "aug": "standard",
                       "target_sparsity": "", "encoder_sparsity": 0.0,
                       "clean_acc": round(float(ckpt["val_metrics"]["acc"]), 4),
                       "ood_mean_acc": round(ood_mean, 4)}
            with open(os.path.join(args.save_dir, "summary.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(summary.keys()))
                w.writeheader(); w.writerow(summary)
            print("SUMMARY:", summary)


if __name__ == "__main__":
    main()
