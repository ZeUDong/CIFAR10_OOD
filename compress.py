"""
Stage 1 -- standalone compression. Produce a sparse subnetwork MASK for a
backbone, independently of FD.

Trains the backbone as a FEATURE EXTRACTOR on the CLEAN, FULL CIFAR-10 training
set (50000 images, no augmentation) to determine which weights to keep.
Compression is fit to the training distribution and never touches CIFAR-10-C.

Methods (--method): imp (one-shot magnitude) | lth (weight rewind) |
lrr (LR rewind) | ep (edge-popup) | bp (biprop). All emit the same mask format.

Output: mask_<method>_<backbone>_s<NN>.pt
"""

import os
import sys
import argparse
import random
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms.functional as TF
from tqdm import tqdm
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import build_backbone
import data_paths
from utils.pruning import (
    MaskManager, make_full_mask, global_magnitude_mask, save_mask,
)
from utils.ep_layers import convert_to_subnet, extract_mask_from_subnet

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _norm(img):
    return TF.normalize(img, CIFAR10_MEAN, CIFAR10_STD)


class CleanDataset(Dataset):
    """Clean CIFAR-10 (no augmentation): normalised image + label."""

    def __init__(self, images, labels):
        self.images = images
        self.labels = np.asarray(labels).astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = torch.from_numpy(np.ascontiguousarray(self.images[idx])).permute(2, 0, 1).float() / 255.0
        return {"image": _norm(img), "label": torch.tensor(self.labels[idx], dtype=torch.long)}


class Classifier(nn.Module):
    def __init__(self, backbone, num_classes=10):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.feature_dim, num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))


def load_cifar10_raw(root, train):
    ds = torchvision.datasets.CIFAR10(root=root, train=train, download=data_paths.DOWNLOAD)
    return ds.data, np.array(ds.targets, dtype=np.int64)


def train_epochs(model, loader, device, epochs, lr, weight_decay, params=None,
                 mask_manager=None, desc="train"):
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(params if params is not None else model.parameters(),
                     lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    model.train()
    for ep in range(epochs):
        pbar = tqdm(loader, desc=f"{desc} e{ep+1}/{epochs}")
        for batch in pbar:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            if mask_manager is not None:
                mask_manager.mask_grads()
            opt.step()
            if mask_manager is not None:
                mask_manager.apply()
            pbar.set_postfix(loss=float(loss))
        sched.step()


@torch.no_grad()
def quick_acc(model, loader, device):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        x = batch["image"].to(device)
        ys.extend(batch["label"].numpy())
        ps.extend(torch.argmax(model(x), 1).cpu().numpy())
    return accuracy_score(ys, ps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["imp", "lth", "lrr", "ep", "bp"])
    p.add_argument("--sparsity", type=float, required=True, help="target fraction pruned (e.g. 0.9)")
    p.add_argument("--backbone", default="vit_s", choices=["vit_s", "swin_t"])
    p.add_argument("--data_root", default=data_paths.DATA_ROOT)
    p.add_argument("--save_dir", default="./masks")
    p.add_argument("--img_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50, help="epochs per training stage/round")
    p.add_argument("--rounds", type=int, default=3, help="iterative rounds (lth/lrr)")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, "| method:", args.method, "| sparsity:", args.sparsity)
    print("Compression fits CLEAN full CIFAR-10 train (no augmentation).")

    train_x, train_y = load_cifar10_raw(args.data_root, train=True)
    test_x, test_y = load_cifar10_raw(args.data_root, train=False)
    train_loader = DataLoader(CleanDataset(train_x, train_y), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(CleanDataset(test_x, test_y), batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    def new_backbone():
        return build_backbone(args.backbone, img_size=args.img_size).to(device)

    mask = None

    if args.method in ("imp", "lth", "lrr"):
        model = Classifier(new_backbone()).to(device)
        theta0 = copy.deepcopy(model.state_dict())   # for LTH weight rewinding
        full_mask = make_full_mask(model.backbone, device=device)
        mm = MaskManager(model.backbone, full_mask)

        if args.method == "imp":
            train_epochs(model, train_loader, device, args.epochs, args.lr,
                         args.weight_decay, mask_manager=mm, desc="imp-train")
            mask = global_magnitude_mask(model.backbone, args.sparsity)
        else:
            mask = full_mask
            for r in range(1, args.rounds + 1):
                cum = 1.0 - (1.0 - args.sparsity) ** (r / args.rounds)  # cumulative target
                if args.method == "lth":
                    model.load_state_dict(theta0)        # rewind weights to init
                mm = MaskManager(model.backbone, mask)   # re-apply current mask after rewind
                train_epochs(model, train_loader, device, args.epochs, args.lr,
                             args.weight_decay, mask_manager=mm, desc=f"{args.method}-r{r}")
                mask = global_magnitude_mask(model.backbone, cum, prev_mask=mask)
                print(f"[round {r}] cumulative target sparsity = {cum:.3f}")

        MaskManager(model.backbone, mask)  # apply final mask
        print("final clean acc:", quick_acc(model, test_loader, device))

    else:  # ep / bp
        binary = args.method == "bp"
        backbone = convert_to_subnet(new_backbone(), keep_fraction=1.0 - args.sparsity,
                                     binary=binary).to(device)
        model = Classifier(backbone).to(device)
        score_params = [prm for n, prm in model.named_parameters() if prm.requires_grad]
        train_epochs(model, train_loader, device, args.epochs, args.lr,
                     args.weight_decay, params=score_params, desc=f"{args.method}-scores")
        mask = extract_mask_from_subnet(model.backbone)
        print("final clean acc:", quick_acc(model, test_loader, device))

    out = os.path.join(args.save_dir, f"mask_{args.method}_{args.backbone}_s{int(args.sparsity*100)}.pt")
    achieved = 1.0 - sum(m.sum().item() for m in mask.values()) / sum(m.numel() for m in mask.values())
    save_mask(mask, out, meta={"method": args.method, "backbone": args.backbone,
                               "target_sparsity": args.sparsity, "achieved_sparsity": achieved})
    print(f"Saved mask -> {out} | achieved sparsity = {achieved:.3f}")


if __name__ == "__main__":
    main()
