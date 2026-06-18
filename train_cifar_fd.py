
import os
import sys
import csv
import random
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision
import torchvision.transforms.functional as TF
from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Backbone (hand-written ViT-S / Swin-T, defined under model/ + layers/).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import build_backbone
import data_paths


# ============================================================
# Dataset
# ============================================================

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR10C_CORRUPTIONS = (
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
)


class ParameterizedAugment:
    """
    Samples a continuous augmentation-parameter vector and applies it
    deterministically. The vector *is* sim_param (the domain), the direct analog
    of the SimXRD continuous simulation parameters.

    Order (d_sim = 8): [brightness, contrast, saturation, hue, rotation, tx, ty, scale]
    Each parameter is uniform on its range and standardized by the analytic
    uniform mean/std (zero-mean, unit-std), mirroring the sim_param
    standardization in the original code.
    """

    PARAM_NAMES = ("brightness", "contrast", "saturation", "hue",
                   "rotation", "translate_x", "translate_y", "scale")

    def __init__(self, ranges=None, identity=False):
        self.ranges = ranges or {
            "brightness": (0.6, 1.4), "contrast": (0.6, 1.4), "saturation": (0.6, 1.4),
            "hue": (-0.1, 0.1), "rotation": (-15.0, 15.0),
            "translate_x": (-0.1, 0.1), "translate_y": (-0.1, 0.1), "scale": (0.8, 1.2),
        }
        self.identity = identity
        lows = np.array([self.ranges[n][0] for n in self.PARAM_NAMES], dtype=np.float32)
        highs = np.array([self.ranges[n][1] for n in self.PARAM_NAMES], dtype=np.float32)
        self.sim_mean = (lows + highs) / 2.0
        self.sim_std = (highs - lows) / np.sqrt(12.0)
        self.sim_std[self.sim_std < 1e-6] = 1.0
        self.sim_dim = len(self.PARAM_NAMES)
        self.identity_params = np.array([1, 1, 1, 0, 0, 0, 0, 1], dtype=np.float32)

    def _sample_params(self):
        if self.identity:
            return self.identity_params.copy()
        return np.array([random.uniform(*self.ranges[n]) for n in self.PARAM_NAMES],
                        dtype=np.float32)

    def standardize(self, p):
        return (p - self.sim_mean) / self.sim_std

    def __call__(self, img_chw):
        p = self._sample_params()
        img = img_chw
        if not self.identity:
            b, c, s, h, rot, tx, ty, sc = p.tolist()
            img = TF.adjust_brightness(img, max(b, 1e-3))
            img = TF.adjust_contrast(img, max(c, 1e-3))
            img = TF.adjust_saturation(img, max(s, 1e-3))
            img = TF.adjust_hue(img, float(np.clip(h, -0.5, 0.5)))
            _, H, W = img.shape
            img = TF.affine(img, angle=float(rot),
                            translate=[int(round(tx * W)), int(round(ty * H))],
                            scale=float(sc), shear=[0.0, 0.0],
                            interpolation=TF.InterpolationMode.BILINEAR)
            img = img.clamp(0.0, 1.0)
        return img, self.standardize(p).astype(np.float32)


def _normalize(img):
    return TF.normalize(img, CIFAR10_MEAN, CIFAR10_STD)


class CIFAR10AugDomainDataset(Dataset):
    """
    Augmentation-as-domains. Each clean image is expanded into
    sims_per_image augmented views (the "environments").

    item:
        image      [3,H,W]  normalised input / reconstruction target
        label      scalar
        sim_param  [d_sim]  standardized augmentation params
        content_id scalar   image index (== "crystal_id" in the original)
    """

    def __init__(self, images_uint8, labels, sims_per_image=5, augment=None):
        self.images = images_uint8
        self.labels = np.asarray(labels).astype(np.int64)
        self.sims_per_image = int(sims_per_image)
        self.augment = augment or ParameterizedAugment()
        self.N = len(self.labels)
        self.content_ids = np.repeat(np.arange(self.N, dtype=np.int64), self.sims_per_image)

    @property
    def sim_dim(self):
        return self.augment.sim_dim

    def __len__(self):
        return self.N * self.sims_per_image

    def __getitem__(self, idx):
        cid = idx // self.sims_per_image
        img = torch.from_numpy(np.ascontiguousarray(self.images[cid])).permute(2, 0, 1).float() / 255.0
        aug_img, sim_param = self.augment(img)
        return {
            "image": _normalize(aug_img),
            "label": torch.tensor(self.labels[cid], dtype=torch.long),
            "sim_param": torch.from_numpy(sim_param),
            "content_id": torch.tensor(cid, dtype=torch.long),
        }


class CIFAR10EvalDataset(Dataset):
    """Classification-only dataset (clean or CIFAR-10-C): image + label, no sim_param."""

    def __init__(self, images_uint8, labels):
        self.images = images_uint8
        self.labels = np.asarray(labels).astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = torch.from_numpy(np.ascontiguousarray(self.images[idx])).permute(2, 0, 1).float() / 255.0
        return {"image": _normalize(img),
                "label": torch.tensor(self.labels[idx], dtype=torch.long)}


def load_cifar10_raw(root, train):
    ds = torchvision.datasets.CIFAR10(root=root, train=train, download=data_paths.DOWNLOAD)
    return ds.data, np.array(ds.targets, dtype=np.int64)


def load_cifar10c(root, corruption, severity):
    arr = np.load(os.path.join(root, f"{corruption}.npy"))
    labels = np.load(os.path.join(root, "labels.npy"))
    lo, hi = (severity - 1) * 10000, severity * 10000
    return arr[lo:hi], labels[lo:hi]


# ============================================================
# Batch sampler: keep multiple views of the same image in batch
# ============================================================

class ImageBatchSampler(Sampler):
    """
    Group all augmented views of the same image into the same batch, so the
    same-content invariance loss has multiple views to work with. Direct analog
    of the original CrystalBatchSampler (crystal_id -> content_id).
    """

    def __init__(self, content_ids, batch_size, shuffle=True, drop_last=False):
        self.content_ids = np.asarray(content_ids)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.indices_by_content = defaultdict(list)
        for idx, cid in enumerate(self.content_ids):
            self.indices_by_content[int(cid)].append(idx)
        self.unique_contents = list(self.indices_by_content.keys())

    def __iter__(self):
        contents = self.unique_contents.copy()
        if self.shuffle:
            random.shuffle(contents)
        batch = []
        for cid in contents:
            inds = self.indices_by_content[cid]
            if len(batch) > 0 and len(batch) + len(inds) > self.batch_size:
                yield batch
                batch = []
            batch.extend(inds)
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        avg = np.mean([len(v) for v in self.indices_by_content.values()])
        per_batch = max(1, int(self.batch_size // max(1, avg)))
        return int(np.ceil(len(self.unique_contents) / per_batch))


# ============================================================
# Utilities
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_mlp(in_dim, hidden_dim, out_dim, dropout=0.1):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


# ============================================================
# Feature disentanglement model
# ============================================================

class FeatureDisentangled(nn.Module):
    """
    Simulation-conditioned feature disentanglement model.

    Shared branch:
        z_s = E_s(x)
        logits = C(z_s)

    Private branch:
        h_p = E_p(x)
        phi_s = SimEncoder(sim_param)
        z_p = PrivateFusion([h_p, phi_s])

    Prediction uses only z_s. The private branch is used for auxiliary
    disentanglement losses. Identical structure to the original
    FeatureDisentangledPatchTST; only the encoder is now a 2D image backbone.
    """

    def __init__(self, backbone_builder, input_image, recon_dim, sim_dim,
                 num_classes=10, latent_dim=128, sim_embed_dim=64,
                 hidden_dim=256, recon_hidden_dim=512, dropout=0.1):
        super().__init__()
        self.recon_dim = recon_dim
        self.sim_dim = sim_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        # Two independent image encoders.
        self.shared_encoder = backbone_builder()
        self.private_encoder = backbone_builder()

        # Infer encoder feature dimensions.
        self.eval()
        with torch.no_grad():
            shared_raw = self.shared_encoder(input_image)
            private_raw = self.private_encoder(input_image)
        shared_raw_dim = shared_raw.shape[1]
        private_raw_dim = private_raw.shape[1]
        print(f"[FD Model] shared encoder raw feature dim:  {shared_raw_dim}")
        print(f"[FD Model] private encoder raw feature dim: {private_raw_dim}")

        self.shared_proj = make_mlp(shared_raw_dim, hidden_dim, latent_dim, dropout)
        self.sim_encoder = make_mlp(sim_dim, hidden_dim, sim_embed_dim, dropout)
        self.private_fusion = make_mlp(private_raw_dim + sim_embed_dim, hidden_dim,
                                       latent_dim, dropout)
        self.classifier = make_mlp(latent_dim, hidden_dim, num_classes, dropout)
        self.sim_predictor = make_mlp(latent_dim, hidden_dim, sim_dim, dropout)
        self.reconstructor = nn.Sequential(
            nn.Linear(2 * latent_dim + sim_embed_dim, recon_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(recon_hidden_dim, recon_dim),
        )

    def forward_shared_only(self, image):
        z_s = self.shared_proj(self.shared_encoder(image))
        return self.classifier(z_s)

    def forward(self, image, sim_param):
        shared_raw = self.shared_encoder(image)
        private_raw = self.private_encoder(image)

        z_s = self.shared_proj(shared_raw)
        sim_emb = self.sim_encoder(sim_param)
        z_p = self.private_fusion(torch.cat([private_raw, sim_emb], dim=1))

        logits = self.classifier(z_s)
        sim_hat = self.sim_predictor(z_p)
        recon = self.reconstructor(torch.cat([z_s, z_p, sim_emb], dim=1))

        return {"logits": logits, "z_s": z_s, "z_p": z_p, "sim_emb": sim_emb,
                "sim_hat": sim_hat, "recon": recon}


# ============================================================
# Losses
# ============================================================

def separation_loss(z_s, z_p, eps=1e-6):
    if z_s.size(0) <= 1:
        return z_s.new_tensor(0.0)
    z_s = z_s - z_s.mean(dim=0, keepdim=True)
    z_p = z_p - z_p.mean(dim=0, keepdim=True)
    z_s = z_s / (z_s.std(dim=0, keepdim=True) + eps)
    z_p = z_p / (z_p.std(dim=0, keepdim=True) + eps)
    corr = torch.matmul(z_s.T, z_p) / z_s.size(0)
    return (corr ** 2).mean()


def same_content_invariance_loss(z_s, content_id):
    unique_ids = torch.unique(content_id)
    loss = z_s.new_tensor(0.0)
    count = 0
    for cid in unique_ids:
        mask = content_id == cid
        if mask.sum() < 2:
            continue
        z_group = z_s[mask]
        z_mean = z_group.mean(dim=0, keepdim=True)
        loss = loss + ((z_group - z_mean) ** 2).mean()
        count += 1
    if count == 0:
        return z_s.new_tensor(0.0)
    return loss / count


# ============================================================
# Training / evaluation
# ============================================================

def run_epoch_fd(model, loader, criterion, device, optimizer=None, desc="train",
                 lambda_rec=0.05, lambda_inv=0.1, lambda_sim=0.1, lambda_sep=0.01,
                 grad_clip=0.0, mask_manager=None):
    # mask_manager (optional): for sparse / pruned training. When provided, the
    # pruned weights' grads are zeroed before the step and the pruned weights
    # are re-zeroed after it, so the mask is preserved throughout training.
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss_sum = cls_loss_sum = rec_loss_sum = 0.0
    inv_loss_sum = sim_loss_sum = sep_loss_sum = 0.0
    total_count = 0
    all_labels, all_preds = [], []

    pbar = tqdm(loader, desc=desc)
    for batch in pbar:
        image = batch["image"].to(device)            # [B, 3, H, W]
        labels = batch["label"].to(device)           # [B]
        recon_target = image.reshape(image.size(0), -1)  # [B, C*H*W]

        has_sim = "sim_param" in batch
        has_content = "content_id" in batch
        sim_param = batch["sim_param"].to(device) if has_sim else None
        content_id = batch["content_id"].to(device) if has_content else None

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            if sim_param is not None:
                out = model(image, sim_param)
                logits = out["logits"]

                loss_cls = criterion(logits, labels)
                loss_rec = F.mse_loss(out["recon"], recon_target)
                loss_sim = F.mse_loss(out["sim_hat"], sim_param)
                loss_sep = separation_loss(out["z_s"], out["z_p"])
                if content_id is not None:
                    loss_inv = same_content_invariance_loss(out["z_s"], content_id)
                else:
                    loss_inv = logits.new_tensor(0.0)

                loss = (loss_cls + lambda_rec * loss_rec + lambda_inv * loss_inv
                        + lambda_sim * loss_sim + lambda_sep * loss_sep)
            else:
                logits = model.forward_shared_only(image)
                loss_cls = criterion(logits, labels)
                loss_rec = loss_sim = loss_sep = loss_inv = logits.new_tensor(0.0)
                loss = loss_cls

            if is_train:
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if mask_manager is not None:
                    mask_manager.mask_grads()
                optimizer.step()
                if mask_manager is not None:
                    mask_manager.apply()

        bs = labels.size(0)
        total_loss_sum += loss.item() * bs
        cls_loss_sum += loss_cls.item() * bs
        rec_loss_sum += loss_rec.item() * bs
        inv_loss_sum += loss_inv.item() * bs
        sim_loss_sum += loss_sim.item() * bs
        sep_loss_sum += loss_sep.item() * bs
        total_count += bs

        preds = torch.argmax(logits, dim=1)
        all_labels.extend(labels.detach().cpu().numpy())
        all_preds.extend(preds.detach().cpu().numpy())
        pbar.set_postfix(total=total_loss_sum / total_count, cls=cls_loss_sum / total_count)

    return {
        "loss": total_loss_sum / total_count,
        "cls_loss": cls_loss_sum / total_count,
        "rec_loss": rec_loss_sum / total_count,
        "inv_loss": inv_loss_sum / total_count,
        "sim_loss": sim_loss_sum / total_count,
        "sep_loss": sep_loss_sum / total_count,
        "acc": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "macro_precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "macro_recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
    }


def print_metrics(prefix, metrics):
    print(
        f"{prefix} | "
        f"loss={metrics['loss']:.5f} | cls={metrics['cls_loss']:.5f} | "
        f"rec={metrics['rec_loss']:.5f} | inv={metrics['inv_loss']:.5f} | "
        f"sim={metrics['sim_loss']:.5f} | sep={metrics['sep_loss']:.5f} | "
        f"acc={metrics['acc']:.5f} | macro_f1={metrics['macro_f1']:.5f} | "
        f"macro_precision={metrics['macro_precision']:.5f} | "
        f"macro_recall={metrics['macro_recall']:.5f}"
    )


def evaluate_cifar10c(model, args, criterion, device, idx=None):
    if not args.cifar10c_root or not os.path.isdir(args.cifar10c_root):
        print("\n[CIFAR-10-C] root not found; skipping OOD evaluation.")
        return []
    corruptions = args.corruptions.split(",") if args.corruptions else list(CIFAR10C_CORRUPTIONS)
    severities = [int(s) for s in args.severities.split(",")]
    rows, accs = [], []
    print("\n[CIFAR-10-C] OOD evaluation:")
    for corr in corruptions:
        for sev in severities:
            try:
                imgs, labels = load_cifar10c(args.cifar10c_root, corr, sev)
                if idx is not None:
                    imgs, labels = imgs[idx], labels[idx]
            except FileNotFoundError:
                print(f"  missing: {corr} (skipped)")
                continue
            loader = DataLoader(CIFAR10EvalDataset(imgs, labels), batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers,
                                pin_memory=torch.cuda.is_available())
            m = run_epoch_fd(model, loader, criterion, device, optimizer=None, desc=f"{corr}-s{sev}")
            rows.append({"corruption": corr, "severity": sev, "acc": m["acc"],
                         "macro_f1": m["macro_f1"], "cls_loss": m["cls_loss"]})
            accs.append(m["acc"])
            print(f"  {corr:>18s} sev{sev}: acc={m['acc']:.4f} f1={m['macro_f1']:.4f}")
    if accs:
        print(f"[CIFAR-10-C] mean OOD acc over {len(accs)} settings: {np.mean(accs):.4f}")
    return rows


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    # Data / paths
    parser.add_argument("--backbone", type=str, default="vit_s", choices=["vit_s", "swin_t"])
    parser.add_argument("--data_root", type=str, default=data_paths.DATA_ROOT)
    parser.add_argument("--cifar10c_root", type=str, default=data_paths.CIFAR10C_ROOT)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_cifar_fd")

    # Domain construction
    parser.add_argument("--sims_per_image", type=int, default=5)
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--corruptions", type=str, default="")
    parser.add_argument("--severities", type=str, default="1,2,3,4,5")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--no_grouped_batches", action="store_true")

    # Feature disentanglement dimensions
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--sim_embed_dim", type=int, default=64)
    parser.add_argument("--fd_hidden_dim", type=int, default=256)
    parser.add_argument("--recon_hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Loss weights
    parser.add_argument("--lambda_rec", type=float, default=0.05)
    parser.add_argument("--lambda_inv", type=float, default=0.1)
    parser.add_argument("--lambda_sim", type=float, default=0.1)
    parser.add_argument("--lambda_sep", type=float, default=0.01)

    # Backbone knobs (native 32x32 from scratch)
    parser.add_argument("--vit_patch_size", type=int, default=4)
    parser.add_argument("--swin_patch_size", type=int, default=2)
    parser.add_argument("--swin_window_size", type=int, default=4)

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Loading CIFAR-10 from:", args.data_root)

    train_x, train_y = load_cifar10_raw(args.data_root, train=True)
    test_x, test_y = load_cifar10_raw(args.data_root, train=False)
    print("\ntrain:", train_x.shape, "test:", test_x.shape)
    print("train labels:", np.bincount(train_y, minlength=10))

    train_aug = ParameterizedAugment()
    train_dataset = CIFAR10AugDomainDataset(train_x, train_y, args.sims_per_image, train_aug)
    val_dataset = CIFAR10AugDomainDataset(test_x, test_y, 1, ParameterizedAugment(identity=True))
    sim_dim = train_dataset.sim_dim
    print(f"sim_param dim: {sim_dim} | train logical size: {len(train_dataset)}")

    if args.no_grouped_batches:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
        print("\nUsing normal shuffled training batches.")
    else:
        train_sampler = ImageBatchSampler(train_dataset.content_ids, args.batch_size,
                                          shuffle=True, drop_last=False)
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                                  num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
        print("\nUsing same-image grouped training batches.")

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    # Backbone builder (two independent encoders are created inside the model).
    def backbone_builder():
        return build_backbone(
            args.backbone, img_size=args.img_size, in_chans=3, drop_rate=args.dropout,
            vit_patch_size=args.vit_patch_size, swin_patch_size=args.swin_patch_size,
            swin_window_size=args.swin_window_size,
        ).to(device)

    input_image = torch.zeros(2, 3, args.img_size, args.img_size, device=device)
    recon_dim = 3 * args.img_size * args.img_size

    model = FeatureDisentangled(
        backbone_builder=backbone_builder, input_image=input_image, recon_dim=recon_dim,
        sim_dim=sim_dim, num_classes=10, latent_dim=args.latent_dim,
        sim_embed_dim=args.sim_embed_dim, hidden_dim=args.fd_hidden_dim,
        recon_hidden_dim=args.recon_hidden_dim, dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_cls_loss = float("inf")
    best_epoch, patience_counter = -1, 0
    best_ckpt_path = os.path.join(args.save_dir, f"best_cifar_fd_{args.backbone}.pt")

    metric_keys = ["epoch", "loss", "cls_loss", "rec_loss", "inv_loss", "sim_loss",
                   "sep_loss", "acc", "macro_f1", "macro_precision", "macro_recall"]
    loss_keys = {"loss", "cls_loss", "rec_loss", "inv_loss", "sim_loss", "sep_loss"}
    train_log, val_log = [], []

    print("\nLoss weights:")
    print("lambda_rec:", args.lambda_rec)
    print("lambda_inv:", args.lambda_inv)
    print("lambda_sim:", args.lambda_sim)
    print("lambda_sep:", args.lambda_sep)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        kw = dict(lambda_rec=args.lambda_rec, lambda_inv=args.lambda_inv,
                  lambda_sim=args.lambda_sim, lambda_sep=args.lambda_sep)

        train_metrics = run_epoch_fd(model, train_loader, criterion, device,
                                     optimizer=optimizer, desc="train",
                                     grad_clip=args.grad_clip, **kw)
        val_metrics = run_epoch_fd(model, val_loader, criterion, device,
                                   optimizer=None, desc="val", **kw)

        print_metrics("Train", train_metrics)
        print_metrics("Val  ", val_metrics)
        train_log.append({"epoch": epoch, **train_metrics})
        val_log.append({"epoch": epoch, **val_metrics})

        current_val_score = val_metrics["cls_loss"]
        if current_val_score < best_val_cls_loss:
            best_val_cls_loss = current_val_score
            best_epoch, patience_counter = epoch, 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "script_args": vars(args),
                "sim_mean": train_aug.sim_mean,
                "sim_std": train_aug.sim_std,
            }, best_ckpt_path)
            print(f"Saved best checkpoint to: {best_ckpt_path}")
        else:
            patience_counter += 1
            print(f"No val cls improvement. Patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print("Early stopping.")
                break

    print("\nTraining finished.")
    print("Best epoch:", best_epoch)
    print("Best val cls loss:", best_val_cls_loss)

    def write_metrics_csv(log, path):
        if not log:
            return
        best_row = {"epoch": "best"}
        for key in metric_keys[1:]:
            values = [row[key] for row in log]
            best_row[key] = min(values) if key in loss_keys else max(values)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metric_keys)
            writer.writeheader()
            writer.writerows(log)
            writer.writerow(best_row)
        print(f"Saved metrics CSV: {path}")

    write_metrics_csv(train_log, os.path.join(args.save_dir, "train_metrics.csv"))
    write_metrics_csv(val_log, os.path.join(args.save_dir, "val_metrics.csv"))

    # Final evaluation using best checkpoint: clean + CIFAR-10-C OOD.
    print("\nLoading best checkpoint for final evaluation...")
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    clean_metrics = run_epoch_fd(model, val_loader, criterion, device, optimizer=None, desc="clean test")
    print_metrics("Clean Test", clean_metrics)

    ood_rows = evaluate_cifar10c(model, args, criterion, device)
    if ood_rows:
        ood_path = os.path.join(args.save_dir, "cifar10c_ood_metrics.csv")
        with open(ood_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["corruption", "severity", "acc", "macro_f1", "cls_loss"])
            w.writeheader()
            w.writerows(ood_rows)
            w.writerow({"corruption": "MEAN", "severity": "",
                        "acc": float(np.mean([r["acc"] for r in ood_rows])),
                        "macro_f1": float(np.mean([r["macro_f1"] for r in ood_rows])),
                        "cls_loss": float(np.mean([r["cls_loss"] for r in ood_rows]))})
        print(f"Saved OOD metrics CSV: {ood_path}")


if __name__ == "__main__":
    main()
