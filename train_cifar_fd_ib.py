"""
FD + Information Bottleneck (SimXRD-style IBB) for CIFAR-10 / CIFAR-10-C.

Adds the SimXRD IB method on top of the FD model:
  * the shared branch is made VARIATIONAL: z_s = mu_s + eps * sigma_s (train),
    z_s = mu_s (eval); classification uses z_s.
  * IBB loss = classwise variational upper bound on I(mu_s ; x), using
    positive/negative (K=2) centroids of the INPUT image and the OUTPUT shared
    latent. Centroids are recomputed at the END of each epoch on the train loader
    and reused (detached) by the next epoch.
  * WARM-UP: the first --ib_warmup_epochs run with lambda_ib = 0 (IB off); after
    that, IB activates using the previous epoch's centroids.

Reuses the cifar10c leave-domains-out protocol, the image/3-way-corruption split,
the compression mask (so IB can run on a pruned model, e.g. the TTP ep s75 mask)
and the base-backbone warm-start, exactly like train_cifar_fd_sparse.py.

Example
-------
python train_cifar_fd_ib.py --backbone vit_s --mask none --domain_source cifar10c \
    --train_corruptions ... --val_corruptions ... --test_corruptions ... \
    --latent_dim 384 --heldout_images 2000 --init_backbone <base.pt> \
    --lambda_ib 1e-3 --ib_warmup_epochs 30 --save_dir ./cards_ib/ib1e-3_dense_vit_s
"""

import os
import sys
import csv
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import build_backbone
import data_paths
from train_cifar_fd import (
    set_seed, make_mlp, separation_loss, same_content_invariance_loss,
    CIFAR10AugDomainDataset, CIFAR10EvalDataset, ImageBatchSampler,
    ParameterizedAugment, load_cifar10_raw, load_cifar10c, CIFAR10C_CORRUPTIONS,
)
from utils.pruning import MaskManager, load_mask, remap_mask
from utils.domain_data import CIFAR10CDomainDataset, split_corruptions


# ============================================================
# FD + IB model (variational shared branch)
# ============================================================

class FeatureDisentangledIB(nn.Module):
    def __init__(self, backbone_builder, input_image, recon_dim, sim_dim,
                 num_classes=10, latent_dim=128, sim_embed_dim=64, hidden_dim=256,
                 recon_hidden_dim=512, dropout=0.1, ib_logvar_min=-8.0, ib_logvar_max=8.0):
        super().__init__()
        self.recon_dim = recon_dim
        self.sim_dim = sim_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.ib_logvar_min = ib_logvar_min
        self.ib_logvar_max = ib_logvar_max

        self.shared_encoder = backbone_builder()
        self.private_encoder = backbone_builder()

        self.eval()
        with torch.no_grad():
            shared_raw = self.shared_encoder(input_image)
            private_raw = self.private_encoder(input_image)
        shared_raw_dim = shared_raw.shape[1]
        private_raw_dim = private_raw.shape[1]
        print(f"[FD-IB Model] shared raw dim: {shared_raw_dim} | private raw dim: {private_raw_dim}")

        # Variational shared head.
        self.shared_base = nn.Sequential(
            nn.Linear(shared_raw_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.shared_mu = nn.Linear(hidden_dim, latent_dim)
        self.shared_logvar = nn.Linear(hidden_dim, latent_dim)

        self.sim_encoder = make_mlp(sim_dim, hidden_dim, sim_embed_dim, dropout)
        self.private_fusion = make_mlp(private_raw_dim + sim_embed_dim, hidden_dim, latent_dim, dropout)
        self.classifier = make_mlp(latent_dim, hidden_dim, num_classes, dropout)
        self.sim_predictor = make_mlp(latent_dim, hidden_dim, sim_dim, dropout)
        self.reconstructor = nn.Sequential(
            nn.Linear(2 * latent_dim + sim_embed_dim, recon_hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(recon_hidden_dim, recon_dim),
        )

    def _shared_distribution(self, image):
        h = self.shared_base(self.shared_encoder(image))
        mu_s = self.shared_mu(h)
        logvar_s = torch.clamp(self.shared_logvar(h), self.ib_logvar_min, self.ib_logvar_max)
        return mu_s, logvar_s

    def _reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def forward_shared_only(self, image):
        mu_s, _ = self._shared_distribution(image)
        return self.classifier(mu_s)

    def forward(self, image, sim_param):
        mu_s, logvar_s = self._shared_distribution(image)
        z_s = self._reparameterize(mu_s, logvar_s)
        private_raw = self.private_encoder(image)
        sim_emb = self.sim_encoder(sim_param)
        z_p = self.private_fusion(torch.cat([private_raw, sim_emb], dim=1))
        logits = self.classifier(z_s)
        sim_hat = self.sim_predictor(z_p)
        recon = self.reconstructor(torch.cat([mu_s, z_p, sim_emb], dim=1))
        return {"logits": logits, "z_s": z_s, "mu_s": mu_s, "logvar_s": logvar_s,
                "z_p": z_p, "sim_emb": sim_emb, "sim_hat": sim_hat, "recon": recon}


# ============================================================
# IBB upper-bound loss (classwise, K=2 pos/neg centroids)
# ============================================================

def compute_cluster_scores(features, centroids, beta=1.0, normalize=True):
    if normalize:
        features = F.normalize(features, p=2, dim=1)
        centroids = F.normalize(centroids, p=2, dim=1)
    score = -beta * torch.cdist(features, centroids, p=2) ** 2
    return torch.softmax(score, dim=-1)


def estimate_q(scores_first, scores_second, eps=1e-9):
    denom = scores_second.sum(dim=0).clamp_min(eps)
    return (scores_first.transpose(0, 1) @ scores_second) / denom


def ib_upper_bound(input_scores, scores_first, scores_second, q, eps=1e-9):
    log_input = torch.log(input_scores.clamp_min(eps))
    log_q = torch.log(q.clamp_min(eps))
    term_1 = torch.einsum('ni,nj,nj->n', scores_first, input_scores, log_input)
    term_2 = torch.einsum('ni,nj,ij->n', scores_first, scores_second, log_q)
    return (term_1 - term_2).mean()


class IBUpperBoundLoss(nn.Module):
    def __init__(self, beta=1.0, symmetric=False, reduction='mean', eps=1e-9):
        super().__init__()
        self.beta = beta
        self.symmetric = symmetric
        self.reduction = reduction
        self.eps = eps

    def per_class(self, fa, fb, isa, isb, ca, cb):
        sa = compute_cluster_scores(fa, ca, beta=self.beta)
        sb = compute_cluster_scores(fb, cb, beta=self.beta)
        qa = estimate_q(sa, sb, eps=self.eps)
        la = ib_upper_bound(isa, sa, sb, qa, eps=self.eps)
        if not self.symmetric:
            return la, None
        qb = estimate_q(sb, sa, eps=self.eps)
        lb = ib_upper_bound(isb, sb, sa, qb, eps=self.eps)
        return la, lb

    def forward(self, features_a, features_b, input_scores_a, input_scores_b,
                centroids_a, centroids_b, class_mask=None):
        num_classes = centroids_a.shape[0]
        if class_mask is None:
            class_indices = range(num_classes)
        else:
            class_indices = torch.where(class_mask.detach().bool())[0].tolist()
        if len(class_indices) == 0:
            zero = features_a.new_zeros(())
            return (zero, zero) if self.symmetric else zero
        loss_a = features_a.new_zeros(())
        loss_b = features_a.new_zeros(())
        for cls in class_indices:
            la, lb = self.per_class(features_a, features_b,
                                    input_scores_a[:, cls], input_scores_b[:, cls],
                                    centroids_a[cls], centroids_b[cls])
            loss_a = loss_a + la
            if lb is not None:
                loss_b = loss_b + lb
        if self.reduction == 'mean':
            denom = len(class_indices)
            loss_a = loss_a / denom
            loss_b = loss_b / denom
        return (loss_a, loss_b) if self.symmetric else loss_a


def _flatten(x):
    return x.reshape(x.size(0), -1) if x.dim() > 2 else x


def _classwise_centroids(class_sums, class_counts, total_sum, total_count):
    num_classes = class_sums.shape[0]
    global_mean = total_sum / max(float(total_count), 1.0)
    centroids = []
    present_mask = class_counts > 0
    for cls in range(num_classes):
        pos_count = class_counts[cls].item()
        neg_count = total_count - pos_count
        pos = class_sums[cls] / pos_count if pos_count > 0 else global_mean
        neg = (total_sum - class_sums[cls]) / neg_count if neg_count > 0 else global_mean
        centroids.append(torch.stack([pos, neg], dim=0))
    return torch.stack(centroids, dim=0), present_mask


@torch.no_grad()
def compute_epoch_ib_centroids(model, loader, device, num_classes, desc="IB centroids"):
    was_training = model.training
    model.eval()
    in_sums = out_sums = in_total = out_total = None
    class_counts = torch.zeros(num_classes, device=device)
    total_count = 0
    for batch in tqdm(loader, desc=desc):
        image = batch["image"].to(device)
        labels = batch["label"].to(device).long()
        sim_param = batch["sim_param"].to(device) if "sim_param" in batch else None
        if sim_param is not None:
            mu_s = model(image, sim_param)["mu_s"]
        else:
            mu_s, _ = model._shared_distribution(image)
        in_feat = _flatten(image)
        out_feat = _flatten(mu_s)
        if in_sums is None:
            in_sums = torch.zeros(num_classes, in_feat.size(1), device=device)
            out_sums = torch.zeros(num_classes, out_feat.size(1), device=device)
            in_total = torch.zeros(in_feat.size(1), device=device)
            out_total = torch.zeros(out_feat.size(1), device=device)
        in_total += in_feat.sum(0)
        out_total += out_feat.sum(0)
        total_count += labels.size(0)
        for cls in range(num_classes):
            m = labels == cls
            c = int(m.sum().item())
            if c == 0:
                continue
            in_sums[cls] += in_feat[m].sum(0)
            out_sums[cls] += out_feat[m].sum(0)
            class_counts[cls] += c
    if total_count == 0:
        raise RuntimeError("empty loader for IB centroids")
    in_cent, class_mask = _classwise_centroids(in_sums, class_counts, in_total, total_count)
    out_cent, _ = _classwise_centroids(out_sums, class_counts, out_total, total_count)
    if was_training:
        model.train()
    return {"input_centroids": in_cent.detach(), "output_centroids": out_cent.detach(),
            "class_mask": class_mask.detach(), "class_counts": class_counts.detach()}


def build_classwise_input_scores(features, centroids, beta=1.0):
    features = _flatten(features)
    return torch.stack([compute_cluster_scores(features, centroids[c], beta=beta, normalize=True)
                        for c in range(centroids.size(0))], dim=1)


def ibb_loss(input_features, output_features, labels, ib_criterion, ib_centroids, num_classes):
    if ib_centroids is None:
        return output_features.new_zeros(())
    in_feat = _flatten(input_features)
    out_feat = _flatten(output_features)
    in_cent = ib_centroids["input_centroids"].to(out_feat.device)
    out_cent = ib_centroids["output_centroids"].to(out_feat.device)
    global_mask = ib_centroids["class_mask"].to(out_feat.device).bool()
    labels = labels.long()
    batch_mask = torch.zeros(num_classes, dtype=torch.bool, device=out_feat.device)
    present = torch.unique(labels)
    present = present[(present >= 0) & (present < num_classes)]
    if present.numel() > 0:
        batch_mask[present] = True
    class_mask = global_mask & batch_mask
    in_scores = build_classwise_input_scores(in_feat, in_cent, beta=ib_criterion.beta)
    return ib_criterion(features_a=out_feat, features_b=in_feat,
                        input_scores_a=in_scores, input_scores_b=in_scores,
                        centroids_a=out_cent, centroids_b=in_cent, class_mask=class_mask)


# ============================================================
# Epoch loop (IB-aware, mask-aware)
# ============================================================

METRIC_KEYS = ["epoch", "lambda_ib_eff", "loss", "cls_loss", "ib_loss", "rec_loss",
               "inv_loss", "sim_loss", "sep_loss", "acc", "macro_f1"]


def run_epoch_fd_ib(model, loader, criterion, device, optimizer=None, desc="train",
                    lambda_ib=0.0, lambda_rec=0.05, lambda_inv=0.1, lambda_sim=0.1,
                    lambda_sep=0.01, grad_clip=0.0, mask_manager=None,
                    ib_criterion=None, ib_centroids=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    sums = dict(loss=0.0, cls=0.0, ib=0.0, rec=0.0, inv=0.0, sim=0.0, sep=0.0)
    total = 0
    all_labels, all_preds = [], []
    for batch in tqdm(loader, desc=desc):
        image = batch["image"].to(device)
        labels = batch["label"].to(device)
        recon_target = image.reshape(image.size(0), -1)
        sim_param = batch["sim_param"].to(device) if "sim_param" in batch else None
        content_id = batch["content_id"].to(device) if "content_id" in batch else None
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            if sim_param is not None:
                out = model(image, sim_param)
                logits = out["logits"]
                loss_cls = criterion(logits, labels)
                if lambda_ib > 0.0 and ib_centroids is not None:
                    loss_ib = ibb_loss(recon_target, out["mu_s"], labels,
                                       ib_criterion, ib_centroids, model.num_classes)
                else:
                    loss_ib = logits.new_tensor(0.0)
                loss_rec = F.mse_loss(out["recon"], recon_target)
                loss_sim = F.mse_loss(out["sim_hat"], sim_param)
                loss_sep = separation_loss(out["mu_s"], out["z_p"])
                loss_inv = (same_content_invariance_loss(out["mu_s"], content_id)
                            if content_id is not None else logits.new_tensor(0.0))
                loss = (loss_cls + lambda_ib * loss_ib + lambda_rec * loss_rec
                        + lambda_inv * loss_inv + lambda_sim * loss_sim + lambda_sep * loss_sep)
            else:
                logits = model.forward_shared_only(image)
                loss_cls = criterion(logits, labels)
                loss_ib = loss_rec = loss_inv = loss_sim = loss_sep = logits.new_tensor(0.0)
                loss = loss_cls
            if is_train:
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if mask_manager is not None:
                    mask_manager.mask_grads()
                optimizer.step()
                if mask_manager is not None:
                    mask_manager.apply()
        bs = labels.size(0)
        sums["loss"] += loss.item() * bs; sums["cls"] += loss_cls.item() * bs
        sums["ib"] += float(loss_ib) * bs; sums["rec"] += loss_rec.item() * bs
        sums["inv"] += float(loss_inv) * bs; sums["sim"] += loss_sim.item() * bs
        sums["sep"] += loss_sep.item() * bs; total += bs
        all_labels.extend(labels.detach().cpu().numpy())
        all_preds.extend(torch.argmax(logits, 1).detach().cpu().numpy())
    return {"loss": sums["loss"] / total, "cls_loss": sums["cls"] / total,
            "ib_loss": sums["ib"] / total, "rec_loss": sums["rec"] / total,
            "inv_loss": sums["inv"] / total, "sim_loss": sums["sim"] / total,
            "sep_loss": sums["sep"] / total,
            "acc": accuracy_score(all_labels, all_preds),
            "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0)}


def print_metrics(prefix, m):
    print(f"{prefix} | loss={m['loss']:.4f} | cls={m['cls_loss']:.4f} | ib={m['ib_loss']:.4f} | "
          f"rec={m['rec_loss']:.4f} | inv={m['inv_loss']:.4f} | sim={m['sim_loss']:.4f} | "
          f"sep={m['sep_loss']:.4f} | acc={m['acc']:.4f} | f1={m['macro_f1']:.4f}")


@torch.no_grad()
def evaluate_cifar10c_ib(model, args, criterion, device, idx=None):
    if not args.cifar10c_root or not os.path.isdir(args.cifar10c_root):
        print("[CIFAR-10-C] root not found; skipping OOD eval.")
        return []
    corruptions = args.corruptions.split(",") if args.corruptions else list(CIFAR10C_CORRUPTIONS)
    severities = [int(s) for s in args.severities.split(",")]
    rows, accs = [], []
    for corr in corruptions:
        for sev in severities:
            try:
                imgs, labels = load_cifar10c(args.cifar10c_root, corr, sev)
                if idx is not None:
                    imgs, labels = imgs[idx], labels[idx]
            except FileNotFoundError:
                continue
            loader = DataLoader(CIFAR10EvalDataset(imgs, labels), batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers)
            m = run_epoch_fd_ib(model, loader, criterion, device, optimizer=None, desc=f"{corr}-s{sev}")
            rows.append({"corruption": corr, "severity": sev, "acc": m["acc"],
                         "macro_f1": m["macro_f1"], "cls_loss": m["cls_loss"]})
            accs.append(m["acc"])
            print(f"  {corr:>18s} sev{sev}: acc={m['acc']:.4f}")
    if accs:
        print(f"[CIFAR-10-C] mean OOD acc over {len(accs)}: {np.mean(accs):.4f}")
    return rows


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="vit_s", choices=["vit_s", "swin_t"])
    p.add_argument("--mask", default="none", help="mask file from compress*.py; 'none' = dense")
    p.add_argument("--data_root", default=data_paths.DATA_ROOT)
    p.add_argument("--cifar10c_root", default=data_paths.CIFAR10C_ROOT)
    p.add_argument("--save_dir", default="./checkpoints_cifar_fd_ib")
    p.add_argument("--img_size", type=int, default=32)
    p.add_argument("--corruptions", default="")
    p.add_argument("--severities", default="1,2,3,4,5")
    # domain (cifar10c leave-domains-out, same as sparse trainer)
    p.add_argument("--domain_source", default="cifar10c", choices=["synthetic", "cifar10c"])
    p.add_argument("--num_train_domains", type=int, default=10)
    p.add_argument("--train_corruptions", default="")
    p.add_argument("--test_corruptions", default="")
    p.add_argument("--val_corruptions", default="")
    p.add_argument("--val_severities", default="3")
    p.add_argument("--no_clean_domain", action="store_true")
    p.add_argument("--no_aug_domains", action="store_true")
    p.add_argument("--init_backbone", default="", help="base checkpoint to warm-start BOTH encoders")
    p.add_argument("--heldout_images", type=int, default=2000)
    p.add_argument("--sims_per_image", type=int, default=5)
    # training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument("--no_grouped_batches", action="store_true")
    # FD dims
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--sim_embed_dim", type=int, default=64)
    p.add_argument("--fd_hidden_dim", type=int, default=256)
    p.add_argument("--recon_hidden_dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    # FD loss weights
    p.add_argument("--lambda_rec", type=float, default=0.05)
    p.add_argument("--lambda_inv", type=float, default=0.1)
    p.add_argument("--lambda_sim", type=float, default=0.1)
    p.add_argument("--lambda_sep", type=float, default=0.01)
    # IB
    p.add_argument("--lambda_ib", type=float, default=1e-3)
    p.add_argument("--ib_warmup_epochs", type=int, default=30)
    p.add_argument("--ib_beta", type=float, default=1.0)
    p.add_argument("--ib_reduction", default="mean", choices=["mean", "sum"])
    p.add_argument("--ib_eps", type=float, default=1e-9)
    p.add_argument("--ib_logvar_min", type=float, default=-8.0)
    p.add_argument("--ib_logvar_max", type=float, default=8.0)
    # backbone knobs
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
        print("No mask -> dense FD+IB.")
    else:
        mask, meta = load_mask(args.mask)
        print(f"Loaded mask: {args.mask} | meta={meta}")

    # ---- data: cifar10c leave-domains-out (image split + 3-way corruption split) ----
    ood_idx = None
    if args.domain_source == "cifar10c":
        if not args.cifar10c_root or not os.path.isdir(args.cifar10c_root):
            raise SystemExit("--domain_source cifar10c requires a valid --cifar10c_root")
        test_x, test_y = load_cifar10_raw(args.data_root, train=False)
        train_corr, test_corr = split_corruptions(args.num_train_domains,
                                                   args.train_corruptions or None,
                                                   args.test_corruptions or None)
        sev = [int(s) for s in args.severities.split(",")]
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
        val_corr = [c.strip() for c in args.val_corruptions.split(",") if c.strip()]
        val_sev = [int(s) for s in args.val_severities.split(",")]
        if val_corr and ho > 0:
            ti = np.array(test_idx)
            vx, vy = [], []
            for c in val_corr:
                arr = np.load(os.path.join(args.cifar10c_root, f"{c}.npy"), mmap_mode="r")
                for s in val_sev:
                    blk = arr[(s - 1) * 10000:s * 10000]
                    vx.append(np.asarray(blk[ti])); vy.append(test_y[ti])
            val_ds = CIFAR10EvalDataset(np.concatenate(vx), np.concatenate(vy))
            print(f"[cifar10c] VAL = {val_corr} x sev{val_sev} -> {len(np.concatenate(vy))} imgs")
        elif ho > 0:
            val_ds = CIFAR10EvalDataset(test_x[test_idx], test_y[test_idx])
        else:
            val_ds = CIFAR10EvalDataset(test_x, test_y)
        sim_dim = train_ds.sim_dim
        sim_mean, sim_std = train_ds.sim_mean, train_ds.sim_std
        args.corruptions = ",".join(test_corr)
        print(f"[cifar10c] images: train {len(train_idx)} / held-out {len(test_idx)}")
        print(f"[cifar10c] train domains ({train_ds.K}): {train_ds.domain_names}")
        print(f"[cifar10c] held-out OOD corruptions ({len(test_corr)}): {test_corr}")
    else:
        train_x, train_y = load_cifar10_raw(args.data_root, train=True)
        test_x, test_y = load_cifar10_raw(args.data_root, train=False)
        train_aug = ParameterizedAugment()
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

    def backbone_builder():
        return build_backbone(args.backbone, img_size=args.img_size, in_chans=3,
                              drop_rate=args.dropout, vit_patch_size=args.vit_patch_size,
                              swin_patch_size=args.swin_patch_size,
                              swin_window_size=args.swin_window_size).to(device)

    input_image = torch.zeros(2, 3, args.img_size, args.img_size, device=device)
    model = FeatureDisentangledIB(
        backbone_builder=backbone_builder, input_image=input_image,
        recon_dim=3 * args.img_size * args.img_size, sim_dim=sim_dim, num_classes=10,
        latent_dim=args.latent_dim, sim_embed_dim=args.sim_embed_dim,
        hidden_dim=args.fd_hidden_dim, recon_hidden_dim=args.recon_hidden_dim,
        dropout=args.dropout, ib_logvar_min=args.ib_logvar_min, ib_logvar_max=args.ib_logvar_max,
    ).to(device)

    if args.init_backbone and os.path.isfile(args.init_backbone):
        _bsd = torch.load(args.init_backbone, map_location=device, weights_only=False)["model_state_dict"]
        _enc = {k[len("backbone."):]: v for k, v in _bsd.items() if k.startswith("backbone.")}
        model.shared_encoder.load_state_dict(_enc)
        model.private_encoder.load_state_dict(_enc)
        print(f"Warm-started both encoders from {args.init_backbone} ({len(_enc)} tensors)")

    if dense:
        mm, sp = None, 0.0
    else:
        combined = {**remap_mask(mask, "shared_encoder"), **remap_mask(mask, "private_encoder")}
        mm = MaskManager(model, combined)
        sp, total = mm.sparsity()
        print(f"Applied mask | encoder sparsity={sp:.3f} over {total} weights")

    criterion = nn.CrossEntropyLoss()
    ib_criterion = IBUpperBoundLoss(beta=args.ib_beta, symmetric=False,
                                    reduction=args.ib_reduction, eps=args.ib_eps).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    kw = dict(lambda_rec=args.lambda_rec, lambda_inv=args.lambda_inv,
              lambda_sim=args.lambda_sim, lambda_sep=args.lambda_sep)

    best_val, best_epoch, patience = float("inf"), -1, 0
    ckpt_path = os.path.join(args.save_dir, f"best_fd_ib_{args.backbone}.pt")
    train_log, val_log = [], []
    ib_centroids = None
    prev_ib_active = False
    print(f"\nIB: lambda_ib={args.lambda_ib} warmup={args.ib_warmup_epochs} beta={args.ib_beta}")

    for epoch in range(1, args.epochs + 1):
        cur_lambda_ib = 0.0 if (epoch <= args.ib_warmup_epochs or ib_centroids is None) else args.lambda_ib
        ib_active = cur_lambda_ib > 0.0
        # When IB first turns on, reset best/patience so the checkpoint we keep
        # (and early-stopping budget) come from the IB phase, not the warm-up.
        if ib_active and not prev_ib_active:
            print("== IB activated: resetting best_val/patience for the IB phase ==")
            best_val, patience = float("inf"), 0
        prev_ib_active = ib_active
        print(f"\nEpoch {epoch}/{args.epochs} | lambda_ib={cur_lambda_ib:g}")
        tm = run_epoch_fd_ib(model, train_loader, criterion, device, optimizer=optimizer,
                             desc="train", lambda_ib=cur_lambda_ib, grad_clip=args.grad_clip,
                             mask_manager=mm, ib_criterion=ib_criterion, ib_centroids=ib_centroids, **kw)
        vm = run_epoch_fd_ib(model, val_loader, criterion, device, optimizer=None, desc="val", **kw)
        # refresh centroids from train split for next epoch
        ib_centroids = compute_epoch_ib_centroids(model, train_loader, device, model.num_classes)
        print_metrics("Train", tm); print_metrics("Val  ", vm)
        train_log.append({"epoch": epoch, "lambda_ib_eff": cur_lambda_ib, **tm})
        val_log.append({"epoch": epoch, "lambda_ib_eff": cur_lambda_ib, **vm})

        if vm["cls_loss"] < best_val:
            best_val, best_epoch, patience = vm["cls_loss"], epoch, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "script_args": vars(args), "val_metrics": vm, "mask_meta": meta,
                        "encoder_sparsity": sp, "sim_mean": sim_mean, "sim_std": sim_std,
                        "lambda_ib": args.lambda_ib}, ckpt_path)
            print(f"Saved best -> {ckpt_path}")
        else:
            patience += 1
            print(f"No improvement. Patience {patience}/{args.patience}")
            # Only allow early stopping AFTER IB has activated, so the warm-up
            # phase can never end training before IB ever turns on.
            if ib_active and patience >= args.patience:
                print("Early stopping (IB phase)."); break

    def write_csv(log, path):
        if not log:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=METRIC_KEYS, extrasaction="ignore")
            w.writeheader(); w.writerows(log)
    write_csv(train_log, os.path.join(args.save_dir, "train_metrics.csv"))
    write_csv(val_log, os.path.join(args.save_dir, "val_metrics.csv"))

    # ---- final eval: clean + OOD ----
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    clean = run_epoch_fd_ib(model, val_loader, criterion, device, optimizer=None, desc="clean", **kw)
    print_metrics("Clean", clean)
    ood_rows = evaluate_cifar10c_ib(model, args, criterion, device, idx=ood_idx)
    if ood_rows:
        with open(os.path.join(args.save_dir, "cifar10c_ood_metrics.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["corruption", "severity", "acc", "macro_f1", "cls_loss"])
            w.writeheader(); w.writerows(ood_rows)
            w.writerow({"corruption": "MEAN", "severity": "",
                        "acc": float(np.mean([r["acc"] for r in ood_rows])),
                        "macro_f1": float(np.mean([r["macro_f1"] for r in ood_rows])),
                        "cls_loss": float(np.mean([r["cls_loss"] for r in ood_rows]))})
        method = (meta.get("method", "dense") + "_ib")
        summary = {"method": method, "backbone": args.backbone, "lambda_ib": args.lambda_ib,
                   "ib_warmup": args.ib_warmup_epochs,
                   "target_sparsity": meta.get("target_sparsity", ""),
                   "encoder_sparsity": round(sp, 4), "clean_acc": round(clean["acc"], 4),
                   "ood_mean_acc": round(float(np.mean([r["acc"] for r in ood_rows])), 4)}
        with open(os.path.join(args.save_dir, "summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary.keys()))
            w.writeheader(); w.writerow(summary)
        print("SUMMARY:", summary)


if __name__ == "__main__":
    main()
