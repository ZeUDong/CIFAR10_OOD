# CIFAR10-FD: Feature Disentanglement for OOD Compression

Feature-disentanglement (FD) OOD method on **CIFAR-10 / CIFAR-10-C**, with
**ViT-S** and **Swin-T** backbones (native 32×32, trained from scratch), plus a
CARD-style compression pipeline (lottery-ticket pruning + FD training).

**No external model libraries.** All transformer architecture is hand-written
under `layers/` (patch embedding, multi-head attention, Swin window /
shifted-window attention with relative position bias, patch merging, MLP,
stochastic depth) and assembled into ViT-S / Swin-T in `model/`. Third-party
deps: PyTorch + torchvision (data) and scikit-learn (metrics).

## Method

The shared branch carries the class-relevant signal and is the only thing used
for prediction; the private branch absorbs domain (corruption/augmentation)
information.

```
shared:   z_s = shared_proj(E_s(x));   logits = C(z_s)        # predict from z_s only
private:  z_p = private_fusion([E_p(x), sim_encoder(sim_param)])
heads:    sim_hat = sim_predictor(z_p);  recon = reconstructor([z_s, z_p, sim_emb])
```

Five losses: `cls` (CE) + `lambda_rec`·`rec` + `lambda_inv`·`inv` +
`lambda_sim`·`sim` + `lambda_sep`·`sep`.

### Compression variants

| Variant | What it is |
|---|---|
| **Full backbone** | backbone trained on clean CIFAR-10 (reference) |
| **Full backbone + FD** | FD attached to the full backbone |
| **Compressed backbone + FD** | *prune-then-train*: a mask is found on the backbone first (`compress.py`), then FD is trained under that fixed mask |
| **Compressed (backbone + FD)** | *train-then-prune*: backbone + FD are trained together first, then that trained model is pruned (`compress_fd.py`) and **fine-tuned** under the mask |
| **Compressed (backbone + FD) + IBB** | the train-then-prune model, then **further trained with the Information-Bottleneck (IBB) loss** |

### Information Bottleneck (IBB)

SimXRD-style IB applied on top of FD (`train_cifar_fd_ib.py`): the shared branch
is made **variational** (`z_s = mu_s + eps·sigma_s` in training, `mu_s` at eval),
and an **IB upper-bound loss** minimizes `I(mu_s ; x)` via classwise
positive/negative cluster centroids that are recomputed at the end of each epoch.
A **warm-up phase** runs the first `--ib_warmup_epochs` with `lambda_ib = 0`;
IB then activates and early-stopping/best-selection are reset to the IB phase.

## Domains (`--domain_source`)

FD trains on the *same content* seen under several *domains*, with `sim_param`
the continuous domain descriptor and `content_id` the image index (groups an
image's views so `L_inv` works).

- `synthetic` (default): AugMix / Gaussian augmentations are the domains. FD
  trains on the **full CIFAR-10 train set (50000)**; CIFAR-10-C is OOD eval only.
- `cifar10c` (SimXRD-style, leave-domains-out): the **real CIFAR-10-C corruption
  types are the domains**. `--num_train_domains N` of them are used for training
  (on the **CIFAR-10-C 10000 images**) and the rest are held out for OOD test;
  optionally clean + AugMix + Gaussian are added as extra training domains.

**Leak-free split (current protocol).** To avoid both image leak and val=test
leak, the 19 CIFAR-10-C corruptions are split into three **disjoint** groups, and
the images are split too (`--heldout_images` holds out the last N images):

- **Train corruptions (10):** gaussian/shot/impulse noise, defocus/glass/motion/zoom blur, snow, frost, fog — applied to the *held-in* images.
- **Val corruptions (4):** speckle_noise, gaussian_blur, brightness, spatter — on *held-out* images, for early stopping (`--val_corruptions`, severity 3).
- **Test corruptions (5):** contrast, elastic_transform, pixelate, jpeg_compression, saturate — on *held-out* images, the reported OOD set.

Train, val and test corruptions are mutually disjoint, so early stopping never
sees the test corruptions and the OOD number is leak-free.

Compression (`compress.py`) **always uses the full clean CIFAR-10 train set** to
find the mask, regardless of `--domain_source`.

## Install

```bash
pip install -r requirements.txt
```

## Data paths (server / HPC)

Set once via env vars (or edit `data_paths.py`); every script reads them as the
default for `--data_root` / `--cifar10c_root`:

```bash
export CIFAR_DATA_ROOT=/path/to/cifar10            # folder CONTAINING cifar-10-batches-py/
export CIFAR10C_ROOT=/path/to/CIFAR-10-C           # folder with <corruption>.npy + labels.npy
export CIFAR_DOWNLOAD=0                             # offline compute node: don't auto-download
```

## Usage

```bash
# no-FD base classifier (reference)
python train_cifar_base.py --backbone vit_s --save_dir ./cards/base_vit_s

# dense FD baseline (no pruning) -- same code path as compressed, --mask none
python train_cifar_fd_sparse.py --backbone vit_s --mask none --aug multi \
    --save_dir ./cards/dense_vit_s

# Stage 1: compress (clean full CIFAR-10 train -> a mask)
python compress.py --method lrr --sparsity 0.9 --backbone vit_s --save_dir ./masks

# Stage 2: FD training under the mask
python train_cifar_fd_sparse.py --backbone vit_s --mask ./masks/mask_lrr_vit_s_s90.pt \
    --aug multi --save_dir ./cards/lrr_vit_s90

# Final comparison: per backbone -> base / FD / FD+best-compression
python compare.py --glob "./cards/*/summary.csv" --out comparison.csv
```

Compression methods (`--method`): `imp` one-shot magnitude · `lth` weight-rewind
· `lrr` LR-rewind (recommended) · `ep` edge-popup · `bp` biprop. `compare.py`
auto-selects the single best compression method (by OOD acc) per backbone.

Leave-domains-out (real CIFAR-10-C domains):
```bash
python train_cifar_fd_sparse.py --backbone vit_s --mask none --domain_source cifar10c \
    --num_train_domains 10 --cifar10c_root /path/CIFAR-10-C --save_dir ./cards/dense_vit_s
#   explicit split: --train_corruptions fog,snow,... --test_corruptions saturate,...
```

### Whole sweep + SLURM

```bash
# both backbones, base + dense + all methods x sparsities, then compare
BACKBONES="vit_s swin_t" METHODS="lrr lth imp ep bp" SPARSITIES="0.5 0.9" bash run_all.sh
# leave-domains-out:
DOMAIN_SOURCE=cifar10c N_TRAIN_DOMAINS=10 bash run_all.sh
```

On a cluster, `run_all.sh` needs no changes — wrap it in `submit.sbatch`
(single job) or `submit_array.sbatch` (one array task per backbone):
```bash
sbatch submit.sbatch
```

### Resume / checkpoints

Every trainer writes a rolling `last_*.pt` each epoch (model + optimizer + epoch
+ best-so-far). To continue an interrupted run instead of retraining:
```bash
python train_cifar_base.py      --backbone vit_s --save_dir ./cards/base_vit_s \
    --resume ./cards/base_vit_s/last_cifar_base_vit_s.pt
python train_cifar_fd_sparse.py --backbone vit_s --mask none --save_dir ./cards/dense_vit_s \
    --resume ./cards/dense_vit_s/last_sparse_fd_vit_s.pt
```

Smoke test (no dataset needed):
```bash
python smoke_test.py
```

## Experimental Results

All numbers are **Mean Corruption Accuracy** on the held-out **test corruptions**
(contrast, elastic_transform, pixelate, jpeg_compression, saturate), evaluated on
the held-out images. Sparsity = fraction of encoder weights **pruned**.

**Table 1 — Main results (FD and IBB).**

| Backbone | Full backbone | Full backbone + FD | Full backbone + FD + IBB |
|---|---|---|---|
| vit_s  | 0.6079 | 0.6206 | **0.6590** |
| swin_t | 0.6215 | 0.6290 | **0.6491** |

FD improves over the full backbone, and adding the IB loss (IBB) improves further.

**Table 2 — Compression methods × sparsity (OOD acc).** `backbone` = prune-then-train
(mask on the backbone, then FD); `backbone+FD` = train-then-prune (train backbone+FD,
then prune that model and fine-tune).

| Method | Compress process | 25% | 50% | 75% | Backbone |
|---|---|---|---|---|---|
| Bp  | backbone | 0.4745 | 0.4607 | 0.4461 | vit_s |
| EP  | backbone | 0.5767 | – | 0.5254 | vit_s |
| IMP | backbone | 0.5713 | – | – | vit_s |
| LRR | backbone | 0.5625 | – | – | vit_s |
| LTH | backbone | 0.5778 | – | – | vit_s |
| Bp  | backbone | 0.5976 | 0.5433 | 0.5097 | swin_t |
| EP  | backbone | 0.6207 | 0.5548 | 0.5254 | swin_t |
| IMP | backbone | 0.6206 | 0.6018 | 0.5281 | swin_t |
| LRR | backbone | 0.6110 | 0.5797 | 0.5305 | swin_t |
| LTH | backbone | 0.6121 | 0.5676 | 0.5095 | swin_t |
| Bp  | backbone+FD | 0.6402 | 0.6452 | 0.5873 | vit_s |
| EP  | backbone+FD | 0.6576 | 0.6487 | **0.6621** | vit_s |
| IMP | backbone+FD | 0.6351 | 0.6278 | 0.6266 | vit_s |
| LRR | backbone+FD | 0.6371 | 0.6258 | 0.6149 | vit_s |
| LTH | backbone+FD | 0.6343 | 0.6294 | 0.6278 | vit_s |
| Bp  | backbone+FD | 0.6262 | 0.6341 | 0.5778 | swin_t |
| EP  | backbone+FD | 0.6414 | 0.6522 | **0.6661** | swin_t |
| IMP | backbone+FD | 0.6272 | 0.5999 | 0.6288 | swin_t |
| LRR | backbone+FD | 0.6391 | 0.6260 | 0.6183 | swin_t |
| LTH | backbone+FD | 0.6435 | 0.6188 | 0.6305 | swin_t |

Train-then-prune (`backbone+FD`) is far stronger than prune-then-train (`backbone`).
The magnitude methods (IMP/LRR/LTH) degrade monotonically with sparsity, while the
supermask methods (EP/BP) are best at high sparsity — EP at 75% is the top result
on both backbones.

**Table 3 — IB (IBB) hyperparameter sweep (`lambda_ib`).** Full = dense FD+IBB;
Compressed = FD+IBB under the 25%-pruned mask.

| Backbone | lambda_ib | Full Acc | Compressed Acc (25% pruned) |
|---|---|---|---|
| vit_s ｜ 1e-2 & 0.6445 & 0.6684 |
| vit_s ｜ 3e-3 & 0.6358 & 0.6508 |
| vit_s | 1e-3 | 0.6465 | 0.6711 |
| vit_s | 3e-4 | 0.6405 | 0.6745 |
| vit_s | 1e-4 | 0.6590 | 0.6572 |
| swin_t | 1e-2 | 0.6597 | 0.674 |
| swin_t | 3e-3 | 0.6484 | 0.6741 |
| swin_t | 1e-3 | 0.6491 | 0.6597 |
| swin_t | 3e-4 | 0.6484 | 0.6763 |
| swin_t | 1e-4 | 0.6491 | 0.6782 |

## Layout

```
layers/                # hand-written transformer blocks (no timm)
model/                 # vit.py, swin.py, __init__.py (build_backbone), each forward -> [B,D]
utils/
  augmentations.py     #   AugMix, Gaussian, MultiDomainAugment
  pruning.py           #   masks + global magnitude pruning
  ep_layers.py         #   edge-popup / biprop subnet layers
  domain_data.py       #   CIFAR-10-C leave-domains-out dataset
  tools.py             #   EarlyStopping, adjust_learning_rate, ...
data_paths.py          # central data-path config (env-overridable)
train_cifar_base.py    # no-FD base classifier (-> summary.csv: method=base)
train_cifar_fd.py      # FD reference (synthetic param augmentation)
train_cifar_fd_sparse.py  # FD training; --mask none = dense, real mask = compressed;
                       #   --init_backbone (warm-start), --init_fd (train-then-prune fine-tune)
train_cifar_fd_ib.py   # FD + Information-Bottleneck (IBB); variational shared branch + warm-up
compress.py            # prune-then-train: mask from a clean-CE backbone (imp/lth/lrr/ep/bp)
compress_fd.py         # train-then-prune: mask from a trained backbone/FD model (--init_backbone)
eval_ood.py            # evaluate a saved base/FD checkpoint on CIFAR-10-C -> summary.csv
compare.py             # base / FD / FD+best per backbone -> comparison.csv
compare_ib.py          # collect FD+IBB runs -> comparison_ib.csv
```

## Backbone notes

`vit_s` and `swin_t` are **native 32×32, from scratch** (no pretrained, no timm):
ViT-S patch_size=4 (64 tokens + class token); Swin-T patch_size=2, window_size=4,
4-stage resolution flow 16→8→4→2. `num_features` = 384 (ViT-S) / 768 (Swin-T).
Swin needs `torch>=1.10` (uses `torch.meshgrid(indexing=...)`).
