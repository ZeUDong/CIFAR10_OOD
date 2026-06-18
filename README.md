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
train_cifar_fd_sparse.py  # FD-native training; --mask none = dense, real mask = compressed
compress.py            # Stage 1: clean-data mask (imp/lth/lrr/ep/bp)
compare.py             # Stage 3: base / FD / FD+best per backbone -> comparison.csv
train_cifar_nas.py     # legacy Gumbel-NAS scaffold (superseded)
run_all.sh             # full sweep; submit.sbatch / submit_array.sbatch wrap it for SLURM
smoke_test.py
```

## Backbone notes

`vit_s` and `swin_t` are **native 32×32, from scratch** (no pretrained, no timm):
ViT-S patch_size=4 (64 tokens + class token); Swin-T patch_size=2, window_size=4,
4-stage resolution flow 16→8→4→2. `num_features` = 384 (ViT-S) / 768 (Swin-T).
Swin needs `torch>=1.10` (uses `torch.meshgrid(indexing=...)`).
