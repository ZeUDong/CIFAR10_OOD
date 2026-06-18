"""
CIFAR-10-C leave-domains-out dataset (SimXRD-style).

The "domains" are real corruption types. A chosen subset is used for training
and the rest are held out for OOD testing. Content (the underlying image) is
shared across domains -- exactly the SimXRD protocol where the same content
appears under different environments. Optionally the clean image and the
synthetic AugMix / Gaussian augmentations are added as extra training domains.

    content    = the 10000 CIFAR-10-C images (== CIFAR-10 test set, in order)
    domains    = {clean, augmix, gaussian} + N selected CIFAR-10-C corruptions
    sim_param  = [ one-hot domain id , severity/5 ]   (the FD domain descriptor)
    content_id = image index (groups an image's views across domains -> L_inv)

Note: this trains on the corrupted CIFAR-10-C *test images* (under the training
corruptions) and evaluates on the same images under the held-out corruptions.
It measures generalization to unseen corruption *types*, not unseen images.
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from .augmentations import AugMixAugment, GaussianAugment

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# Fixed ordering used for the default train/test split (grouped noise/blur/
# weather/digital). Matches the 19 files of standard CIFAR-10-C.
ALL_CORRUPTIONS = (
    "gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "gaussian_blur",
    "snow", "frost", "fog", "brightness", "spatter",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression", "saturate",
)


def split_corruptions(num_train, train_list=None, test_list=None, available=None):
    """
    Decide the train / test corruption split.

    Explicit `train_list` / `test_list` (comma-separated str or list) override
    `num_train`. Otherwise the first `num_train` of `available` (default
    ALL_CORRUPTIONS) are training domains and the rest are held-out test domains.
    """
    def _as_list(x):
        if x is None:
            return None
        return [c.strip() for c in x.split(",")] if isinstance(x, str) else list(x)

    pool = list(available) if available else list(ALL_CORRUPTIONS)
    train_list = _as_list(train_list)
    test_list = _as_list(test_list)

    train = [c for c in train_list if c in pool] if train_list else pool[:num_train]
    if test_list:
        test = [c for c in test_list if c in pool]
    else:
        test = [c for c in pool if c not in train]
    return train, test


def _to_chw(img_hwc_uint8):
    return torch.from_numpy(np.ascontiguousarray(img_hwc_uint8)).permute(2, 0, 1).float() / 255.0


def _norm(img):
    return TF.normalize(img, CIFAR10_MEAN, CIFAR10_STD)


class CIFAR10CDomainDataset(Dataset):
    """
    domains = {clean, augmix, gaussian (optional)} + train_corruptions.
    Returns image / label / sim_param / content_id for the FD trainer.
    """

    def __init__(self, clean_images, labels, train_corruptions, cifar10c_root,
                 severities=(1, 2, 3, 4, 5), use_clean=True, use_augmix=True,
                 use_gaussian=True, images_per_severity=10000, content_indices=None):
        self.clean = clean_images            # [N,32,32,3] uint8 (CIFAR-10 test)
        self.labels = np.asarray(labels).astype(np.int64)
        self.N = len(self.labels)
        self.content_indices = (list(range(self.N)) if content_indices is None
                                else [int(x) for x in content_indices])
        self.severities = list(severities)
        self.block = images_per_severity     # 10000 images per severity block

        self.synth = []
        if use_clean:
            self.synth.append("clean")
        if use_augmix:
            self.synth.append("augmix")
        if use_gaussian:
            self.synth.append("gaussian")
        self.augmix = AugMixAugment()
        self.gaussian = GaussianAugment()

        self.corruptions = list(train_corruptions)
        self.domain_names = self.synth + self.corruptions
        self.K = len(self.domain_names)
        self.sim_dim = self.K + 1            # one-hot domain id + severity scalar

        # memory-map corruption arrays so we don't load gigabytes into RAM
        self.corr_arrays = {
            c: np.load(os.path.join(cifar10c_root, f"{c}.npy"), mmap_mode="r")
            for c in self.corruptions
        }

        # logical index: (domain_idx, severity_or_0, image_idx)
        self.index = []
        for di, name in enumerate(self.domain_names):
            if name in ("clean", "augmix", "gaussian"):
                self.index.extend((di, 0, g) for g in self.content_indices)
            else:
                for sev in self.severities:
                    self.index.extend((di, sev, g) for g in self.content_indices)
        self.content_ids = np.array([t[2] for t in self.index], dtype=np.int64)

        # sim_param already meaningful (one-hot 0/1 + severity/5); no rescaling
        self.sim_mean = np.zeros(self.sim_dim, dtype=np.float32)
        self.sim_std = np.ones(self.sim_dim, dtype=np.float32)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        di, sev, i = self.index[idx]
        name = self.domain_names[di]
        if name == "clean":
            img = _to_chw(self.clean[i])
        elif name == "augmix":
            img, _ = self.augmix(_to_chw(self.clean[i]))
        elif name == "gaussian":
            img, _ = self.gaussian(_to_chw(self.clean[i]))
        else:
            arr = self.corr_arrays[name]
            img = _to_chw(arr[(sev - 1) * self.block + i])

        onehot = np.zeros(self.K, dtype=np.float32)
        onehot[di] = 1.0
        sim = np.concatenate([onehot, [sev / 5.0]]).astype(np.float32)

        return {
            "image": _norm(img.clamp(0.0, 1.0)),
            "label": torch.tensor(self.labels[i], dtype=torch.long),
            "sim_param": torch.from_numpy(sim),
            "content_id": torch.tensor(i, dtype=torch.long),
        }
