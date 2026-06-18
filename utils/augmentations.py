"""
CARD-paper augmentations (AugMix + Gaussian), wrapped to fit the FD dataset.

Both augmenters implement the same interface as ParameterizedAugment:
    __call__(img_chw_float_0_1) -> (aug_img_0_1, sim_param_float32)
so they drop straight into CIFAR10AugDomainDataset (sim_param is the continuous
"domain" descriptor used by the FD private branch / L_sim / L_rec).

MultiDomainAugment combines several augmenters into one FD multi-domain set.
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


# ------------------------------------------------------------
# AugMix operations (operate on a float CHW tensor in [0, 1])
# ------------------------------------------------------------

def _to_uint8(x):
    return (x.clamp(0, 1) * 255).round().to(torch.uint8)


def _to_float(x):
    return x.float() / 255.0


def _autocontrast(x, _):
    return TF.autocontrast(x)


def _equalize(x, _):
    return _to_float(TF.equalize(_to_uint8(x)))


def _posterize(x, level):
    bits = max(1, int(8 - round((level / 10.0) * 4)))   # severity -> fewer bits (8..4)
    return _to_float(TF.posterize(_to_uint8(x), bits))


def _solarize(x, level):
    thr = 1.0 - (level / 10.0)                   # severity -> lower threshold
    return TF.solarize(x, thr)


def _rotate(x, level):
    deg = (level / 10.0) * 30.0
    if np.random.rand() < 0.5:
        deg = -deg
    return TF.rotate(x, deg, interpolation=TF.InterpolationMode.BILINEAR)


def _shear_x(x, level):
    v = (level / 10.0) * 0.3
    if np.random.rand() < 0.5:
        v = -v
    return TF.affine(x, angle=0, translate=[0, 0], scale=1.0,
                     shear=[np.degrees(np.arctan(v)), 0.0],
                     interpolation=TF.InterpolationMode.BILINEAR)


def _shear_y(x, level):
    v = (level / 10.0) * 0.3
    if np.random.rand() < 0.5:
        v = -v
    return TF.affine(x, angle=0, translate=[0, 0], scale=1.0,
                     shear=[0.0, np.degrees(np.arctan(v))],
                     interpolation=TF.InterpolationMode.BILINEAR)


def _translate_x(x, level):
    _, _, W = x.shape
    v = int((level / 10.0) * 0.3 * W)
    if np.random.rand() < 0.5:
        v = -v
    return TF.affine(x, angle=0, translate=[v, 0], scale=1.0, shear=[0.0, 0.0],
                     interpolation=TF.InterpolationMode.BILINEAR)


def _translate_y(x, level):
    _, H, _ = x.shape
    v = int((level / 10.0) * 0.3 * H)
    if np.random.rand() < 0.5:
        v = -v
    return TF.affine(x, angle=0, translate=[0, v], scale=1.0, shear=[0.0, 0.0],
                     interpolation=TF.InterpolationMode.BILINEAR)


AUGMIX_OPS = [_autocontrast, _equalize, _posterize, _solarize, _rotate,
              _shear_x, _shear_y, _translate_x, _translate_y]


class AugMixAugment:
    """
    AugMix augmenter returning (mixed_image, sim_param).

    sim_param (dim = width + 2): [beta_mix_weight, dirichlet_w_1..width, mean_op_severity]
    """

    def __init__(self, severity=3, width=3, depth=-1, alpha=1.0):
        self.severity = severity
        self.width = width
        self.depth = depth          # -1 => random depth in [1, 3]
        self.alpha = alpha
        self.sim_dim = width + 2
        self._mean = np.array([0.5] + [1.0 / width] * width + [severity / 10.0],
                              dtype=np.float32)
        self._std = np.array([0.29] + [0.25] * width + [0.29], dtype=np.float32)

    def __call__(self, img_chw):
        ws = np.float32(np.random.dirichlet([self.alpha] * self.width))
        m = np.float32(np.random.beta(self.alpha, self.alpha))

        mix = torch.zeros_like(img_chw)
        sev_acc = 0.0
        for i in range(self.width):
            x_aug = img_chw.clone()
            d = self.depth if self.depth > 0 else np.random.randint(1, 4)
            for _ in range(d):
                op = AUGMIX_OPS[np.random.randint(len(AUGMIX_OPS))]
                level = np.random.uniform(1, self.severity)
                x_aug = op(x_aug, level).clamp(0, 1)
                sev_acc += level
            mix = mix + ws[i] * x_aug

        mixed = ((1 - m) * img_chw + m * mix).clamp(0, 1)
        mean_sev = sev_acc / max(1, self.width)
        raw = np.concatenate([[m], ws, [mean_sev / 10.0]]).astype(np.float32)
        sim_param = ((raw - self._mean) / self._std).astype(np.float32)
        return mixed, sim_param


class GaussianAugment:
    """Add N(0, sigma) noise with probability p. Returns (img, sim_param=[sigma, applied])."""

    def __init__(self, sigma=0.1, p=0.5):
        self.sigma = sigma
        self.p = p
        self.sim_dim = 2
        self._mean = np.array([sigma * p, p], dtype=np.float32)
        self._std = np.array([sigma, 0.5], dtype=np.float32)

    def __call__(self, img_chw):
        if np.random.rand() < self.p:
            img = (img_chw + torch.randn_like(img_chw) * self.sigma).clamp(0, 1)
            applied, sigma_used = 1.0, self.sigma
        else:
            img, applied, sigma_used = img_chw, 0.0, 0.0
        raw = np.array([sigma_used, applied], dtype=np.float32)
        sim_param = ((raw - self._mean) / self._std).astype(np.float32)
        return img, sim_param


class MultiDomainAugment:
    """
    FD multi-domain augmenter: per sample, randomly pick ONE of several
    augmentation schemes and apply it, returning a UNIFIED sim_param so a single
    FD model can treat several augmentation methods as distinct domains.

    sim_param layout (dim = K + max_sub_dim):
        [ one-hot domain id (K) , chosen scheme's params padded to max_sub_dim ]
    """

    def __init__(self, augmenters=None):
        if augmenters is None:
            augmenters = [("augmix", AugMixAugment()), ("gaussian", GaussianAugment())]
        self.names = [n for n, _ in augmenters]
        self.augs = [a for _, a in augmenters]
        self.K = len(self.augs)
        self.max_sub = max(a.sim_dim for a in self.augs)
        self.sim_dim = self.K + self.max_sub
        self.sim_mean = np.zeros(self.sim_dim, dtype=np.float32)
        self.sim_std = np.ones(self.sim_dim, dtype=np.float32)

    def __call__(self, img_chw):
        k = np.random.randint(self.K)
        aug_img, p = self.augs[k](img_chw)
        onehot = np.zeros(self.K, dtype=np.float32)
        onehot[k] = 1.0
        padded = np.zeros(self.max_sub, dtype=np.float32)
        padded[:len(p)] = p
        sim_param = np.concatenate([onehot, padded]).astype(np.float32)
        return aug_img, sim_param


# ------------------------------------------------------------
# AugMix Jensen-Shannon consistency loss
# ------------------------------------------------------------

def jsd_loss(logits_clean, logits_aug1, logits_aug2):
    """JSD consistency across clean + two AugMix views."""
    p_clean = F.softmax(logits_clean, dim=1)
    p_aug1 = F.softmax(logits_aug1, dim=1)
    p_aug2 = F.softmax(logits_aug2, dim=1)
    p_mix = torch.clamp((p_clean + p_aug1 + p_aug2) / 3.0, 1e-7, 1.0).log()
    return (
        F.kl_div(p_mix, p_clean, reduction="batchmean")
        + F.kl_div(p_mix, p_aug1, reduction="batchmean")
        + F.kl_div(p_mix, p_aug2, reduction="batchmean")
    ) / 3.0
