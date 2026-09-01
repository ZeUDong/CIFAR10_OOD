# CIFAR OOD Experimental Report

**Author:** Zeyu  
**Date:** June 2026

---

## 1. Datasets and Evaluation Protocol

### 1.1 Datasets

#### CIFAR-10

CIFAR-10 contains 10 classes of 32 × 32 color images. In this experiment, the CIFAR-10 training split is used to train the backbone. A validation subset is held out from the training data for model selection.

- **Training set:** 45,000 images
- **Validation set:** 5,000 images
- **Clean test set:** the official CIFAR-10 test set with 10,000 images

#### CIFAR-10-C

CIFAR-10-C is a corruption robustness benchmark derived from CIFAR-10. It contains multiple corruption types, each with 5 severity levels.

In this experiment, we use the official CIFAR-10-C data and split it by corruption type into training, validation, and test corruptions. The goal is to evaluate whether the FD-trained model can generalize from seen corruption types to unseen corruption types.

#### Corrupted Training Data for FD

To train the FD module, we use a subset of official CIFAR-10-C corruption types as the corrupted training data.

**Training corruptions:**

- Gaussian noise
- Shot noise
- Impulse noise
- Defocus blur
- Glass blur
- Motion blur
- Zoom blur
- Snow
- Frost
- Fog

**Validation corruptions:**

- Speckle noise
- Gaussian blur
- Brightness
- Spatter

**Test corruptions:**

- Contrast
- Elastic transform
- Pixelate
- JPEG compression
- Saturate

For each corrupted image, we record:

- **Image ID:** identifier of the original CIFAR-10 test image, used for shared feature invariance across different corruption views in the training split
- **Corruption type:** corruption category, encoded as a one-hot vector or class index
- **Severity:** corruption severity level from 1 to 5, optionally normalized to `[0, 1]`
- **Label:** original class label

---

## 2. Method

### 2.1 Backbone Training

A clean image classification backbone is trained on the CIFAR-10 training split.

Candidate backbones include:

- **ViT-S**
- **Swin-T**

### 2.2 Model Compression

Model compression is implemented by applying pre-computed binary pruning masks to the backbone. Each mask specifies which weights or structural components are retained in the compressed model.

Different pruning methods and sparsity levels are evaluated for both ViT-S and Swin-T backbones.

The available pruning masks are organized by:

- Backbone
- Pruning method
- Sparsity level

The naming convention is:

```text
mask_{method}_{backbone}_s{sparsity}.pt
```

For example:

```text
mask_imp_vit_s_s50.pt
```

denotes the mask obtained by the **IMP** method for **ViT-S** at the **50% sparsity** level.

After applying a mask, the resulting compressed backbone is fine-tuned on CIFAR-10 before being used alone or combined with the FD module.

#### Mask-Based Model Compression Configuration

| Item | Value |
|---|---|
| Compression type | Mask-based pruning |
| Backbones | ViT-S, Swin-T |
| Sparsity levels | 25%, 50%, 75% |
| Pruning methods | BP (Biprop), EP (Edge-Popup), IMP (Iterative Magnitude Pruning), LRR (Learning Rate Rewinding), LTH (Lottery Ticket Hypothesis) |

### 2.3 Baselines

The report considers representative robustness methods that can serve as base strategies on top of which the proposed method can be applied.

| Method | Venue | Type | Stage | Core Idea |
|---|---|---|---|---|
| AugMix | ICLR 2020 | Data Augmentation | Train | Constructs several stochastic augmentation chains and mixes their outputs with the original image. A Jensen-Shannon consistency loss encourages predictions to remain consistent across clean and augmented views. |
| PixMix | CVPR 2022 | Data Augmentation | Train | Mixes training images with structurally complex auxiliary images such as fractals using simple image mixing operations, encouraging classifiers to rely less on fragile local statistics and improving multiple robustness measures. |
| PRIME | ECCV 2022 | Data Augmentation | Train | Generates diverse training transformations using simple maximum-entropy primitives, including spectral, color, and spatial transformations, followed by image mixing to improve robustness to unseen corruptions. |
| DAMP | NeurIPS 2024 | Weight Perturbation | Train | Applies random multiplicative perturbations directly to network weights during training. The perturbations mimic internal representation shifts induced by input corruptions and encourage flatter and more corruption-robust solutions. |
| DiffAug | NeurIPS 2024 | Diffusion Augmentation | Train | Generates robust training samples through a lightweight diffuse-and-denoise process consisting of one forward diffusion step followed by one reverse denoising step, exposing the classifier to semantic-preserving variations. |
| GFN | CVPR 2025 | Robust Training | Train | Generates adversarial perturbations toward semantically neighboring classes using their gradient information. Training against these neighboring-class perturbations improves class separation and robustness to natural distribution shifts. |
| SATA | CVPR 2025 | Token Processing | Inference | Measures spatial autocorrelation among ViT tokens using attention relationships, identifies tokens with abnormal spatial statistics, and merges selected tokens to suppress corruption-sensitive representations without additional training. |

---

## 3. Experimental Setup

### 3.1 Compared Methods

| Method | Description |
|---|---|
| Full backbone | Original model trained on clean CIFAR-10 |
| Full backbone + FD | FD module attached to the full backbone |
| Compressed backbone + FD | FD module attached to the compressed backbone |
| Compressed (backbone + FD) | Backbone and FD module are trained together first, then compressed jointly |
| Compressed (backbone + FD) + IBB | Backbone and FD module are trained together first, compressed jointly, and then further trained with IBB |

---

## 4. Main Results

### 4.1 CIFAR-10-C Main Results

**Metric:** Mean Corr. Acc., the average accuracy over all CIFAR-10-C corruption types and severity levels.

| Backbone | Full Backbone | Full Backbone + FD | Full Backbone + FD + IBB |
|---|---:|---:|---:|
| ViT-S | 0.6079 | 0.6206 | 0.6590 |
| Swin-T | 0.6215 | 0.6290 | 0.6491 |

### 4.2 ImageNet Main Results

> The original report labels this table as ImageNet results.

**Metric:** Mean Corr. Acc., the average accuracy over all ImageNet corruption types and severity levels.

| Backbone | Full Backbone | Full Backbone + FD | Full Backbone + FD + IBB |
|---|---:|---:|---:|
| ViT-S | 0.6293 | 0.7190 | 0.8828 (1e-3) |
| Swin-T | 0.5248 | 0.8212 | 0.8697 |

#### Edge-Popup Compression Results

| Backbone | EP 25% | EP 50% | EP 75% |
|---|---:|---:|---:|
| ViT-S | 0.8901 | 0.8522 | 0.7853 |
| Swin-T | 0.8921 | 0.8605 | 0.7712 |

#### Full Backbone + IBB vs. EP 25%

| Backbone | Full Backbone + IBB | Full Backbone + IBB (EP 25%) | Full Backbone + IBB (EP 25%) |
|---|---:|---:| ---:|
| ViT-S | 0.8892 | 0.8683 | 0.5003|
| Swin-T | 0.8672 | 0.8374 | 0.7267 |

---

## 5. CIFAR-10: Different Compression Methods

### 5.1 Backbone-Only Compression

| Method | Compression Process | 25% | 50% | 75% | Backbone |
|---|---|---:|---:|---:|---|
| BP | Backbone | 0.4745 | 0.4607 | 0.4461 | ViT-S |
| EP | Backbone | 0.5767 | - | 0.5254 | ViT-S |
| IMP | Backbone | 0.5713 | - | - | ViT-S |
| LRR | Backbone | 0.5625 | - | - | ViT-S |
| LTH | Backbone | 0.5778 | - | - | ViT-S |
| BP | Backbone | 0.5976 | 0.5433 | 0.5097 | Swin-T |
| EP | Backbone | 0.6207 | 0.5548 | 0.5254 | Swin-T |
| IMP | Backbone | 0.6206 | 0.6018 | 0.5281 | Swin-T |
| LRR | Backbone | 0.6110 | 0.5797 | 0.5305 | Swin-T |
| LTH | Backbone | 0.6121 | 0.5676 | 0.5095 | Swin-T |

### 5.2 Backbone + FD Compression

| Method | Compression Process | 25% | 50% | 75% | Backbone |
|---|---|---:|---:|---:|---|
| BP | Backbone + FD | 0.6402 | 0.6452 | 0.5873 | ViT-S |
| EP | Backbone + FD | 0.6576 | 0.6487 | 0.6621 | ViT-S |
| IMP | Backbone + FD | 0.6351 | 0.6278 | 0.6266 | ViT-S |
| LRR | Backbone + FD | 0.6371 | 0.6258 | 0.6149 | ViT-S |
| LTH | Backbone + FD | 0.6343 | 0.6294 | 0.6278 | ViT-S |
| BP | Backbone + FD | 0.6262 | 0.6341 | 0.5778 | Swin-T |
| EP | Backbone + FD | 0.6414 | 0.6522 | 0.6661 | Swin-T |
| IMP | Backbone + FD | 0.6272 | 0.5999 | 0.6288 | Swin-T |
| LRR | Backbone + FD | 0.6391 | 0.6260 | 0.6183 | Swin-T |
| LTH | Backbone + FD | 0.6435 | 0.6188 | 0.6305 | Swin-T |

---

## 6. ImageNet: Different Compression Methods

| Method | Compression Process | 25% | 50% | 75% | Backbone |
|---|---|---:|---:|---:|---|
| EP | Backbone | 0.8782 | 0.5696 | 0.3420 | ViT-S |
| EP | Backbone | 0.8617 | 0.3151 | 0.2573 | Swin-T |

---

## 7. IB Hyperparameter Study

### Results of IB with Different Hyperparameters

| Backbone | Hyperparameter | Full Acc | Compressed Acc (Pruned 25%) |
|---|---:|---:|---:|
| ViT-S | 1e-2 | 0.6445 | 0.6684 |
| ViT-S | 3e-3 | 0.6358 | 0.6508 |
| ViT-S | 1e-3 | 0.6465 | 0.6711 |
| ViT-S | 3e-4 | 0.6405 | 0.6745 |
| ViT-S | 1e-4 | 0.6590 | 0.6572 |
| Swin-T | 1e-2 | 0.6597 | 0.6740 |
| Swin-T | 3e-3 | 0.6484 | 0.6741 |
| Swin-T | 1e-3 | 0.6491 | 0.6597 |
| Swin-T | 3e-4 | 0.6484 | 0.6763 |
| Swin-T | 1e-4 | 0.6491 | 0.6782 |
