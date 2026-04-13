# Project I — I2I Translation & Input Space UDA: Technical Guide

## Table of Contents

1. [The Problem: Domain Shift](#1-the-problem-domain-shift)
2. [Input Space Adaptation](#2-input-space-adaptation)
3. [CycleGAN: The Core Engine](#3-cyclegan-the-core-engine)
   - [Architecture](#31-architecture)
   - [Losses](#32-losses)
   - [Training Tricks](#33-training-tricks)
4. [Spectral CycleGAN: Frequency-Domain Variant](#4-spectral-cyclegan-frequency-domain-variant)
   - [Fourier Transform Intuition](#41-fourier-transform-intuition)
   - [FFT Decomposition Pipeline](#42-fft-decomposition-pipeline)
   - [Pixel vs. Spectral Comparison](#43-pixel-vs-spectral-comparison)
5. [Deep Dives](#5-deep-dives)
   - [FFT Math](#51-fft-math)
   - [PatchGAN](#52-patchgan)
   - [InstanceNorm for Style Transfer](#53-instancenorm-for-style-transfer)
   - [LSGAN vs. Vanilla GAN](#54-lsgan-vs-vanilla-gan)
6. [Task I Deliverables](#6-task-i-deliverables)
7. [Task II Preview](#7-task-ii-preview)
8. [Code Map](#8-code-map)

---

## 1. The Problem: Domain Shift

You train a classifier on **source domain** images (e.g., product photos from Amazon), but at test time it sees **target domain** images (e.g., webcam photos). The distributions differ — lighting, texture, style, resolution — so the classifier performs poorly. This is **domain shift**.

**Unsupervised Domain Adaptation (UDA)** solves this *without* target labels. We have:
- Source domain: labeled images `{(x_s, y_s)}`
- Target domain: unlabeled images `{x_t}`
- Goal: learn a classifier that works well on target domain

---

## 2. Input Space Adaptation

Instead of modifying the model, modify the **input images**. Translate source images to *look like* target images, then train the classifier on translated data. At test time, the classifier sees real target images — which now match the training distribution.

```
Source Images ──→ Style Transfer ──→ "Fake Target" Images ──→ Train Classifier ──→ Evaluate on Real Target
                      ↑
               Target Images (no labels, just for style reference)
```

This is **input space UDA** — bridge the domain gap in pixel space before feeding into the model.

Alternative approaches (not input-space):
- **Feature-space UDA**: align feature distributions (e.g., domain-adversarial neural networks)
- **Output-space UDA**: align predictions across domains

Input-space is appealing because it's **model-agnostic** — any downstream classifier benefits.

---

## 3. CycleGAN: The Core Engine

**Problem**: We have unpaired images from two domains (A and B). We want to learn a mapping A→B and B→A without any paired examples.

**CycleGAN** [Zhu et al., ICCV 2017] solves this with 4 networks and 3 loss terms.

### 3.1 Architecture

#### Generator: ResNet-based

The generator follows an **encoder-transformer-decoder** pattern:

```
Input (3×256×256)
  │
  ├─ 7×7 Conv, InstanceNorm, ReLU          →  64 channels    (initial features)
  ├─ 3×3 Conv stride=2, IN, ReLU           → 128 channels    (downsample 1)
  ├─ 3×3 Conv stride=2, IN, ReLU           → 256 channels    (downsample 2)
  │
  ├─ ResNet Block × 9                      → 256 channels    (transform)
  │   Each block: ReflectPad → Conv → IN → ReLU → ReflectPad → Conv → IN + skip
  │
  ├─ 3×3 ConvTranspose stride=2, IN, ReLU  → 128 channels    (upsample 1)
  ├─ 3×3 ConvTranspose stride=2, IN, ReLU  →  64 channels    (upsample 2)
  └─ 7×7 Conv, Tanh                        →   3 channels    (output image)
```

Key design choices:
- **Reflection padding** (not zero padding): avoids border artifacts
- **InstanceNorm** (not BatchNorm): normalizes per-image, crucial for style transfer
- **Tanh** output: maps to [-1, 1] matching our image normalization
- **Skip connections** in ResNet blocks: preserves spatial information through the bottleneck

For small images (32×32 digits): fewer blocks (6), fewer filters (ngf=32), only 1 downsample/upsample.

#### Discriminator: PatchGAN

Instead of classifying the whole image as real/fake (a single scalar), PatchGAN outputs a **2D map** where each cell classifies a local 70×70 patch:

```
Input (3×256×256)
  │
  ├─ 4×4 Conv stride=2, LeakyReLU(0.2)     →  64 ch, 128×128
  ├─ 4×4 Conv stride=2, IN, LeakyReLU(0.2) → 128 ch, 64×64
  ├─ 4×4 Conv stride=2, IN, LeakyReLU(0.2) → 256 ch, 32×32
  ├─ 4×4 Conv stride=1, IN, LeakyReLU(0.2) → 512 ch, 31×31
  └─ 4×4 Conv stride=1                     →   1 ch, 30×30   (prediction map)
```

Each value in the 30×30 output = "is the corresponding 70×70 input patch real or fake?"

Advantages over a global discriminator:
- Fewer parameters → less overfitting
- Works on any image size
- Enforces local texture/style realism
- Effectively acts as a learned texture/style loss

### 3.2 Losses

CycleGAN uses three complementary losses:

#### Adversarial Loss (LSGAN)

Makes translated images indistinguishable from real images in the target domain.

```
L_GAN(G_AB, D_B) = E[(D_B(G_AB(a)) - 1)²]     (generator wants D to output 1)
L_GAN(D_B)       = E[(D_B(b) - 1)²] + E[D_B(G_AB(a))²]  (discriminator: real→1, fake→0)
```

Applied in both directions (A→B and B→A). Weight: 1.0.

#### Cycle-Consistency Loss

The key innovation. If you translate A→B→A, you should get back the original:

```
L_cyc = E[|G_BA(G_AB(a)) - a|₁] + E[|G_AB(G_BA(b)) - b|₁]
```

This prevents **mode collapse** (mapping all inputs to the same output) and ensures the translation preserves content. Weight: **λ_cyc = 10**.

Intuition: if G_AB maps a horse to a zebra, G_BA must be able to map that zebra back to the original horse. This forces G_AB to only change the "style" (stripes) while preserving the "content" (horse shape, pose).

#### Identity Loss

Regularization: if you feed a target-domain image to the generator, it should remain unchanged:

```
L_idt = E[|G_BA(a) - a|₁] + E[|G_AB(b) - b|₁]
```

Weight: **λ_idt = 5**. Helps preserve color composition and prevents unnecessary changes.

#### Total Generator Loss

```
L_G = L_GAN(G_AB, D_B) + L_GAN(G_BA, D_A)
    + λ_cyc × L_cyc
    + λ_idt × L_idt
```

### 3.3 Training Tricks

**Image Pool (Replay Buffer)**:
The discriminator doesn't just see the latest generated images — it sees a mix from a buffer of 50 recent images. With 50% probability, an old image is swapped in. This stabilizes training by preventing the discriminator from overfitting to the generator's current behavior.

**Learning Rate Schedule**:
```
Epoch  1 → N/2:   lr = 0.0002 (constant)
Epoch N/2 → N:    lr = 0.0002 × (1 - (epoch - N/2) / (N/2))  (linear decay to 0)
```

**Optimizer**: Adam with β₁=0.5 (not the usual 0.9), β₂=0.999. Lower β₁ is standard for GANs — reduces momentum, which helps with the adversarial dynamics.

**Weight Init**: Normal distribution, mean=0, std=0.02 for conv layers.

**Batch Size**: 1 for 256×256 images (standard for CycleGAN — InstanceNorm works best with batch size 1).

---

## 4. Spectral CycleGAN: Frequency-Domain Variant

### 4.1 Fourier Transform Intuition

Any image can be decomposed into frequency components:

```
                  Low Frequency                    High Frequency
              ┌─────────────────┐              ┌─────────────────┐
              │  Global color   │              │  Edges, lines   │
              │  Overall tone   │              │  Fine textures  │
              │  Lighting       │              │  Sharp details  │
              │  Smooth regions │              │  Noise          │
              │  "Style"        │              │  "Structure"    │
              └─────────────────┘              └─────────────────┘
```

The **2D FFT** converts an image from spatial domain (pixel values at x,y positions) to frequency domain (amplitude and phase at different spatial frequencies).

In the frequency domain representation:
- **Center** = low frequencies (slow spatial variations: color, brightness)
- **Edges** = high frequencies (rapid spatial variations: edges, texture)
- **Amplitude** = how much of each frequency
- **Phase** = where each frequency component is spatially located

The key insight from **FDA** [Yang et al., CVPR 2020]:

> Domain shift between real-world image domains is largely a **low-frequency phenomenon**. Two domains may differ in color palette, lighting, contrast (all low-freq), but edges and textures of objects remain similar.

### 4.2 FFT Decomposition Pipeline

```
Image (spatial domain)
  │
  ▼
2D FFT → Complex frequency map F(u,v)
  │
  ▼
fftshift → DC component at center
  │
  ├──── Circular mask (radius = β × min(H,W)/2) ────┐
  │                                                    │
  ▼                                                    ▼
Low-freq F_low = F × mask              High-freq F_high = F × (1 - mask)
  │                                                    │
  ▼                                                    │
ifftshift → IFFT → Low-freq image                     │
  │                                                    │
  ▼                                                    │
CycleGAN Generator                                     │  (unchanged)
  │                                                    │
  ▼                                                    ▼
Translated low-freq           +            Original high-freq
  │                                                    │
  └────────────────── ADD ─────────────────────────────┘
                       │
                       ▼
                Recombined output (clamp to [-1, 1])
```

**β (beta)** controls the cutoff radius:

| β value | What's included as "low-freq" | Effect |
|---------|-------------------------------|--------|
| 0.01 | Only DC + very coarse variations | Transfers only overall brightness/color |
| 0.05 | Global color palette + smooth gradients | Good default — transfers style without touching structure |
| 0.10 | Broader band, some coarse structure | More aggressive transfer, closer to pixel-space |
| 0.20 | Most of the image content | Almost pixel-space CycleGAN |

### 4.3 Pixel vs. Spectral Comparison

| Aspect | Pixel CycleGAN | Spectral CycleGAN |
|--------|---------------|-------------------|
| **What the generator sees** | Full image | Only low-freq spatial image (blurry version) |
| **What changes** | Everything — color, texture, edges, structure | Only low-freq content (color, lighting, tone) |
| **What's preserved** | Nothing guaranteed (cycle loss provides soft constraint) | High-freq exactly preserved (hard constraint) |
| **Flexibility** | High — can handle any domain gap | Limited to low-frequency domain gaps |
| **Artifact risk** | Can distort structure, introduce texture artifacts | Minimal — structure is locked by high-freq preservation |
| **Best for** | Large visual differences (color↔grayscale, photo↔sketch) | Lighting/color shifts (Amazon↔Webcam, day↔night) |
| **Failure mode** | Structure distortion, hallucinated textures | Cannot adapt when domain gap is in high frequencies |

**Per-dataset expectations:**

| Dataset Pair | Domain Gap Nature | Pixel CycleGAN | Spectral CycleGAN |
|---|---|---|---|
| MNIST → USPS | Minor stroke style | Good | Good (gap is small either way) |
| SVHN → MNIST | Color→grayscale + noise→clean | Better (needs to strip color + noise, which spans all frequencies) | Weaker (can only change low-freq, SVHN noise is high-freq) |
| Amazon → Webcam | Lighting, background, white-balance | Good | Better (gap is mostly low-freq lighting/color) |
| Photo → Sketch | Structure change (edges, fill patterns) | Better (sketch style is fundamentally high-freq) | Weaker (can't change edge structure) |

---

## 5. Deep Dives

### 5.1 FFT Math

The **2D Discrete Fourier Transform** of an image f(x,y) of size M×N:

```
F(u,v) = Σ_x Σ_y f(x,y) · exp(-j2π(ux/M + vy/N))
```

Where:
- `(x, y)` = spatial coordinates (pixel position)
- `(u, v)` = frequency coordinates
- `F(u,v)` = complex number with amplitude and phase

**Amplitude** `|F(u,v)|` = strength of frequency (u,v) in the image
**Phase** `∠F(u,v)` = spatial position/alignment of that frequency component

The **inverse** reconstructs the image perfectly:
```
f(x,y) = (1/MN) Σ_u Σ_v F(u,v) · exp(+j2π(ux/M + vy/N))
```

Important property: **Parseval's theorem** — energy is conserved between spatial and frequency domains. So splitting F into F_low + F_high and inverting each gives us two images that sum to the original (our `fft_decompose` + `fft_recombine` exploits this — verified lossless with max error ~7e-7).

**Why circular mask?** Frequency distance from center = `sqrt(u² + v²)`. A circular mask selects all frequencies below a certain spatial rate, regardless of direction. This is isotropic — it doesn't favor horizontal or vertical frequencies.

### 5.2 PatchGAN

Traditional discriminators output a single real/fake probability for the entire image. PatchGAN instead outputs a **grid of predictions**, each covering a local receptive field.

Why this matters for style transfer:

1. **Style is local**: realistic textures, lighting consistency, and color can be judged locally
2. **Content is global**: the overall structure/layout of the image is best judged at full-image level
3. CycleGAN already has **cycle-consistency loss** for content preservation → the discriminator can focus on local style realism

The 70×70 receptive field was empirically found to be the sweet spot — large enough to capture texture patterns, small enough to have many independent judgments per image.

Computation: the discriminator loss averages over all spatial locations in the output map:
```
L_D = (1/N) Σ_i [(D(real)_i - 1)² + D(fake)_i²]
```

This acts like an ensemble of many independent patch-level discriminators sharing weights.

### 5.3 InstanceNorm for Style Transfer

**BatchNorm**: normalize across the batch → `μ, σ` computed over (B, H, W) per channel
**InstanceNorm**: normalize per-instance → `μ, σ` computed over (H, W) per channel per image

```
InstanceNorm(x):  x̂ = (x - μ_instance) / σ_instance
```

Why InstanceNorm is critical for style transfer:

1. **Style information lives in feature statistics** (mean and variance of activations). This was shown by Gatys et al. (style transfer via Gram matrices) and later by AdaIN (Adaptive Instance Normalization).

2. InstanceNorm **removes the original style** (by normalizing away the per-instance mean/variance) and lets the network learn a new style through the learnable affine parameters γ, β.

3. With BatchNorm, the normalization statistics are shared across images in the batch → images influence each other's normalization → inconsistent per-image style transfer. With InstanceNorm, each image is independently normalized → consistent style transfer regardless of batch composition.

4. This is why CycleGAN traditionally uses **batch size 1** — at batch size 1, BatchNorm and InstanceNorm are identical. But InstanceNorm generalizes to any batch size.

### 5.4 LSGAN vs. Vanilla GAN

**Vanilla GAN** loss (binary cross-entropy):
```
L_D = -E[log D(real)] - E[log(1 - D(fake))]
L_G = -E[log D(fake)]
```

Problem: when D is confident (D(fake)≈0), the gradient for G vanishes → training stalls.

**LSGAN** loss (least-squares):
```
L_D = E[(D(real) - 1)²] + E[D(fake)²]
L_G = E[(D(fake) - 1)²]
```

Advantages:
1. **No vanishing gradients**: even when D(fake)≈0, the squared error still provides a strong gradient
2. **Penalizes overconfidence**: samples far from the decision boundary get large penalties → more stable training
3. **Higher quality images**: empirically produces sharper, more realistic outputs
4. **No sigmoid needed**: discriminator outputs raw scores, not probabilities

---

## 6. Task I Deliverables

### Report Contents

1. **Method Description** (~1 page)
   - Standard CycleGAN architecture and losses
   - Spectral variant: FFT decomposition, low-freq-only translation, recombination
   - Include a diagram of the spectral pipeline

2. **Experimental Setup** (~0.5 page)
   - 4 dataset pairs with sizes and preprocessing
   - Architecture configs (ngf, n_blocks, etc.)
   - Hyperparameters (lr, lambda_cyc, lambda_idt, beta)

3. **Qualitative Results** (~2 pages)
   - Side-by-side grids per dataset: Source | Pixel CycleGAN | Spectral CycleGAN | Target ref
   - Spectral decomposition visualization: Original | Low-freq | High-freq | Translated-low | Recombined
   - Frequency spectrum visualization (FFT magnitude plots)

4. **Quantitative Results** (~0.5 page)
   - FID table: lower = more realistic translation
   - Optional: classification accuracy if time permits

5. **Analysis and Discussion** (~1 page)
   - When does spectral outperform pixel and why?
   - Connection to frequency content of domain gap
   - Limitations of each approach

---

## 7. Task II Preview

Task II uses the style transfer from Task I as a **tool for UDA classification**. Benchmark 5 approaches:

| # | Method | How it works |
|---|--------|-------------|
| 1 | **Source-only** | Train on raw source, test on target. Baseline. |
| 2 | **CycleGAN** | Translate source→target with pixel CycleGAN, train classifier on translated images. |
| 3 | **Spectral CycleGAN** | Same but with spectral variant from Task I. |
| 4 | **CyCADA** [Hoffman 2018] | CycleGAN + semantic consistency loss (classifier predictions should be preserved through translation). |
| 5 | **FDA** [Yang 2020] | No learning — directly swap low-freq Fourier components from target into source images. |

```
Source (labeled) ───→ [Adaptation Method] ───→ Adapted Source ───→ Train Classifier ───→ Eval on Target
                            ↑
                     Target (unlabeled)
```

The metric is **classification accuracy** on the target test set.

---

## 8. Code Map

| Concept | File | Key class/function |
|---------|------|-------------------|
| ResNet Generator (G_AB, G_BA) | `models/generator.py` | `ResNetGenerator` |
| PatchGAN Discriminator (D_A, D_B) | `models/discriminator.py` | `PatchGANDiscriminator` |
| LSGAN + cycle + identity losses | `utils/losses.py` | `lsgan_loss_D/G`, `cycle_consistency_loss` |
| Image replay buffer | `utils/image_pool.py` | `ImagePool` |
| Full pixel-space training loop | `models/cyclegan.py` | `CycleGANTrainer.train_step()` |
| FFT decompose / recombine | `utils/spectral.py` | `fft_decompose()`, `fft_recombine()` |
| Spectral variant training | `models/spectral_cyclegan.py` | `SpectralCycleGANTrainer` |
| Unpaired dataset loading | `data/datasets.py` | `UnpairedDataset`, `build_dataset()` |
| Training with wandb logging | `train.py` | `main()` |
| Inference & image generation | `test.py` | `main()` |

---

## References

- [1] Zhu et al., "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks", ICCV 2017
- [2] Hoffman et al., "CyCADA: Cycle-Consistent Adversarial Domain Adaptation", ICML 2018
- [3] Yang et al., "FDA: Fourier Domain Adaptation for Semantic Segmentation", CVPR 2020
- [4] Mao et al., "Least Squares Generative Adversarial Networks" (LSGAN), ICCV 2017
- [5] Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks" (pix2pix, PatchGAN), CVPR 2017
- [6] Ulyanov et al., "Instance Normalization: The Missing Ingredient for Fast Stylization", 2016
