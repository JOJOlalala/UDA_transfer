# Architecture Overview — CycleGAN UDA Pipeline

## 1. Big Picture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        TASK I: Style Transfer                            │
│                                                                          │
│   Source Images (A)          Target Images (B)                           │
│   e.g. Amazon photos         e.g. Webcam photos                         │
│        SVHN digits                MNIST digits                           │
│        Photos                     Sketches                               │
│                                                                          │
│        ┌─────────┐    adversarial    ┌─────────┐                        │
│   A ──→│  G_AB   │──→ fake_B ──→    │  D_B    │──→ real/fake?           │
│        └─────────┘                   └─────────┘                        │
│        ┌─────────┐    adversarial    ┌─────────┐                        │
│   B ──→│  G_BA   │──→ fake_A ──→    │  D_A    │──→ real/fake?           │
│        └─────────┘                   └─────────┘                        │
│                                                                          │
│   Losses:                                                                │
│     L_GAN:   fool discriminators (adversarial)                           │
│     L_cyc:   A → G_AB → G_BA → A'  ≈ A   (cycle consistency)           │
│     L_idt:   G_BA(A) ≈ A            (identity regularization)           │
│                                                                          │
│   Variants:                                                              │
│     Pixel:    G sees full image                                          │
│     Spectral: G sees only low-freq(image), high-freq preserved exactly   │
└──────────────────────────────────────────────────────────────────────────┘

                              │
                    trained G_AB checkpoint
                              │
                              ▼

┌──────────────────────────────────────────────────────────────────────────┐
│                     TASK II: UDA Classification                          │
│                                                                          │
│   ┌─────────────────┐     ┌─────────────┐     ┌──────────────────┐      │
│   │ Source (labeled) │────→│ Adaptation  │────→│ Adapted Source   │      │
│   │ {(x_s, y_s)}    │     │ Method      │     │ {(x_adapted, y_s)}│     │
│   └─────────────────┘     └─────────────┘     └──────────────────┘      │
│                                  ↑                      │               │
│                           ┌──────┴──────┐               │               │
│                           │ Target      │          Train classifier      │
│                           │ (unlabeled) │               │               │
│                           │ {x_t}       │               ▼               │
│                           └─────────────┘     ┌──────────────────┐      │
│                                  │            │  Classifier f    │      │
│                                  │            │  (SmallCNN or    │      │
│                                  └───────────→│   ResNet-18)     │      │
│                                  test on      └──────────────────┘      │
│                                  real target          │                 │
│                                                       ▼                 │
│                                               target accuracy           │
└──────────────────────────────────────────────────────────────────────────┘
```

## 2. Methods and What Each Changes

```
Method    Adaptation Step                           Classifier Input
──────    ───────────────                           ────────────────
M0        None (source-only)                        Raw source images
M1        None (target oracle, upper bound)         Raw target images + labels
M2        G_AB(source) → fake target                Translated source images
M3        G_AB(low(src)) + high(src) → recombined   Spectral-recombined images
M4a       FDA amplitude swap at test time           M0 classifier, FDA-ed test imgs
M4b       FDA amplitude swap at train time          FDA-augmented source images
M5        CyCADA: G_AB + semantic loss, retranslate CyCADA-translated source images
```

### M0: Source-Only (lower bound)
```
Source (labeled) ──→ Train classifier ──→ Test on target
                     directly
```
No adaptation. Performance depends entirely on how similar source and target are.

### M1: Target Oracle (upper bound)
```
Target (labeled) ──→ Train classifier ──→ Test on target
                     (cheating — uses target labels)
```
Best possible. Not a real method — just shows the ceiling.

### M2: CycleGAN Pixel
```
Source ──→ G_AB ──→ "Fake target" ──→ Train classifier ──→ Test on real target
           │                           (source labels)
           └── trained in Task I (pixel mode)
```
Risk: G_AB may damage class-discriminative content (label flipping, artifacts).

### M3: CycleGAN Spectral
```
Source ──→ FFT ──→ low_freq ──→ G_AB ──→ translated_low ──┐
                   high_freq ─────────────────────────────→ + ──→ recombined
                   (preserved exactly)                          │
                                                                ▼
                                                Train classifier ──→ Test
```
Key difference: high-frequency detail (edges, texture, class features) is **hard-preserved**
from the source. Only low-frequency style (color, lighting) is translated.

### M4: FDA (Fourier Domain Adaptation)
```
M4a (at test):   test_img ──→ swap low-freq amplitude with source ──→ M0 classifier
M4b (at train):  source ──→ swap low-freq amplitude with target ──→ Train new classifier
```
No generator. No learning. Just spectral statistics matching.
Cannot flip labels or damage content — only shifts color/lighting statistics.

### M5: CyCADA-lite
```
                    ┌────── L_sem = CE(f(G_AB(x)), y) ──────┐
                    │       frozen classifier                │
                    ▼                                        │
Source ──→ G_AB ──→ "Fake target" ──→ f(·) ──→ prediction ──┘
           │         │
           │         └──→ Retranslate ──→ Retrain classifier ──→ Test
           └── fine-tuned 5 epochs with semantic loss
```
The semantic loss forces G_AB to preserve class predictions through translation.
Direct fix for CycleGAN label-flipping.

## 3. Architecture Details

### CycleGAN Generator (ResNetGenerator)
```
Input (3 × H × W)
  │
  ├─ 7×7 Conv, InstanceNorm, ReLU           →  ngf ch
  ├─ 3×3 Conv stride=2, IN, ReLU            →  ngf×2 ch       (downsample)
  ├─ 3×3 Conv stride=2, IN, ReLU            →  ngf×4 ch       (downsample)
  │
  ├─ ResNet Block × n_blocks                 →  ngf×4 ch       (transform)
  │   (ReflectPad → Conv → IN → ReLU → ReflectPad → Conv → IN + skip)
  │
  ├─ ConvTranspose stride=2, IN, ReLU       →  ngf×2 ch       (upsample)
  ├─ ConvTranspose stride=2, IN, ReLU       →  ngf ch         (upsample)
  └─ 7×7 Conv, Tanh                         →  3 ch           (output)
```

| Dataset type | ngf | n_blocks | n_downsample | Input size |
|---|---|---|---|---|
| Digits (32×32) | 32 | 6 | 1 | 32×32 |
| Office/PACS (current) | 64 | 9 | 2 | 64×64 |
| Office/PACS (standard) | 64 | 9 | 2 | **256×256** |

### Classifier

| Dataset | Architecture | Input | Pretrained | Params |
|---|---|---|---|---|
| mnist_usps, svhn_mnist | SmallCNN (3 conv + GN + FC) | 32×32 | No | ~200K |
| Office/PACS | ResNet-18 | 224×224 | ImageNet | ~11M |

### ResNet-18 feature map sizes by input resolution
```
Input   → stem(s2) → maxpool(s2) → layer1 → layer2(s2) → layer3(s2) → layer4(s2) → avgpool
  64         32          16          16          8           4            2×2=4       1×1
 128         64          32          32         16           8            4×4=16      1×1
 224        112          56          56         28          14            7×7=49      1×1
 256        128          64          64         32          16            8×8=64      1×1
```
At 64: only 4 spatial features → catastrophic information loss.
At 224+: rich spatial representation → standard operating range.

## 4. Current Hyperparameters

### CycleGAN Training
| Param | Value | Source |
|---|---|---|
| lr | 0.0002 | CycleGAN paper default |
| optimizer | Adam(β1=0.5, β2=0.999) | CycleGAN paper default |
| λ_cycle | 10.0 | CycleGAN paper default |
| λ_identity | 0.0 | Was 5.0 (paper), collapsed on close domains |
| epochs | 50 | Reduced from paper's 200 |
| decay_epoch | 25 | Linear LR decay in second half |
| batch_size | 8 (256px), 32 (64px), 256 (32px) | GPU memory limited |
| augmentation | Resize(1.12×) + RandomCrop + HFlip | CycleGAN paper |

### Classifier Training
| Param | SmallCNN (digits) | ResNet-18 (Office/PACS) |
|---|---|---|
| epochs | 20 | 30 |
| lr | 1e-3 | 1e-4 |
| optimizer | Adam | Adam |
| weight_decay | 5e-4 | 1e-4 |
| scheduler | CosineAnnealing | CosineAnnealing |
| batch_size | 256 | 64 |

## 5. Literature Baselines (what "good" looks like)

### Digit datasets (ResNet/LeNet at 32×32)
| Method | MNIST→USPS | SVHN→MNIST | Paper |
|---|---|---|---|
| Source-only | 78–82% | 60–67% | various |
| DANN | 91% | 73% | Ganin 2016 |
| CyCADA | 96% | 90% | Hoffman 2018 |
| SHOT | 98% | 98% | Liang 2020 |
| **Ours M0** | **90.1%** | **73.9%** | — |

Our M0 is already strong on digits (GroupNorm fix + good training recipe).

### Office-31 Amazon→Webcam (ResNet-50 at 224)
| Method | A→W | Paper |
|---|---|---|
| Source-only (ResNet-50) | 68–76% | various |
| DANN | 82% | Ganin 2016 |
| CyCADA | 90% | Hoffman 2018 |
| CDAN | 94% | Long 2018 |
| **Ours M0 (ResNet-18)** | **58.7%** | — |

Gap: our M0 is ~10–18 pp below literature source-only. Reasons:
- ResNet-18 vs ResNet-50 (less capacity)
- Our CycleGAN at 64×64 (literature uses 256×256)
- Smaller batch, fewer epochs

### Office-Home Art→Real (ResNet-50 at 224)
| Method | A→R | Paper |
|---|---|---|
| Source-only (ResNet-50) | 60–65% | various |
| DANN | 68% | |
| CDAN | 72% | Long 2018 |
| **Ours M0 (ResNet-18)** | **36.0%** | — |

Large gap — 65-class problem at low resolution with weaker backbone.

### PACS Photo→Sketch (ResNet-18/50 at 224)
| Method | P→S | Paper |
|---|---|---|
| Source-only | 35–47% | various |
| DANN | 50–55% | |
| **Ours M0** | **20.4%** | — |

## 6. Diagnosis: Why Our Results Lag Literature

| Factor | Impact | Fix |
|---|---|---|
| **CycleGAN resolution 64 vs 256** | Dominant for Office/PACS M2 | Retrain CycleGAN at 256 |
| **ResNet-18 vs ResNet-50** | ~5–10 pp on M0 | Could upgrade backbone |
| **50 CycleGAN epochs vs 200** | Unknown — may underfit | Try longer training |
| **No feature-level alignment** | Major — all lit methods add this | M5 CyCADA-lite partially addresses |
| **No pretrained classifier for M5** | CyCADA needs a good frozen classifier | Run M1 (oracle) to calibrate ceiling |
| **Small source sets** | Office-Home: 37 imgs/class | Heavier augmentation, regularization |
| **λ_identity=0** | Correct for close domains, may hurt distant ones | Per-dataset tuning |

### Key insight
Literature UDA methods (DANN, CyCADA, CDAN) combine **input-space AND feature-space** adaptation.
Our pipeline is **input-space only** (translate → retrain). This is inherently weaker — the
classifier never sees real target features during training. M5 (CyCADA-lite) adds a partial
feature signal via the semantic loss, but it's not the full joint-training approach.

For the report, this is the honest framing: "We benchmark input-space-only UDA methods
and analyze where they succeed (small domain gaps, digit tasks) and where they fall short
(large gaps, fine-grained classification) compared to joint input+feature methods."
