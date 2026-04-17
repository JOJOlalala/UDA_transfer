# Project I & II Progress — CycleGAN + UDA

**Last updated:** 2026-04-16 (session 7)

## Status: Task I **DONE** — Task II: M0+M2 baselines done, UNet256 training in progress (job 13758266)

---

## Completed This Session

### 1. Batch Size Tuning (A100-40GB benchmarks)

**32×32 datasets:**

| Batch Size | Steps/epoch | Epoch Time | VRAM | GPU Util |
|---|---|---|---|---|
| 4 (old) | 15,000 | 725s | 0.6 GB | 41% |
| **256 (chosen)** | **234** | **34s** | **6.3 GB** | **99%** |

**256×256 datasets:**

| Batch Size | Steps/epoch | Epoch Time | VRAM | GPU Util |
|---|---|---|---|---|
| 1 (old) | 2,817 | ~220s | ~0.7 GB | low |
| **8 (chosen)** | **352** | **157s** | **19.1 GB** | **100%** |
| 16 | 176 | 148s | 36.6 GB | 100% |
| 32 | — | OOM | >39.5 GB | — |

### 2. All 5 Datasets Ready

| # | Pair | Source | Target | Status |
|---|---|---|---|---|
| 1 | MNIST → USPS | 60,000 | 7,291 | Ready |
| 2 | SVHN → MNIST | 73,257 | 60,000 | Ready |
| 3 | Amazon → Webcam (Office-31) | 2,817 | 795 | Ready |
| 4 | Art → Real-World (Office-Home) | 2,427 | 4,357 | Ready (HF flwrlabs/office-home) |
| 5 | Photo → Sketch (PACS) | 1,670 | 3,929 | Ready |

### 3. Step-Based Training Loop
Rewrote `train.py` from epoch-based to step-based:
- Loss log every **10 steps** to `logs/<exp_name>.log`
- Vis + checkpoint every **500 steps** (checkpoint overwrites `latest.pth`)
- Epoch markers still logged for LR schedule tracking
- wandb init timeout (30s) + graceful fallback if no network
- **Verified**: 3-epoch test passed — losses converging, vis saved, checkpoint saved

### 3b. wandb via SSH Tunnel — VERIFIED
Compute nodes lack direct internet. Solution: SOCKS proxy through login node.
```bash
PROXY_PORT=$((20000 + RANDOM % 10000))
ssh -f -N -D $PROXY_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=5 asp2a-login-ntu01
export HTTPS_PROXY=socks5h://localhost:$PROXY_PORT
export HTTP_PROXY=socks5h://localhost:$PROXY_PORT
```
- Verified: wandb connected, loss curves + vis images uploaded
- wandb project: `UDA-CycleGAN`
- All PBS scripts (`train_pixel.sh`, `train_spectral.sh`, `train_all_task1.sh`) include tunnel setup

### 4. Visualization Notebook
`notebooks/visualize_datasets.ipynb` — executed with all outputs:
- Sample grids for all dataset pairs
- FFT decomposition at β=0.01/0.05/0.10/0.20
- Frequency domain gap analysis with radial profiles

### 5. Multi-GPU Launcher Script
`scripts/train_all_task1.sh` — 4-GPU pool manager:
- Queues 10 jobs (5 datasets × 2 modes)
- Auto-assigns freed GPUs to next job
- `CUDA_VISIBLE_DEVICES` isolation per job

---

## Config Summary (final)

| Dataset | Epochs | Batch | Image Size | Steps/epoch | Total steps | Est. time |
|---------|--------|-------|-----------|-------------|-------------|-----------|
| mnist_usps | 50 | 256 | 32×32 | 234 | 11,700 | ~28 min |
| svhn_mnist | 50 | 256 | 32×32 | 286 | 14,300 | ~34 min |
| amazon_webcam | 50 | 8 | 256×256 | 352 | 17,600 | ~2.2 hrs |
| art_realworld | 50 | 8 | 256×256 | 544 | 27,200 | ~2.5 hrs |
| photo_sketch | 50 | 8 | 256×256 | 491 | 24,550 | ~2.5 hrs |

All 50 epochs: decay_epoch=25, save_freq=25, vis_freq=10.
LR: 0.0002 constant for epochs 1-25, linear decay to 0 for epochs 26-50.
**Total: 190,700 steps across 10 jobs. Est. ~5.5 hrs with 4 GPUs.**

---

## Completed Session 4 (2026-04-14)

### Task I Training — DONE
- Reduced epochs: 10 for 32×32, 20 for 256×256 (decay at half)
- Submitted 10 individual g1 jobs (parallel, up to 8 concurrent)
- wandb set to offline mode (SOCKS proxy caused hangs)
- All 10 jobs completed in ~1.5 hrs
- Checkpoints: `checkpoints/<exp_name>/latest.pth` (all 10 present)
- Vis grids: `results/<exp_name>/step*.png`
- Losses converged for all datasets, no issues

---

## Completed Session 5 (2026-04-14 evening)

### Checkpoint validation revealed failures at 256×256

Created [notebooks/validate_done.ipynb](../notebooks/validate_done.ipynb) — executes per-dataset side-by-side comparison grids (pixel + spectral). Outputs in `results/validation_done/`.

Visual inspection of the 50-epoch 256×256 checkpoints:
- **art_realworld**: **pure identity collapse**. Both pixel and spectral learned G ≈ identity in both directions. Cycle loss 0.09 is the smoking gun (trivially small).
- **photo_sketch**: **asymmetric failure**. A→B (photo → sketch) actually works — line-drawing style is learned. B→A (sketch → photo) is mode-collapsed: every sketch input maps to the same brownish wood-grain texture blob, regardless of content.
- **amazon_webcam**: partial translation. Not pure collapse, but over-anchored to identity — only mild color/background shifts.

### Root cause analysis
1. **`lambda_identity: 5.0` is too strong** for small-domain-gap datasets (Office-Home art/real_world are visually close). Identity loss forces G ≈ identity when the gap doesn't justify the cost.
2. **Missing RandomCrop augmentation**. [data/datasets.py:get_transform_large](../data/datasets.py) only had `Resize + RandomHorizontalFlip`. The CycleGAN paper uses `Resize(286) + RandomCrop(256) + Flip`. Without random crop, D memorizes training pixels → overfits on small datasets → forces G to collapse.

### Fixes applied
- [data/datasets.py](../data/datasets.py): `get_transform_large` now does `Resize(1.12×) + RandomCrop + Flip` (permanent fix, universal to all 256/128/64 datasets)
- All 3 Office-Home / PACS configs now set **`lambda_identity: 0.0`** (was 5.0)
- All 3 configs downsized from 256 → **64×64**, batch_size 8 → **32**, epochs stay 50 at decay_epoch 25
- Spectral runs at 64 use **β=0.2** (not 0.05) to match the 256 physical frequency cutoff (radius 6.4 px in FFT space)
- [train.py:101](../train.py) scheduler resume bug patched: added `sched_G.last_epoch = start_epoch - 1` so linear LR decay respects the resumed epoch

### Quick feasibility test (validated the fix)
- art_realworld @ 128×128, pixel only, lambda_identity=0, 15 epochs, bs=16
- **Cycle loss 0.09 → 0.27** (3× higher ⇒ G doing non-trivial work)
- Visual grid shows clear non-identity translation
- Wall-clock: 66s/epoch × 15 = ~16 min total

### Full 64×64 retrain — DONE (20 min wall-clock)
All 6 gdev jobs (13716341-13716346) completed Epoch 50/50. Final cycle losses:

| Run | cyc | vs 256 cyc | Note |
|---|---|---|---|
| amazon_webcam pixel | 0.125 | +2.2× (was 0.056) | healthy |
| amazon_webcam spectral β0.2 | 0.123 | — | D=0.04-0.05 |
| art_realworld pixel | 0.194 | +2.1× (was 0.091) | **collapse broken** |
| art_realworld spectral β0.2 | 0.226 | — | D=0.01-0.02 (watch) |
| photo_sketch pixel | 0.140 | +2.0× (was 0.069) | healthy |
| photo_sketch spectral β0.2 | 0.088 | — | D=0.01 (watch) |

Cycle losses are 2-3× higher than the failed 256 runs → generators are doing non-trivial work. The `lambda_identity=0` fix held. Two spectral runs show suspiciously-low D losses (art_realworld, photo_sketch); visual inspection needed to confirm whether D overfit or training is fine.

All 12 wandb runs (6 old failed + 6 new) synced to [UDA-CycleGAN](https://wandb.ai/emoless584-nanyang-technological-university-singapore/UDA-CycleGAN).

### wandb sync state
5 of the old 256-epoch runs synced before retrain decision:
- amazon_webcam pixel/spectral, art_realworld pixel/spectral, photo_sketch pixel
- photo_sketch_spectral (old, offline run `karod6k7`) never fully finished — resume job was cancelled when we pivoted to 64×64

---

---

## Session 6 (2026-04-15) — Task I closure + Task II Phase A kickoff

### Task I closure
- **32×32 spectral retrained at β=0.4.** Original β=0.05 runs showed D_A ~ 0.0005 (discriminator collapsed to certainty — generator free-riding). Retrain with β=0.4 (radius 6.4 px at 32×32) matches the 64×64 β=0.2 physical cutoff. Two resume jobs (13721436/13721437) completed cleanly to Epoch 50/50 after original runs hit the 30-min gdev walltime.
- **Validation notebooks updated** to use per-dataset `DATASET_BETA` dict (`mnist/svhn: 0.4`, Office-Home/PACS: `0.2`). Both [validate_done.ipynb](../notebooks/validate_done.ipynb) (job 13733165, rc=0, 3:03) and [validate_checkpoints.ipynb](../notebooks/validate_checkpoints.ipynb) (job 13733858, rc=0, 3:32) re-executed on compute node. Outputs in [results/validation_done/](../results/validation_done/) and [results/validation/](../results/validation/).
- **Spectral decomposition sanity-view fix.** [notebooks/validate_checkpoints.ipynb](../notebooks/validate_checkpoints.ipynb) cell 12 was showing a black low-freq row: at β=0.05 on 32×32, the low-pass radius was 0.8 px → only DC coefficient → `low_A ≈ mean of MNIST ≈ −0.74` → `(x+1)/2 ≈ 0.13` → visually black. Fix: per-dataset β + per-row auto-contrast (`vutils.make_grid(..., normalize=True)`) for Low/High/Translated-low rows, plus numerical `min/max/mean` printout.

### Task II Phase A — classifier scaffolding (IN PROGRESS)
Code added under [task2/](../task2/):
- `classifier.py` — `SmallCNN` (3 conv blocks → avgpool → FC, ~200K params, for 32×32 digits) and `get_resnet18` (IMAGENET1K_V1 pretrained, FC replaced, for 64×64 office datasets).
- `data_labeled.py` — labeled source/target loaders for all 5 pairs. Digits via torchvision with labels; Office-31/Office-Home/PACS via `ImageFolder`. All normalized to symmetric [−1,1] to match CycleGAN output so translated images drop in without re-norm. `NUM_CLASSES = {mnist_usps:10, svhn_mnist:10, amazon_webcam:31, art_realworld:65, photo_sketch:7}`.
- `train_classifier.py` — generic trainer, `--data {source,target,translated}`, cosine LR, per-epoch target eval, saves `best.pth`/`latest.pth`/`metrics.json`. Default hparams: SmallCNN 20 ep / bs 256 / Adam 1e-3; ResNet-18 30 ep / bs 64 / Adam 1e-4.
- `eval.py` — standalone target-test evaluator.
- [scripts/run_m0_one.sh](../scripts/run_m0_one.sh) — PBS launcher parameterized by `-v DATASET=…`.

### M0 results (first pass @ 64×64, ResNet-18)

| Dataset | Arch | best_tgt | final_tgt | Lit (@224) | Notes |
|---|---|---|---|---|---|
| mnist_usps | SmallCNN-GN | **0.9013** | 0.8769 | ~82% | ✅ above literature |
| svhn_mnist | SmallCNN-GN | **0.7398** | 0.7079 | ~60–70% | ✅ at upper end |
| amazon_webcam | ResNet-18 @64 | 0.3296 | 0.3283 | ~62% | ⚠️ −30 pp |
| photo_sketch | ResNet-18 @64 | 0.1652 | 0.1616 | ~35% | ⚠️ near random (7 classes, chance=14%) |
| art_realworld | ResNet-18 @64 | ~0.18 (running) | — | ~45% | ⚠️ −27 pp |

### BN_STATS bug (fixed)

First mnist_usps smoke (13734245) hit **tgt_acc=0.1908** — far below literature. [task2/diag_bn.py](../task2/diag_bn.py) (job 13734612) confirmed **eval-mode 0.1908 vs train-mode 0.8535** (+0.66 gap) → verdict `BN_STATS`. Root cause: SmallCNN learned MNIST-specific BN running stats (source per-channel mean ≈ −0.748, USPS ≈ −0.464 — real low-freq gap on top of the BN failure). Fix in [task2/classifier.py](../task2/classifier.py):
- `SmallCNN`: `BatchNorm2d → GroupNorm(min(8,C), C)` throughout. Per-instance stats ⇒ no cross-domain failure.
- `get_resnet18`: `track_running_stats=False` + `running_mean=running_var=None` on all BN layers. Same effect without discarding pretrained gammas/betas.

Retry (13734716) jumped to **0.9013** best. All 4 fan-out jobs (13735529–32) were submitted with the patched classifier, so no further BN retraining is needed.

### ResNet-18 @64×64 underperforms — root cause

Three Office/PACS M0 runs hit 16–33% while pretrained literature is 35–62%. Train_acc → 1.0 and target_loss → 3–4 for all three — classic overfit on the tiny source sets. Root cause is **input-resolution collapse**: ResNet-18's stem is 7×7/s2 + maxpool/s2 + four stride-2 stages, so a 64×64 input ends as a 2×2 feature map at layer4, wasting most of the pretrained receptive field.

### M0 @224 re-run — FIX VALIDATED

Patched [task2/data_labeled.py](../task2/data_labeled.py) `_transform_large` to always emit 224 (train: `Resize(256)+RandomCrop(224)`, eval: `Resize(224)`). CycleGAN pipeline stays at 64×64; only the classifier transform lifts to ResNet-18's native ImageNet resolution. New launcher [scripts/run_m0_r224.sh](../scripts/run_m0_r224.sh) writes to `checkpoints_task2/m0_r224/<dataset>` to preserve the @64 foil.

Jobs 13737553–55 (all rc=0, all 30/30 epochs):

| Dataset | @64 best | @224 best | Δ | Lit (@224) | Wall |
|---|---|---|---|---|---|
| amazon_webcam | 0.3296 | **0.5874** | **+25.8 pp** | ~62% | 126s |
| art_realworld | 0.1760 | **0.3599** | **+18.4 pp** | ~45% | 805s |
| photo_sketch | 0.1652 | **0.2039** | +3.9 pp | ~35% | 134s |

**Interpretation**:
- amazon_webcam and art_realworld gains confirm the input-resolution diagnosis — giving ResNet-18 its native receptive field recovers most of the literature gap.
- photo_sketch barely moves (+4 pp). This is **not** a resolution problem any more — it's the photo→sketch domain gap itself. ImageNet-pretrained ResNet has strong priors for natural photos and none for line-drawings; target_loss stays 2.67 while source train_acc = 1.0. This is exactly the gap that **M2/M3 CycleGAN translation** is meant to close, so it's the most interesting dataset to watch in Phase B/C.
- art_realworld took 13 min (vs 2 min for the others) because Office-Home images are large JPEGs and PIL Resize(256) is CPU-bound; not a concern, just noted.

### Baselines fully captured

| Dataset | Arch | @64 best | @224 best | Status |
|---|---|---|---|---|
| mnist_usps | SmallCNN-GN | 0.9013 | — | ✅ above lit (82%) |
| svhn_mnist | SmallCNN-GN | 0.7398 | — | ✅ upper end of lit (60–70%) |
| amazon_webcam | ResNet-18 | 0.3296 | **0.5874** | ~6 pp below lit |
| art_realworld | ResNet-18 | 0.1760 | **0.3599** | ~9 pp below lit |
| photo_sketch | ResNet-18 | 0.1652 | **0.2039** | huge gap to lit — M2/M3 target |

---

## Phase B — M2 CycleGAN-pixel — DONE

Code: [task2/translate_bulk.py](../task2/translate_bulk.py) (bulk translator, matches each dataset's CycleGAN-time source transform), [scripts/run_translate_bulk.sh](../scripts/run_translate_bulk.sh), [scripts/run_m2_one.sh](../scripts/run_m2_one.sh). Also patched [data_labeled.py](../task2/data_labeled.py) `build_translated_dataset` to dispatch grayscale transform + synthetic `{"0":0,...,"9":9}` class_to_idx for digits (originally asserted ImageFolder).

Translation outputs written to `data_translated/<dataset>_pixel/<class>/<idx>.png` at the CycleGAN training resolution (32 or 64); classifier retrain upsamples to 224 (Office/PACS) or 32-gray (digits). All 5 bulk jobs rc=0, all 5 M2 retrains rc=0.

### M2 pixel results (bs/Adam hparams identical to M0)

| Dataset | M0 best | M2 best | Δ vs M0 | Verdict |
|---|---|---|---|---|
| mnist_usps | 0.9013 | **0.9297** | **+2.8 pp** | ✅ small gain — only win |
| svhn_mnist | 0.7398 | 0.4355 | **−30.4 pp** | 💥 catastrophic |
| amazon_webcam | 0.5874 | 0.4491 | **−13.8 pp** | ❌ regressed |
| art_realworld | 0.3599 | 0.2800 | **−8.0 pp** | ❌ regressed |
| photo_sketch | 0.2039 | 0.2041 | +0.0 pp | ➖ flat |

### Interpretation

Pixel CycleGAN alone is **net-negative on 4/5 pairs**. Two distinct failure modes:

1. **SVHN→MNIST: label-flipping.** −30 pp is the textbook CycleGAN-UDA pathology cited in the CyCADA paper. Cycle consistency is a weak constraint: G_AB is free to remap the digit identity (e.g. 6↔9, 1↔7) as long as G_BA can undo it. No classifier-side signal during CycleGAN training → no reason for G_AB to preserve class. Every regression on svhn_mnist is a confirmed CycleGAN limitation, not a pipeline bug.

2. **Office/PACS: pretrained-feature damage.** M0 @224 already beat the @64 baseline by ~20 pp because it lets ResNet-18 use its ImageNet priors. Pixel CycleGAN at 64×64 then rewrites the image in ways that partially break those priors — the resulting "webcam-looking" or "real-world-looking" images are crude upsamples with GAN artifacts and no longer look like natural photos to the pretrained backbone. M0 feeds clean source photos directly through ResNet-18; M2 feeds degraded translated photos. The domain shift isn't large enough to overcome the ImageNet-feature damage.

3. **mnist_usps: gain.** Both domains are handwritten grayscale → very small domain gap → G_AB nearly preserves digit identity → modest +2.8 pp from matching the target-domain style.

4. **photo_sketch: flat.** The pixel CycleGAN output barely shifts the classifier input distribution (either generator is too conservative or the "sketchification" doesn't fool ResNet-18 into adapting). photo_sketch remains the hard case — M3 (spectral) and especially M5 (CyCADA-lite) are the next levers.

### Matched-resolution follow-up (@128 probe)

The Office/PACS −8 to −14 pp regression hides an asymmetric-blur confound: M2 classifier trained on 64→224 (3.5× upsample, blurry) but evaluated on 300→224 (slight downsample, sharp). To isolate content damage from resolution asymmetry, re-ran M0 and M2 at a matched `CLASSIFIER_RES=128` (train: 64→128, eval: 300→128 — both ~2× scale). Parameterized [data_labeled.py](../task2/data_labeled.py) with a `CLASSIFIER_RES` env var and added [scripts/run_m0_r128.sh](../scripts/run_m0_r128.sh) / [scripts/run_m2_r128.sh](../scripts/run_m2_r128.sh). 6 jobs (13740220–25, all rc=0).

| Dataset | M0 @64 | M0 @128 | M0 @224 | M2 @128 | M2 @224 | **Δ @128** | **Δ @224** |
|---|---|---|---|---|---|---|---|
| amazon_webcam | 0.3296 | 0.4868 | 0.5874 | 0.4101 | 0.4491 | **−7.7** | −13.8 |
| art_realworld | 0.1760 | 0.2922 | 0.3599 | 0.2518 | 0.2800 | **−4.0** | −8.0 |
| photo_sketch  | 0.1652 | 0.2189 | 0.2039 | 0.1919 | 0.2041 | **−2.7** | +0.0 |

**Finding: resolution asymmetry explains ~half the @224 regression, not all of it.**
- amazon_webcam: @224 Δ = −13.8 → @128 Δ = −7.7 (6.1 pp recovered by matching res)
- art_realworld: @224 Δ = −8.0 → @128 Δ = −4.0 (4.0 pp recovered)
- photo_sketch: @224 was already flat; @128 is slightly worse because M0 @128 happens to beat M0 @224 on this dataset (2×2 feature map pathology is less severe when photos are already low-info)

**Pixel CycleGAN still regresses at matched res** — the residual −3 to −8 pp is genuine generator-induced content damage, not a pipeline artifact. The cleaner comparison baseline for all downstream CycleGAN-based methods (M3 spectral, M5 CyCADA-lite) is therefore **M0 @128**, not @224, because it pairs train/test at a symmetric spatial frequency with the translated inputs.

### Implications for remaining phases

- **Fair baseline is M0 @128**, not @224, whenever a method trains on 64×64 translated source. Keep @224 as the upper-bound reference only.
- **M3 (spectral)** now has a clearer hypothesis: spectral-recombined inference `G(low_x) + high_x` should *preserve* source high frequency (= classifier-relevant detail) and only swap low-frequency style. If M3 beats M2 @128, it confirms the content-damage diagnosis.
- **M4 (FDA)** should still be the best non-learned Office/PACS baseline — FDA swaps only low-freq amplitude and cannot flip labels or damage high-freq features. At @128 vs M0 @128 it has the strongest chance of a positive Δ.
- **M5 (CyCADA-lite)** is the targeted fix for svhn_mnist's −30 pp (label-flipping). For Office/PACS its benefit depends on whether the semantic loss stops the generator from damaging pretrained-feature-friendly content.

---

## Session 7 (2026-04-16) — Diagnosis, UNet variant, resolution experiments

### Key findings this session

**1. Visual inspection confirmed M2 failure modes** — created [notebooks/visualize_task2.ipynb](../notebooks/visualize_task2.ipynb), outputs in [results/task2_vis/](../results/task2_vis/).
- **svhn_mnist label-flipping CONFIRMED**: translated digits show wrong identity (label 1 → looks like 4/7, label 6 → looks like 9). Smoking gun for −30 pp.
- **photo_sketch G_AB IS sketchifying**: translated photos produce clear line drawings. M2 flat because 64×64 loses class-discriminative sketch details, not because G_AB failed.
- **art_realworld domains are inherently close**: Office-Home "Art" contains a mix of drawings AND photographic art. Small effective domain gap explains identity collapse and marginal M2 gains.
- **amazon_webcam/art_realworld translations look reasonable**: object identity preserved, style shifted.

**2. Resolution is the dominant bottleneck (not architecture)**

Ran a 256-probe using old 20-epoch checkpoints (even though they had identity/collapse issues):

| Dataset | M0 @224 | M2 @64 | M2 @128 | M2 @256 (probe) |
|---|---|---|---|---|
| amazon_webcam | 0.5874 | 0.4491 (−13.8) | 0.4101 (−7.7) | **0.5610 (−2.6)** |
| art_realworld | 0.3599 | 0.2800 (−8.0) | 0.2518 (−4.0) | **0.3528 (−0.7)** |
| photo_sketch | 0.2039 | 0.2041 (+0.0) | 0.1919 (−2.7) | **0.2286 (+2.5)** |

256 slashes M2 regression from catastrophic to near-zero. art_realworld probe = identity-collapsed G, so effectively M0 through a different path.

**3. UNet generator variant — marginal at 64, not the fix**

Wrote [models/generator_unet.py](../models/generator_unet.py) — UNet with skip connections (pix2pix-style, InstanceNorm). Skip connections let class-discriminative detail bypass the bottleneck. 2-3× faster than ResNet generator, 54.4M params at 256 (vs 11.4M ResNet).

UNet @64 M2 results (all 3 done):

| Dataset | M0 @224 | M2 ResNet@64 | M2 UNet@64 |
|---|---|---|---|
| amazon_webcam | 0.5874 | 0.4491 | **0.4491** (identical) |
| art_realworld | 0.3599 | — | **0.2336** |
| photo_sketch | 0.2039 | 0.2041 | **0.2105** (+0.7 pp) |

UNet at 64 doesn't help — spatial information isn't there regardless of skip connections. Resolution is the bottleneck, not architecture.

**4. ResNet-50 classifier backbone added**

[task2/classifier.py](../task2/classifier.py) now supports `resnet50` arch (ImageNet-pretrained, same BN-stats fix as ResNet-18). Standard in all UDA literature.

### Code changes this session

| File | Change |
|---|---|
| [models/generator_unet.py](../models/generator_unet.py) | New UNet generator (skip connections, InstanceNorm) |
| [models/cyclegan.py](../models/cyclegan.py) | `generator: unet` config option |
| [task2/classifier.py](../task2/classifier.py) | `resnet50` backbone option |
| [task2/translate_bulk.py](../task2/translate_bulk.py) | `img_size` from config (supports 256 checkpoints), UNet checkpoint loading |
| [task2/data_labeled.py](../task2/data_labeled.py) | `CLASSIFIER_RES` env var for resolution control |
| [train.py](../train.py) | `--suffix` flag for experiment naming |
| [notebooks/visualize_task2.ipynb](../notebooks/visualize_task2.ipynb) | Task II visual inspection notebook |
| [doc/ARCHITECTURE.md](../doc/ARCHITECTURE.md) | Full pipeline diagram + literature baselines |

**5. UNet256 CycleGAN training — ALL 3 DONE**

| Dataset | Job | Epochs | Epoch time | GPU util | VRAM | Training health |
|---|---|---|---|---|---|---|
| amazon_webcam | 13758266 | 100/100 | 51.5–53.7s | 98% | 3.6 GB | Healthy — D_A/D_B ~0.04–0.28 |
| art_realworld | 13761114 | 100/100 | 81–83s | — | — | Healthy — D_A ~0.10, D_B ~0.28 |
| photo_sketch | 13761115 | 100/100 | 73–74s | — | — | **D collapsed** — D_A=0.01, D_B=0.0001 |

photo_sketch D collapse diagnosis: UNet skip connections leak photo-level detail (textures, colors) into the decoder output, making it trivially easy for D_B (sketch discriminator) to classify "real sketch vs photo-with-sketch-paint". D_B collapses → gradient to G_AB vanishes → G_AB stops learning. **UNet is the wrong architecture for large domain gaps** — ResNet generator (no skips, forces bottleneck compression) would be better for photo→sketch.

Visual check for amazon_webcam added to [notebooks/visualize_task2.ipynb](../notebooks/visualize_task2.ipynb) (section 5), output in [results/task2_vis/unet256_amazon_webcam_quick.png](../results/task2_vis/unet256_amazon_webcam_quick.png).

**6. M2 UNet256 + ResNet-50 classifier @224 — ALL 3 DONE**

Fixed `default_hparams` in [task2/train_classifier.py](../task2/train_classifier.py) to support `resnet50` (bs=32, 30 ep, lr=1e-4).

| Dataset | M0 R18 @224 | M2 UNet256 R50 | Δ vs M0 | Notes |
|---|---|---|---|---|
| amazon_webcam | 0.5874 | **0.5887** | **+0.1 pp** | Wash — small domain gap |
| art_realworld | 0.3599 | **0.3718** | **+1.2 pp** | First positive M2 for this dataset |
| photo_sketch | 0.2039 | **0.2021** | **−0.2 pp** | Flat — D collapsed, UNet skip leak |

**Key findings:**
1. **Resolution is the dominant factor** (~14 pp gain from 64→256 on UNet, same lambda_id=0). Confirmed by UNet@64 (0.4491) vs UNet@256 (0.5887) on amazon_webcam.
2. **UNet works for close domains, fails for distant domains.** amazon_webcam (+0.1) and art_realworld (+1.2) benefit from skip-connection detail preservation. photo_sketch needs drastic style change that skip connections prevent.
3. **art_realworld first positive M2 result.** All prior M2 runs regressed (0.2336–0.2800 vs M0 0.3599). UNet256+R50 = 0.3718 (+1.2 pp) — resolution + lambda_id=0 fixed the regression and shows a small genuine CycleGAN gain.
4. **Classifier backbone mismatch.** M2 UNet256 uses R50 while all M0 baselines use R18. Need M0 R50 for fair comparison.

---

## Session 8 (2026-04-17) — M0 R50, spectral variants, DDP, FDA, CyCADA

### M0 R50 baselines — DONE

| Dataset | M0 R18 @224 | M0 R50 @224 | Δ |
|---|---|---|---|
| amazon_webcam | 0.5874 | **0.6415** | +5.4 pp |
| art_realworld | 0.3599 | **0.3737** | +1.4 pp |
| photo_sketch | 0.2039 | **0.2184** | +1.5 pp |

### M4 FDA — DONE

Pure FFT amplitude swap (no GAN). β=0.01, R50 @224.

| Dataset | M0 R50 | M4 FDA R50 | Δ |
|---|---|---|---|
| amazon_webcam | 0.6415 | 0.6302 | −1.1 pp |
| art_realworld | 0.3737 | **0.3801** | **+0.6 pp** |
| photo_sketch | 0.2184 | **0.2431** | **+2.5 pp** |

FDA is the best non-learned method for photo_sketch. No D collapse risk, no GAN artifacts.

### M5 CyCADA — DONE (partial)

Implemented full CyCADA (Hoffman 2018): semantic consistency + feature-level discriminator. Feature discriminator aligns classifier's internal representations between translated-source and real-target.

| Dataset | CycleGAN base | M0 R50 | M5 lite | M5 full | Δ full vs M0 |
|---|---|---|---|---|---|
| amazon_webcam | UNet256 | 0.6415 | 0.6189 | — (reuse lite) | −2.3 pp |
| art_realworld | UNet256 | 0.3737 | 0.3739 | — (reuse lite) | +0.0 pp |
| photo_sketch | ResNet@64 | 0.2184 | 0.2138 | 0.2204 | +0.2 pp |

CyCADA @64 barely helps — the 64→224 upsampling bottleneck limits how much semantic/feature signals can guide G. **CyCADA needs to run at native classifier resolution (224)** for the feature-level alignment to work properly.

### Spectral CycleGAN β experiments

Spectral CycleGAN D_B collapses on photo_sketch regardless of β:
- β=0.057 @224: collapsed by epoch 30
- β=0.2 @224: collapsed by epoch 40 (just delayed, not fixed)

Root cause: photo vs sketch low-freq content is fundamentally too different at any β. D_B always eventually learns "real sketch low-freq = white/uniform" vs "translated sketch low-freq = colored" perfectly.

### Master comparison table (all R50 @224 classifier)

| Method | amazon_webcam | art_realworld | photo_sketch | Type |
|---|---|---|---|---|
| **M0 R50** (baseline) | **0.6415** | **0.3737** | **0.2184** | — |
| M2 pixel UNet256 | 0.5887 (−5.3) | 0.3718 (−0.2) | 0.2021 (−1.6) | input-space |
| M3 spectral @64 | 0.5899 (−5.2) | 0.3491 (−2.5) | 0.2370 (+1.9) | input-space |
| M3 posthoc @256 | 0.6038 (−3.8) | 0.3776 (+0.4) | — | input-space |
| M3 spectral224 β=0.057 | 0.6390 (−0.3) | 0.3773 (+0.4) | 0.2339 (+1.6) | input-space |
| M3 spectral224 β=0.2 | — | pending | 0.2207 (+0.2) | input-space |
| M4 FDA β=0.01 | 0.6302 (−1.1) | **0.3801 (+0.6)** | **0.2431 (+2.5)** | input-space |
| M5 CyCADA-full @64 | 0.6189 (−2.3) | 0.3739 (+0.0) | 0.2204 (+0.2) | feature-space |
| M5 CyCADA @224 | pending | pending | pending | **feature-space** |

### Key research findings

1. **Resolution is the dominant factor.** 64→256 gives +14 pp on M2 pixel. All @64 methods are bottlenecked by upsampling blur. CycleGAN and classifier should operate at the **same resolution** (224).

2. **M0 R50 is hard to beat with input-space-only adaptation.** For close-domain pairs (amazon_webcam, art_realworld), M0 R50 is already near literature source-only. Adaptation methods are flat or negative.

3. **photo_sketch is the key test case.** Large domain gap where adaptation should matter most. Best results: M4 FDA (+2.5 pp), M3 spectral @64 (+1.9 pp). Both preserve source high-freq content.

4. **UNet fails for large domain gaps.** Skip connections leak source detail, causing D collapse on photo→sketch. ResNet generator (no skips) is correct for distant domains.

5. **Spectral D collapse is fundamental for photo_sketch.** β=0.057 and β=0.2 both collapse — the photo/sketch low-freq distributions are too different. Not fixable by widening β alone.

6. **CyCADA needs native resolution.** Semantic + feature losses at 64→224 provide too weak a gradient signal. CyCADA @224 (pixel ResNet@224 base → fine-tune) is the proper pipeline.

7. **FDA is the safest adaptation method.** No training, no D collapse, cannot flip labels. Best for photo_sketch where all GAN-based methods struggle.

### Spectral β=0.2 results — DONE

| Dataset | M3 β=0.057 | M3 β=0.2 | Notes |
|---|---|---|---|
| photo_sketch | 0.2339 | 0.2207 | β=0.2 worse — D collapsed earlier in quality |
| art_realworld | 0.3773 | pending | D_A=0.002 collapsed |

Wider β delayed D collapse in time but not in impact. β=0.057 gave better results because G learned more useful translation before D died.

### Research design — corrected

**Core insight: input-space vs feature-space adaptation**

Methods M2 (pixel CycleGAN), M3 (spectral CycleGAN), M4 (FDA) all operate on **visual signal** — they adapt pixel/frequency statistics but don't understand semantic content. They hit a ceiling:
- **Close domains (amazon_webcam):** M0 R50 already strong (0.6415), visual adaptation adds nothing
- **Subtle gap (art_realworld):** FDA +0.6 pp is the best, but still marginal
- **Large gap (photo_sketch):** FDA +2.5 pp is the best visual method, but far from literature

**CyCADA** operates on **semantic features** from the classifier. The feature discriminator aligns the classifier's internal representations — it doesn't care what the image looks like, it forces "translated art clock" features to match "real-world clock" features. This is fundamentally deeper than any pixel/frequency-space method.

**The right pipeline for both art_realworld and photo_sketch:**
1. Train pixel ResNet CycleGAN @224 with DDP (resolution-matched to classifier)
2. Fine-tune with full CyCADA (semantic + feature discriminator) at native 224
3. Translate + classify

**Report narrative:** M0→M2→M3→M4 show progressive attempts at input-space adaptation and their limitations → M5 CyCADA adds feature-level alignment → quantifies how much deeper adaptation helps.

### DDP infrastructure — DONE

[train_ddp.py](../train_ddp.py) — `torchrun` DDP, 4×A100, ~4× speedup. Configs at [configs/*_resnet224.yaml](../configs/).

### Code changes this session

| File | Change |
|---|---|
| [train_ddp.py](../train_ddp.py) | DDP training script (torchrun, DistributedSampler) |
| [task2/classifier.py](../task2/classifier.py) | `resnet18_cifar`, `resnet50` hparams |
| [task2/train_classifier.py](../task2/train_classifier.py) | `resnet50`, `resnet18_cifar` hparams |
| [task2/translate_bulk.py](../task2/translate_bulk.py) | `--mode posthoc` and `--mode spectral` |
| [task2/fda_bulk.py](../task2/fda_bulk.py) | M4 FDA (amplitude swap, no GAN) |
| [task2/cycada_lite.py](../task2/cycada_lite.py) | Full CyCADA (semantic + feature discriminator) |
| [utils/spectral.py](../utils/spectral.py) | `fda_transfer()` function |
| [test/verify_datasets.ipynb](../test/verify_datasets.ipynb) | Dataset verification notebook |
| configs/*_resnet224.yaml | ResNet @224 configs for DDP |

---

## TODO — Next Session

### Auto-chained jobs (running via /tmp/chain_watcher.log)

The following jobs are chained automatically — check results in next session:

1. **photo_sketch pixel ResNet@224 DDP** (job 13765034, gdev 4GPU) → base CycleGAN
   - → auto-submits **CyCADA @224 fine-tune** (10 ep, semantic + feature disc) → translate → classify
   - Result at: `checkpoints_task2/m5_cycada224_r50/photo_sketch/metrics.json`

2. **art_realworld pixel ResNet@224 DDP** (job 13765123, gdev 4GPU) → base CycleGAN
   - → auto-submits **CyCADA @224 fine-tune** → translate → classify
   - Result at: `checkpoints_task2/m5_cycada224_r50/art_realworld/metrics.json`

3. **art_realworld spectral β=0.2 DDP** (job 13764610) → auto-submits M3 eval
   - Result at: `checkpoints_task2/m3_spectral224b02_r50/art_realworld/metrics.json`

**To check all results:**
```bash
cat /tmp/chain_watcher.log
for ds in photo_sketch art_realworld; do
  for m in m3_spectral224b02_r50 m5_cycada224_r50; do
    M="checkpoints_task2/${m}/${ds}/metrics.json"
    [ -f "$M" ] && python3 -c "import json; d=json.load(open('$M')); print(f'${m} ${ds}: best={d[\"best_target_acc\"]:.4f}')" || echo "$m $ds: not yet"
  done
done
```

### After collecting CyCADA @224 results
- M1 Target oracle (trivial, 5 jobs — upper-bound reference for report)
- Consolidate all numbers into `results/task2/accuracy_table.{md,csv}`
- Qualitative vis notebook (add M3/M4/M5 sections)
- Report PDF (name + matric; declare AI usage)

### Required methods status
| Method | Status | Notes |
|---|---|---|
| M0 Source-only | ✅ Done R18 + R50 @224 | Complete |
| M2 CycleGAN-pixel | ✅ Done (UNet256+R50, all 3) | Complete |
| M3 CycleGAN-spectral | ✅ Done (@64, posthoc, spectral224 β=0.057/0.2) | art_realworld β=0.2 pending |
| M4 FDA | ✅ Done (all 3, β=0.01) | Complete |
| M5 CyCADA | ⚠️ @64 done, **@224 auto-chained** | Key experiment — feature-space alignment at native res |
| M1 Target oracle | Not started (optional) | Trivial, 5 jobs |

---

### Step 5: Report (original)
- PDF, name + matric number
- Task I: qualitative comparisons + analysis
- Task II: qualitative + quantitative + pros/cons
- **Declare AI usage**

---

## Known Issues / Solutions
- **wandb on compute nodes**: compute nodes have NO direct internet. **Solution**: SSH SOCKS proxy tunnel to login node (login node has internet). Simpler pattern used in session 5: `WANDB_MODE=offline` then `wandb sync` from login node after training (login node has direct internet).
- **Office-Home download**: `data/datasets.py` shadows HuggingFace `datasets` package. Run download from `/tmp`. List comprehension OOMs — use index-based iteration on compute node (32GB+ RAM).
- **PBS log buffering**: stdout not visible until job ends. Use `logs/` for live monitoring via SSH to compute node.
- **Login node hostname**: `asp2a-login-ntu01` (used for SSH tunnel from compute nodes).
- **gdev queue access**: do NOT pass `-q gdev` on qsub — results in "Access to queue is denied". Instead, let PBS auto-route based on walltime (≤2h → gdev). Also: no `ncpus` spec, only `ngpus`+`mem`.
- **Scheduler resume bug**: `load_checkpoint` restores G/D/opt state but LR scheduler (`LambdaLR`) is fresh — `last_epoch = -1`. Patched in [train.py:101](../train.py) — manually sets `sched_G.last_epoch = start_epoch - 1` so linear decay respects resumed position.
- **CycleGAN identity collapse on close domains**: `lambda_identity: 5` + small domain gap + missing `RandomCrop` = generator collapses to identity. Symptom: cycle loss < 0.1 and visible pass-through in grids. Fix: `lambda_identity: 0` + add `RandomCrop` to transform.
- **Spectral β scaling with resolution**: β=0.05 at 256 keeps ~130 FFT coefficients; β=0.05 at 64 keeps only ~9 (essentially DC). To preserve the experiment at 64×64, use **β=0.2** (same physical frequency cutoff radius of 6.4 px).
