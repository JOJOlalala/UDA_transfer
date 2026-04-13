# Task I Progress — CycleGAN Pixel vs. Spectral

**Last updated:** 2026-04-13
**Deadline:** April 17, 2026

## Status: Code Complete, Ready for NSCC Training

---

## Completed

### 1. Project Structure
Full project at `/scratch-share/QIAO0042/models/acv/UDA_trans/` (symlinked from home).

```
UDA_trans/
├── configs/         # 4 dataset YAML configs (absolute data paths)
├── data/            # dataset loaders + download helpers
│   └── raw/         # downloaded datasets (MNIST, USPS, PACS, Office-31)
├── models/          # generator, discriminator, CycleGAN trainers
├── utils/           # losses, image pool, spectral FFT, visualization
├── train.py         # unified entry: --mode pixel|spectral, wandb logging
├── test.py          # inference + FID image generation
├── scripts/         # SLURM job scripts
├── doc/             # INTRO.md (theory guide), PROGRESS.md (this file)
└── requirements.txt # includes wandb
```

### 2. Code Implementation
All modules written, CPU-sanity-tested, and GPU-verified (5-epoch run passed on TC2 L40S):

| Module | File | Notes |
|--------|------|-------|
| ResNet Generator | `models/generator.py` | Configurable: 32×32 (ngf=32, 6 blocks, 1 down) / 256×256 (ngf=64, 9 blocks, 2 down) |
| PatchGAN Discriminator | `models/discriminator.py` | Configurable n_layers/ndf |
| Pixel CycleGAN | `models/cyclegan.py` | LSGAN + cycle (λ=10) + identity (λ=5), image pool, LR decay |
| Spectral CycleGAN | `models/spectral_cyclegan.py` | FFT decompose → CycleGAN on low-freq → recombine with original high-freq |
| FFT utilities | `utils/spectral.py` | `fft_decompose(img, beta)` / `fft_recombine()`, verified lossless (err ~7e-7) |
| Losses | `utils/losses.py` | LSGAN D/G, cycle L1, identity L1 |
| Image pool | `utils/image_pool.py` | Size 50 replay buffer |
| Data pipeline | `data/datasets.py` | `UnpairedDataset`, auto-download for MNIST/USPS/SVHN, folder loader for Office-31/PACS |
| Visualization | `utils/visualization.py` | Comparison grids, spectral decomposition viz, loss curves |
| Training | `train.py` | `--mode pixel|spectral`, `--beta`, `--epochs` override, `--resume`, wandb logging (scalars + images) |
| Inference | `test.py` | Generate translated images, save for FID, spectral decomposition viz |

### 3. Datasets
| Dataset | Location | Status |
|---------|----------|--------|
| MNIST | `data/raw/MNIST/` | Downloaded |
| USPS | `data/raw/usps.bz2` | Downloaded |
| SVHN | auto-download on first use | Pending (torchvision handles it) |
| PACS photo | `data/raw/pacs/photo/` | 1,670 images |
| PACS sketch | `data/raw/pacs/sketch/` | 3,929 images |
| Office-31 amazon | `data/raw/office31/amazon/images/<class>/` | 2,817 images |
| Office-31 webcam | `data/raw/office31/webcam/images/<class>/` | 795 images |

### 4. GPU Sanity Run (TC2, L40S)
5-epoch pixel CycleGAN on MNIST→USPS — **PASSED**:
```
[Epoch 1/5] G=1.6261 D_A=0.1435 D_B=0.1537 cyc=0.0834 lr=0.000200 (336.7s)
[Epoch 2/5] G=1.1814 D_A=0.1549 D_B=0.1635 cyc=0.0561 lr=0.000200 (335.5s)
[Epoch 3/5] G=1.1756 D_A=0.1231 D_B=0.1452 cyc=0.0504 lr=0.000200 (339.1s)
[Epoch 4/5] G=1.2083 D_A=0.1019 D_B=0.1240 cyc=0.0478 lr=0.000200 (332.8s)
[Epoch 5/5] G=1.2128 D_A=0.0929 D_B=0.1074 cyc=0.0446 lr=0.000200 (336.3s)
```
- G loss converging, cycle loss decreasing, D losses stable (~0.1, not collapsing)
- ~336s/epoch for MNIST→USPS (60K samples, batch=4, 32×32)

### 5. Documentation
- `doc/INTRO.md` — full theory guide: CycleGAN, spectral variant, FFT math, PatchGAN, InstanceNorm, LSGAN
- `doc/PROGRESS.md` — this file

---

## TODO on NSCC

### Step 1: Environment Setup
```bash
# Create conda env on NSCC
conda create -n uda_cyclegan python=3.10 -y
conda activate uda_cyclegan
pip install -r requirements.txt
```

### Step 2: Copy/Sync Data
If data isn't accessible on NSCC, copy from TC2:
```bash
rsync -avz /scratch-share/QIAO0042/models/acv/UDA_trans/data/raw/ <nscc_path>/data/raw/
```
Then update `data_root` in all 4 configs to the NSCC absolute path.

### Step 3: Update SLURM Scripts for NSCC
- Change `--partition`, `--qos` to match NSCC cluster
- Update `PYTHON` path to NSCC conda env
- Update `cd` path to NSCC project directory
- With 4-8 A100s you may want to increase batch size (A100 has 80GB VRAM)

### Step 4: Submit Training (8 runs total)
```bash
# 4 datasets × 2 modes = 8 jobs
for DATASET in mnist_usps svhn_mnist amazon_webcam photo_sketch; do
    sbatch scripts/train_pixel.sh $DATASET
    sbatch scripts/train_spectral.sh $DATASET
done
```

### Step 5: After Training
1. Run `test.py` on each checkpoint to generate translated images
2. Compute FID: `python -m pytorch_fid <real_target_dir> <fake_target_dir>`
3. Collect comparison figures for report

### Step 6: Report
- Side-by-side image grids (Source | Pixel | Spectral | Target)
- FID table
- Analysis: which method wins on which dataset and why

---

## Config Summary

| Dataset | Epochs | Batch | Image Size | ngf | n_blocks | beta (spectral) |
|---------|--------|-------|-----------|-----|----------|-----------------|
| mnist_usps | 50 | 4 | 32×32 | 32 | 6 | 0.05 |
| svhn_mnist | 50 | 4 | 32×32 | 32 | 6 | 0.05 |
| amazon_webcam | 100 | 1 | 256×256 | 64 | 9 | 0.05 |
| photo_sketch | 100 | 1 | 256×256 | 64 | 9 | 0.05 |

---

## Known Issues / Notes
- **TC2 `conda run` bug**: `conda run -n uda_cyclegan` uses system Python 3.9 instead of env Python 3.10. Use full path to env Python directly. May or may not apply on NSCC — test first.
- **Config `data_root`**: must be absolute path (already fixed for TC2, needs update for NSCC).
- **Office-31 structure**: has extra `images/` subdir (`office31/amazon/images/<class>/`). `FolderImageDataset` uses `rglob` so this is handled.
- **Epoch timing**: 32×32 datasets ~336s/epoch on L40S. 256×256 will be slower per epoch but fewer iterations (smaller datasets). Adjust epochs to fit wall time.
- **wandb**: add `--no_wandb` flag if wandb isn't configured on NSCC. Otherwise `pip install wandb && wandb login` first.
