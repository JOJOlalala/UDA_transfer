# Project I — I2I Translation and Input Space UDA

## Task I: CycleGAN in Pixel Space vs. Spectral Space

This project compares two image style transfer approaches for unsupervised domain adaptation:

1. **Standard CycleGAN** — operates in pixel (spatial) space, translating full images between domains
2. **Spectral CycleGAN** — decomposes images via FFT, applies CycleGAN only to low-frequency bands, then recombines with original high-frequency content

### Project Structure

```
UDA_trans/
├── configs/              # Per-dataset YAML configs
├── data/
│   ├── download.py       # Dataset download helpers
│   └── datasets.py       # UnpairedDataset + transforms
├── models/
│   ├── generator.py      # ResNet generator
│   ├── discriminator.py  # PatchGAN discriminator
│   ├── cyclegan.py       # Pixel-space CycleGAN trainer
│   └── spectral_cyclegan.py  # Spectral variant
├── utils/
│   ├── image_pool.py     # Replay buffer
│   ├── losses.py         # LSGAN, cycle, identity losses
│   ├── spectral.py       # FFT decompose/recombine
│   └── visualization.py  # Image grids, loss plots
├── train.py              # Training entry point
├── test.py               # Inference / image generation
└── scripts/              # SLURM job scripts
```

### Setup

```bash
# Create conda environment
conda create -n uda_cyclegan python=3.10 -y
conda activate uda_cyclegan
pip install -r requirements.txt
```

### Datasets

| Pair | Source | Image Size |
|------|--------|-----------|
| MNIST -> USPS | torchvision | 32x32 |
| SVHN -> MNIST | torchvision | 32x32 |
| Amazon -> Webcam | Office-31 (manual download) | 256x256 |
| Photo -> Sketch | PACS (manual download) | 256x256 |

Download torchvision datasets:
```bash
python data/download.py
```

For Office-31 and PACS, download manually and place under:
- `data/raw/office31/{amazon,webcam}/`
- `data/raw/pacs/{photo,sketch}/`

### Training

```bash
# Pixel-space CycleGAN
python train.py --config configs/mnist_usps.yaml --mode pixel

# Spectral CycleGAN (beta controls frequency cutoff)
python train.py --config configs/mnist_usps.yaml --mode spectral --beta 0.05

# Resume from checkpoint
python train.py --config configs/mnist_usps.yaml --mode pixel --resume checkpoints/mnist_usps_pixel/ckpt_epoch0060.pth
```

### SLURM Submission

```bash
# Pixel CycleGAN on MNIST->USPS
sbatch scripts/train_pixel.sh mnist_usps

# Spectral CycleGAN on SVHN->MNIST with beta=0.05
sbatch scripts/train_spectral.sh svhn_mnist 0.05
```

### Inference

```bash
python test.py \
    --config configs/mnist_usps.yaml \
    --mode pixel \
    --checkpoint checkpoints/mnist_usps_pixel/ckpt_epoch0100.pth \
    --n_images 16
```

### Architecture Details

**Generator**: ResNet-based with reflection padding and InstanceNorm
- 256x256 images: 64 filters, 9 ResNet blocks, 2 downsample/upsample
- 32x32 images: 32 filters, 6 ResNet blocks, 1 downsample/upsample

**Discriminator**: PatchGAN with InstanceNorm and LeakyReLU
- 256x256 images: 64 filters, 3 layers (70x70 receptive field)
- 32x32 images: 32 filters, 2 layers

**Losses**: LSGAN adversarial + L1 cycle-consistency (weight=10) + L1 identity (weight=5)

**Spectral variant**: FFT -> circular low-pass mask (radius = beta * min(H,W) / 2) -> CycleGAN on low-freq spatial image -> recombine with original high-freq

### References

- [1] Zhu et al., "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks", ICCV 2017
- [2] Hoffman et al., "CyCADA: Cycle-Consistent Adversarial Domain Adaptation", ICML 2018
- [3] Yang et al., "FDA: Fourier Domain Adaptation for Semantic Segmentation", CVPR 2020
