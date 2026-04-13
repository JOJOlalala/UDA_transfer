#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --job-name=sanity
#SBATCH --output=/scratch-share/QIAO0042/models/acv/UDA_trans/logs/sanity_%j.out
#SBATCH --error=/scratch-share/QIAO0042/models/acv/UDA_trans/logs/sanity_%j.err

PYTHON=/home/msai/qiao0042/QIAO0042/.conda/envs/uda_cyclegan/bin/python
cd /scratch-share/QIAO0042/models/acv/UDA_trans

echo "=== Sanity: 5-epoch Pixel CycleGAN on MNIST->USPS ==="
echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

$PYTHON -u train.py --config configs/mnist_usps.yaml --mode pixel --epochs 5

echo "=== Done: $(date) ==="
